import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, accuracy, get_world_size, is_dist_avail_and_initialized)
import numpy as np
from .backbone import build_backbone
from .matcher import build_matcher

from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from pytorch_pretrained_bert.modeling import BertModel
from matplotlib import pyplot as plt
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import XLMTokenizer
from datasets.dataset import VGDataset
from util import box_ops

class VLTVG(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        """ Initializes the model."""
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.trans_encoder.d_model, kernel_size=1)

        bert_mode='bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        # Text feature encoder (BERT)
        self.bert = BertModel.from_pretrained(bert_mode)
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # visual grounding
        self.trans_decoder = build_vg_decoder(args)

        hidden_dim = self.trans_encoder.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)


    def forward(self, image, image_mask, word_id, word_mask, idx, target_dict, postprocessor, keys, keys_mask):

        N = image.size(0)
        dev = torch.device("cuda:1")
        image=image.to(dev)
        image_mask=image_mask.to(dev)
        # Image features
        features, pos = self.backbone(NestedTensor(image, image_mask))

        src, mask = features[-1].decompose()
        assert mask is not None
        from .feature_visualization import draw_feature_map
        img_feat, mask, pos_embed = self.trans_encoder(self.input_proj(src), mask, pos[-1])
        # Text features
        dev = torch.device("cuda:1")
        word_id=word_id.to(dev)

        word_mask=word_mask.to(dev)
        word_feat, _ = self.bert(word_id, token_type_ids=None, attention_mask=word_mask)
        each_feat = []
        word_id_0=keys.view(-1,1).to(torch.device('cuda:1'))
        word_mask_0=keys_mask.view(-1,1).to(torch.device('cuda:1'))
        each_feat = self.bert(word_id_0, token_type_ids=None, attention_mask=word_mask_0)[0]

        word_feat = torch.stack(word_feat[-self.bert_output_layers:], 1).mean(1)
        word_feat = self.bert_proj(word_feat)
        word_feat = word_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        word_mask = ~word_mask

        each_feat = torch.stack(each_feat[-self.bert_output_layers:], 1).mean(1)
        each_feat = self.bert_proj(each_feat)
        each_feat = each_feat.permute(1, 0, 2) # NxLxC -> LxNxC

        # Discriminative feature encoding + Multi-stage reasoning
        hs, kp = self.trans_decoder(img_feat, each_feat, word_mask_0, mask, pos_embed, word_feat, word_mask)
        outputs_coords = self.bbox_embed(hs).sigmoid()
        outputs_class = self.class_embed(hs)
        out = {'pred_boxes': outputs_coords[-1], 'pred_logits': outputs_class[-1]}
        results = postprocessor(out, target_dict)
        pred_all = []
        for i in range(len(results)):
            pred_all.append(np.array(results[i]['boxes'].cpu()).astype(int))
        
        acc_a=[]
        
        for img_bs in range(len(pred_all)):
            iou_a=[]
            for bboxs in range(pred_all[img_bs].shape[0]):
                iou_a.append(box_ops.box_pair_iou(torch.tensor(pred_all[img_bs][bboxs,:]).unsqueeze(0).to(torch.device('cuda:1')), target_dict['ori_bbox'][img_bs].unsqueeze(0).to(torch.device('cuda:1')))[0])
            acc_a.append(iou_a)
        
        kp_ = kp.view(-1,1,400)
        kp_position = torch.argmax(kp_, dim=2)
        kp_position = torch.cat([((kp_position%20)/20).unsqueeze(1),(kp_position/400).unsqueeze(1)],dim=1).squeeze()

        out['pred_scores'] = outputs_class[-1]
        out['kp'] = kp_position
        out['kp_o'] = kp
        out['acc_a'] = acc_a
        if self.training:
            out['aux_outputs'] = [{'pred_boxes': b, 'pred_logits': a} for b, a in zip(outputs_coords[:-1], outputs_class[:-1])]
        return out



class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes,
                         'loss_kp': self.loss_kp}

        self.loss_loc = self.loss_map[loss_loc]

        self.loss_keypoint = self.loss_map['loss_kp']
    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses
    def loss_kp(self, kp, gt_boxes):
        target_bbox = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        target_kp = torch.cat([(target_bbox[:,1]+target_bbox[:,3]).unsqueeze(1),(target_bbox[:,0]+target_bbox[:,2]).unsqueeze(1)],dim=1)/2
        losses= {}
        target_kp = target_kp.repeat_interleave(40,dim=0)
        nonZeroRows = kp.sum(dim=1) > 0.1
        loss_bbox = F.l1_loss(kp[nonZeroRows], target_kp[nonZeroRows], reduction='none')
        losses['kp'] = loss_bbox.sum() / 320
        return losses 
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']
        kp_position = outputs['kp']
        
        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        loss_kp = self.loss_keypoint(kp_position, gt_boxes)
        losses.update(loss)
        losses.update(loss_kp)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']
        out_logits = outputs['pred_logits']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., -1:].max(-1)
        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.device('cuda:1'))
        boxes = boxes * scale_fct[:, None, :]

        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1).to(torch.device('cuda:1'))[:, None, :]
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1).to(torch.device('cuda:1'))[:, None, :]

        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1).to(torch.device('cuda:1'))[:, None, :])
        results = [{'scores': s, 'boxes': b}for s, b in zip(scores, boxes)]
        return results


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def build_vgmodel(args):
    device = torch.device(args.device)

    model = VLTVG(pretrained_weights=args.load_weights_path, args=args)
    matcher = build_matcher(args)
    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef,'kp': 10}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor
