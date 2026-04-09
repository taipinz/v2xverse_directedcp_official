# -*- coding: utf-8 -*-
"""
Direction-Weighted Point Pillar Loss for Directed-CP

This loss function implements the Direction-Weighted Detection Loss (DWLoss)

Configuration:
    dir:
        mode: 'N'  # Non-independent mode (coupled weighting)
        weight: [0.9, 0.9, 0.1, 0.1]  # Per-direction importance weights
        dsigma: 1.0  # Weight normalization factor
        th: 0.1  # Direction activation threshold

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss for gaussian heatmaps."""

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * gaussian_focal_loss(
            pred, target, weight, alpha=self.alpha, gamma=self.gamma,
            reduction=reduction, avg_factor=avg_factor)
        return loss


def gaussian_focal_loss(pred, gaussian_target, weight=None, alpha=2.0, gamma=4.0,
                        reduction='mean', avg_factor=None):
    eps = 1e-12
    device = pred.device
    pos_weights = gaussian_target.eq(1).to(device)
    neg_weights = (1 - gaussian_target).pow(gamma).to(device)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    loss = pos_loss + neg_loss
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(torch.sigmoid(x), min=eps, max=1 - eps)
    return y


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device, torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    if reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        weight = weight.to(loss.device)
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def l1_loss(pred, target):
    device = pred.device
    target = target.to(device)
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

class DirectionWeightedPointPillarLoss(nn.Module):
    def __init__(self, args):
        super(DirectionWeightedPointPillarLoss, self).__init__()
        self.is_centerpoint = 'target_assigner_config' in args or 'cls_weight' in args

        # PointPillar-style args
        self.pos_cls_weight = args.get('pos_cls_weight', None)
        self.cls = args.get('cls', None)
        self.reg = args.get('reg', None)

        # Directed-CP args
        self.dir = args.get('dir', None)

        # CenterPoint-style args (for multiclass)
        if self.is_centerpoint:
            self.cls_weight = args['cls_weight']
            self.loc_weight = args['loc_weight']
            self.code_weights = args['code_weights']
            self.target_cfg = args['target_assigner_config']
            self.lidar_range = self.target_cfg['cav_lidar_range']
            self.voxel_size = self.target_cfg['voxel_size']
            self.loss_cls = GaussianFocalLoss(reduction='mean')
        
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if self.is_centerpoint or f'object_bbx_center{suffix}' in target_dict:
            return self._forward_centerpoint_multiclass(output_dict, target_dict, suffix)

        return self._forward_point_pillar(output_dict, target_dict, suffix)

    def _forward_point_pillar(self, output_dict, target_dict, suffix=""):
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        # Partition size for directional quadrants (dynamic)
        pos_equal_one = target_dict['pos_equal_one']
        H, W = pos_equal_one.shape[1:3]
        h_mid = H // 2
        w_mid = W // 2
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        direction_weights = self._compute_direction_weights()

        for i in range(2):
            for j in range(2):
                # Calculate sub-matrix indices for directional regions
                start_h = 0 if i == 0 else h_mid
                end_h = h_mid if i == 0 else H
                start_w = 0 if j == 0 else w_mid
                end_w = w_mid if j == 0 else W

                # Extract directional sub-matrices
                cls_labls = target_dict['pos_equal_one'][:, start_h:end_h, start_w:end_w, :].reshape(batch_size, -1, 1)
                positives = cls_labls > 0
                negatives = target_dict['neg_equal_one'][:, start_h:end_h, start_w:end_w, :].reshape(batch_size, -1, 1) > 0
                pos_normalizer = positives.sum(1, keepdim=True).float()

                # Rename prediction keys for compatibility
                if f'psm{suffix}' in output_dict:
                    output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
                if f'rm{suffix}' in output_dict:
                    output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']

                # cls loss
                cls_preds = output_dict[f'cls_preds{suffix}'][:,:, start_h:end_h, start_w:end_w] \
                            .permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 1)
                cls_weights = positives * self.pos_cls_weight + negatives * 1.0
                cls_weights /= torch.clamp(pos_normalizer, min=1.0)
                cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
                sub_cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

                # reg loss
                reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
                reg_preds = output_dict[f'reg_preds{suffix}'][:, :, start_h:end_h, start_w:end_w] \
                            .permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 7)
                reg_targets = target_dict['targets'][:, start_h:end_h, start_w:end_w, :] \
                            .reshape(batch_size, -1, 7)
                reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
                reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
                sub_reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

                total_reg_loss += sub_reg_loss
                total_cls_loss += sub_cls_loss
                sub_total_loss = sub_reg_loss + sub_cls_loss
                weight = direction_weights[i * 2 + j]
                total_loss += sub_total_loss * weight

        self.loss_dict.update({'total_loss': total_loss.item(),
                            'reg_loss': total_reg_loss.item(),
                            'cls_loss': total_cls_loss.item()})

        return total_loss

    def _forward_centerpoint_multiclass(self, output_dict, target_dict, suffix=""):
        """
        CenterPoint-style multiclass loss with direction weighting.
        """
        # Predictions
        box_preds = output_dict['bbox_preds{}'.format(suffix)].permute(0, 2, 3, 1).contiguous()
        cls_raw = output_dict['cls_preds{}'.format(suffix)]
        cls_preds = clip_sigmoid(cls_raw)

        # GTs
        bbox_center_all = target_dict['object_bbx_center{}'.format(suffix)]
        bbox_mask_all = target_dict['object_bbx_mask{}'.format(suffix)]

        if bbox_center_all.dim() == 3:
            bbox_center_all = bbox_center_all.unsqueeze(1)
            bbox_mask_all = bbox_mask_all.unsqueeze(1)

        bbox_center_all = bbox_center_all.cpu().numpy()
        bbox_mask_all = bbox_mask_all.cpu().numpy()

        batch_size = bbox_mask_all.shape[0]
        num_class = bbox_center_all.shape[1]

        cls_gt_list = []
        box_gt_list = []
        for i in range(num_class):
            bbox_center = bbox_center_all[:, i, :, :]
            bbox_mask = bbox_mask_all[:, i, :]

            max_gt = int(max(bbox_mask.sum(axis=1)))
            gt_boxes3d = np.zeros((batch_size, max_gt, bbox_center[0].shape[-1]), dtype=np.float32)
            for k in range(batch_size):
                if bbox_mask[k].sum() > 0:
                    gt_boxes3d[k, :int(bbox_mask[k].sum()), :] = bbox_center[k, :int(bbox_mask[k].sum()), :]
            gt_boxes3d = torch.from_numpy(gt_boxes3d).to(box_preds.device)

            targets_dict = self.assign_targets(gt_boxes=gt_boxes3d)
            cls_gt_list.append(targets_dict['heatmaps'])  # [B, 1, H, W]
            box_gt_list.append((targets_dict['anno_boxes'], targets_dict['inds'], targets_dict['masks']))

        cls_gt = torch.stack(cls_gt_list, dim=1)  # [B, C, 1, H, W]
        cls_preds = cls_preds.unsqueeze(2)  # [B, C, 1, H, W]

        # reshape box preds to [B, H, W, C, 8]
        box_preds = box_preds.view(box_preds.shape[0], box_preds.shape[1], box_preds.shape[2],
                                   int(box_preds.shape[3] / 8), 8)

        _, _, _, H, W = cls_gt.shape
        h_mid = H // 2
        w_mid = W // 2

        direction_weights = self._compute_direction_weights()

        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        quad_bounds = [
            (0, h_mid, 0, w_mid),
            (0, h_mid, w_mid, W),
            (h_mid, H, 0, w_mid),
            (h_mid, H, w_mid, W),
        ]

        for q_idx, (start_h, end_h, start_w, end_w) in enumerate(quad_bounds):
            # cls loss per quadrant
            cls_pred_q = cls_preds[..., start_h:end_h, start_w:end_w]
            cls_gt_q = cls_gt[..., start_h:end_h, start_w:end_w]
            num_pos = cls_gt_q.eq(1).float().sum().item()
            cls_loss_q = self.loss_cls(cls_pred_q, cls_gt_q, avg_factor=max(num_pos, 1))
            cls_loss_q = cls_loss_q * self.cls_weight

            # reg loss per quadrant
            reg_loss_q = 0
            for i in range(num_class):
                anno_boxes, inds, masks = box_gt_list[i]
                quad_mask = self._build_quadrant_mask(inds, masks, W, start_h, end_h, start_w, end_w)
                reg_loss_q += self.get_box_reg_layer_loss(box_preds[:, :, :, i, :], (anno_boxes, inds, quad_mask))

            total_reg_loss += reg_loss_q
            total_cls_loss += cls_loss_q
            total_loss += (reg_loss_q + cls_loss_q) * direction_weights[q_idx]

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': total_reg_loss.item(),
                               'cls_loss': total_cls_loss.item()})

        return total_loss

    @staticmethod
    def _build_quadrant_mask(inds, masks, W, start_h, end_h, start_w, end_w):
        if inds.numel() == 0:
            return masks
        y = torch.div(inds, W, rounding_mode='floor')
        x = inds - y * W
        quad = (x >= start_w) & (x < end_w) & (y >= start_h) & (y < end_h)
        quad = quad & masks.bool()
        return quad.to(masks.dtype)

    def _compute_direction_weights(self):
        if self.dir is None:
            return [0.25, 0.25, 0.25, 0.25]
        base_weights = self.dir.get('weight', [1.0, 1.0, 1.0, 1.0])
        if self.dir.get('mode', 'N') == 'N':
            total_weight = sum(base_weights)
            normalized_weights = [w / total_weight for w in base_weights]
            thresholded_weights = [1 if w > self.dir['th'] else 0 for w in normalized_weights]
            added_weights = [w + self.dir['dsigma'] for w in thresholded_weights]
            total_added_weight = sum(added_weights)
            return [w / total_added_weight for w in added_weights]
        total_weight = sum(base_weights)
        return [w / total_weight for w in base_weights]


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

   

    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f " % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)

    # ===== CenterPoint helpers =====
    def get_cls_layer_loss(self, pred_heatmaps, gt_heatmaps):
        num_pos = gt_heatmaps.eq(1).float().sum().item()
        cls_loss = self.loss_cls(
            pred_heatmaps,
            gt_heatmaps,
            avg_factor=max(num_pos, 1))
        cls_loss = cls_loss * self.cls_weight
        return cls_loss

    def _gather_feat(self, feat, ind, mask=None):
        device = feat.device
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind.to(device))
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_box_reg_layer_loss(self, bbox_preds, bbox_gt):
        target_box, inds, masks = bbox_gt
        pred = bbox_preds
        num = masks.float().sum()
        pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = self._gather_feat(pred, inds)
        mask = masks.unsqueeze(2).expand_as(target_box).float()
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = self.code_weights
        bbox_weights = mask * mask.new_tensor(code_weights)
        loc_loss = l1_loss(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
        loc_loss = loc_loss * self.loc_weight
        return loc_loss

    def assign_targets(self, gt_boxes):
        if gt_boxes.shape[-1] == 8:
            gt_bboxes_3d, gt_labels_3d = gt_boxes[..., :-1], gt_boxes[..., -1]
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d, gt_labels_3d)
        elif gt_boxes.shape[-1] == 7:
            gt_bboxes_3d = gt_boxes
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d)

        all_targets_dict = {
            'heatmaps': heatmaps,
            'anno_boxes': anno_boxes,
            'inds': inds,
            'masks': masks
        }
        return all_targets_dict

    def get_targets_single(self, gt_bbox_3d, gt_labels_3d=None):
        batch_size = gt_bbox_3d.shape[0]
        device = gt_bbox_3d.device
        max_objs = self.target_cfg['max_objs']
        pc_range = self.lidar_range
        voxel_size = self.voxel_size

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        feature_map_size = grid_size[:2] // self.target_cfg['out_size_factor']

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for batch in range(batch_size):
            task_boxes = gt_bbox_3d[batch, :, :]
            if not gt_labels_3d is None:
                task_classes = gt_labels_3d[batch, :]

            heatmap = gt_bbox_3d.new_zeros(
                (1, feature_map_size[1], feature_map_size[0]))

            anno_box = gt_bbox_3d.new_zeros((max_objs, 8), dtype=torch.float32)
            ind = gt_bbox_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bbox_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes.shape[0], max_objs)

            for k in range(num_objs):
                coor_x = (task_boxes[k][0] - pc_range[0]) / voxel_size[0] / self.target_cfg['out_size_factor']
                coor_y = (task_boxes[k][1] - pc_range[1]) / voxel_size[1] / self.target_cfg['out_size_factor']
                coor_z = (task_boxes[k][2] - pc_range[2]) / voxel_size[2] / self.target_cfg['out_size_factor']
                h = task_boxes[k][3] / voxel_size[0] / self.target_cfg['out_size_factor']
                w = task_boxes[k][4] / voxel_size[1] / self.target_cfg['out_size_factor']
                l = task_boxes[k][5] / voxel_size[2] / self.target_cfg['out_size_factor']
                rot = task_boxes[k][6]

                if h > 0 and w > 0:
                    radius = gaussian_radius(
                        (h, w),
                        min_overlap=self.target_cfg['gaussian_overlap'])
                    radius = max(self.target_cfg['min_radius'], int(radius))

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    if not (0 <= center_int[0] < feature_map_size[0].item()
                            and 0 <= center_int[1] < feature_map_size[1].item()):
                        continue

                    draw_gaussian(heatmap[0], center_int, radius)

                    x, y = center_int[0], center_int[1]
                    ind[k] = y * feature_map_size[0] + x
                    mask[k] = 1
                    box_dim = torch.cat([h.unsqueeze(0), w.unsqueeze(0), l.unsqueeze(0)], dim=0)
                    anno_box[k] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        coor_z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)

        heatmaps = torch.stack(heatmaps)
        anno_boxes = torch.stack(anno_boxes)
        inds = torch.stack(inds)
        masks = torch.stack(masks)
        return heatmaps, anno_boxes, inds, masks

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss
