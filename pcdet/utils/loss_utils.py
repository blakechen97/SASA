import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.code_weights = code_weights
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        self.code_weights = code_weights
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').mean(dim=-1) * weights
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


class PointSASALoss(nn.Module):
    """
    Layer-wise point segmentation loss, used for SASA.
    """
    def __init__(self,
                 func: str = 'BCE',
                 layer_weights: list = None,
                 extra_width: list = None,
                 set_ignore_flag: bool = False):
        super(PointSASALoss, self).__init__()

        self.layer_weights = layer_weights
        if func == 'BCE':
            self.loss_func = WeightedBinaryCrossEntropyLoss()
        elif func == 'Focal':
            self.loss_func = SigmoidFocalClassificationLoss()
        else:
            raise NotImplementedError

        assert not set_ignore_flag or (set_ignore_flag and extra_width is not None)
        self.extra_width = extra_width
        self.set_ignore_flag = set_ignore_flag
    
    def assign_target(self, points, gt_boxes):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, \
            'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if self.extra_width is not None else gt_boxes
        
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())

            if not self.set_ignore_flag:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

            else:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                ignore_flag = box_fg_flag ^ (extend_box_idx_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            point_cls_labels_single[box_fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single
        
        return point_cls_labels # (N, ) 0: bg, 1: fg, -1: ignore

    def forward(self, l_points, l_scores, gt_boxes):
        """
        Args:
            l_points: List of points, [(N, 4): bs_idx, x, y, z]
            l_scores: List of points, [(N, 1): predicted point scores]
            gt_boxes: (B, M, 8)
        Returns:
            l_labels: List of labels: [(N, 1): assigned segmentation labels]
        """
        l_labels = []
        for i in range(len(self.layer_weights)):
            li_scores = l_scores[i]
            if li_scores is None or self.layer_weights[i] == 0:
                l_labels.append(None)
                continue
            # binary segmentation labels: 0: bg, 1: fg, -1: ignore
            li_labels = self.assign_target(l_points[i], gt_boxes)
            l_labels.append(li_labels)

        return l_labels

    def loss_forward(self, l_scores, l_labels):
        """
        Args:
            l_scores: List of points, [(N, 1): predicted point scores]
            l_labels: List of points, [(N, 1): assigned segmentation labels]
        Returns:
            l_loss: List of segmentation loss
        """
        l_loss = []
        for i in range(len(self.layer_weights)):
            li_scores, li_labels = l_scores[i], l_labels[i]
            if li_scores is None or li_labels is None:
                l_loss.append(None)
                continue

            positives, negatives = li_labels > 0, li_labels == 0
            cls_weights = positives * 1.0 + negatives * 1.0 # (N, 1)
            pos_normalizer = cls_weights.sum(dim=0).float()

            one_hot_targets = li_scores.new_zeros(
                *list(li_labels.shape), 2
            )
            one_hot_targets.scatter_(-1, (li_labels > 0).long().unsqueeze(-1), 1.0)
            one_hot_targets = one_hot_targets[:, 1:] # (N, 1)

            li_loss = self.loss_func(li_scores[None],
                                     one_hot_targets[None],
                                     cls_weights.reshape(1, -1))
            li_loss = self.layer_weights[i] * li_loss.sum() / torch.clamp(
                pos_normalizer, min=1.0)
            l_loss.append(li_loss)

        return l_loss
