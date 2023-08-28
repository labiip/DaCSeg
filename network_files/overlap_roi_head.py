from typing import Optional, List, Dict, Tuple
import numpy

import matplotlib.pyplot as plt

import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align

from . import det_utils
from . import boxes as box_ops
from .con_net import build_reconstruction_head, mask_recon_loss, mask_recon_inference


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BCEFocalLoss(torch.nn.Module):
    """

    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)


    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing

    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]


    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)


    box_loss = det_utils.smooth_l1_loss(

        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """

    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]

    boxes_per_image = [label.shape[0] for label in labels]

    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)

    mask_prob = mask_prob[index, labels][:, None]

    mask_prob = mask_prob.split(boxes_per_image, dim=0)
    # print(mask_prob[0].shape)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """

    Args:
        mask_logits:
        proposals:
        gt_masks:
        gt_labels:
        mask_matched_idxs:

    Returns:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    # a = gt_masks[0].data.cpu().numpy()
    # plt.imshow(a[0,:,:])
    # plt.show()

    # match = mask_matched_idxs.tolist()

    discretization_size = mask_logits.shape[-1]

    # gt_mask_keshihua = gt_masks.numpy()
    # plt.imshow(gt_mask_keshihua[0])
    # plt.show()
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    # b = mask_targets[0].data.cpu().numpy()
    # plt.imshow(b[0,:,:])
    # plt.show()
    # mask_b = mask_targets.data.cpu().numpy()
    # numpy.save("output.npy",mask_b)



    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)



    # mask_b = mask_targets.data.cpu().numpy()
    # numpy.save("output.npy",mask_b)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0


    # a = mask_logits[torch.arange(labels.shape[0])]
    # b = mask_logits[torch.arange(labels.shape[0],device=labels.device), labels]
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )

    return mask_loss


def overlap_maskrcnn_loss(overlap_logits, overlap_boundary_logits, targets, mask_matched_idxs,proposals,overlapelse_logits, overlapelse_boundary_logits):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """

    Args:
        mask_logits:
        proposals:
        gt_masks:
        gt_labels:
        mask_matched_idxs:

    Returns:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = overlap_logits.shape[-1]

    # gt_masks = [t["masks"] for t in targets]
    gt_labels = [t["labels"] for t in targets]
    gt_overlaps = [t["overlaps"] for t in targets]
    gt_elses= [t["overlap_elses"] for t in targets]



    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    overlap_mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_overlaps, proposals, mask_matched_idxs)
    ]

    overlapelse_mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_elses, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    overlap_mask_targets = torch.cat(overlap_mask_targets, dim=0)
    overlapelse_mask_targets = torch.cat(overlapelse_mask_targets, dim=0)
    if overlap_mask_targets.numel() == 0:
        return overlap_logits.sum() * 0
    if overlapelse_mask_targets.numel() == 0:
        return overlapelse_logits.sum() * 0

    # overlap_mask_loss = F.binary_cross_entropy_with_logits(
    #     overlap_logits[torch.arange(labels.shape[0], device=labels.device), labels], overlap_mask_targets

    # )

    overlap_mask_loss = F.binary_cross_entropy_with_logits(
        overlap_logits[torch.arange(labels.shape[0], device=labels.device), labels], overlap_mask_targets

    )
    # alpha = 0.25
    # gamma = 2
    # pt1 = torch.exp(-overlap_maskBCE_loss)
    # F1_loss = alpha * (1 - pt1) ** gamma * overlap_maskBCE_loss
    # overlap_mask_loss = torch.mean(F1_loss)

    overlapelse_mask_loss = F.binary_cross_entropy_with_logits(
        overlapelse_logits[torch.arange(labels.shape[0], device=labels.device), labels], overlapelse_mask_targets

    )
    # pt2 = torch.exp(-overlapelse_maskBCE_loss)
    # F2_loss = alpha * (1 - pt2) ** gamma * overlapelse_maskBCE_loss
    # overlapelse_mask_loss = torch.mean(F2_loss)

    kernel_ = torch.FloatTensor([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)
    weight = torch.nn.Parameter(data=kernel_, requires_grad=False)
    weight = weight.cuda()

    overlap_mask_targets = overlap_mask_targets[:, None, :, :]
    overlapelse_mask_targets = overlapelse_mask_targets[:, None, :, :]

    overlap_edge_gt = F.conv2d(overlap_mask_targets, weight, padding=1).clamp(min=0)  #
    overlapelse_edge_gt = F.conv2d(overlapelse_mask_targets, weight, padding=1).clamp(min=0)  #

    overlap_edge_gt_new = torch.zeros_like(overlap_edge_gt)
    overlapelse_edge_gt_new = torch.zeros_like(overlapelse_edge_gt)
    # edge_gt_id = ((edge_gt > 99) & (edge_gt < 104)) #
    overlap_edge_gt_id = (overlap_edge_gt > 0.1)
    overlapelse_edge_gt_id = (overlapelse_edge_gt > 0.1)
    overlap_edge_gt_new[overlap_edge_gt_id] = 1
    overlapelse_edge_gt_new[overlapelse_edge_gt_id] = 1


    overlap_edge_gt_w_id = (overlap_edge_gt_new == 1)  # True or False
    overlapelse_edge_gt_w_id = (overlapelse_edge_gt_new == 1)

    overlap_edge_gt_w = torch.ones_like(overlap_edge_gt)
    overlap_edge_gt_w[overlap_edge_gt_w_id] = 5
    overlap_edge_gt_w = torch.squeeze(overlap_edge_gt_w)
    # img5 = edge_gt_w.cpu().numpy()
    if overlap_edge_gt_new.numel() == 0:
        return overlap_edge_gt_new.sum() * 0
    overlap_edge_gt_new = torch.squeeze(overlap_edge_gt_new)
    # a = overlap_mask_targets[0].data.cpu().numpy()
    # plt.imshow(a[0,:,:])
    # plt.show()
    # c = torch.squeeze(overlap_edge_gt_new).data.cpu().numpy()
    # plt.imshow(c[0,:,:])
    # plt.show()

    overlapelse_edge_gt_w = torch.ones_like(overlapelse_edge_gt)
    overlapelse_edge_gt_w[overlapelse_edge_gt_w_id] = 5
    overlapelse_edge_gt_w = torch.squeeze(overlapelse_edge_gt_new)
    # img5 = edge_gt_w.cpu().numpy()
    if overlapelse_edge_gt_new.numel() == 0:
        return overlapelse_edge_gt_new.sum() * 0
    overlapelse_edge_gt_new = torch.squeeze(overlapelse_edge_gt_new)
    # b = overlapelse_mask_targets[0].data.cpu().numpy()
    # plt.imshow(b[0,:,:])
    # plt.show()
    # d = torch.squeeze(overlapelse_edge_gt_new).data.cpu().numpy()
    # plt.imshow(d[0,:,:])
    # plt.show()


    # a = overlap_boundary_logits[torch.arange(labels.shape[0])]
    # b = overlap_boundary_logits[torch.arange(labels.shape[0],device=labels.device), labels]
    overlap_edge_loss = F.binary_cross_entropy_with_logits(
        overlap_boundary_logits[torch.arange(labels.shape[0], device=labels.device), labels], overlap_edge_gt_new

    )
    # pt3 = torch.exp(-overlap_maskBCE_loss)
    # F3_loss = alpha * (1 - pt3) ** gamma * overlap_edgeBCE_loss
    # overlap_edge_loss = torch.mean(F3_loss)

    overlapelse_edge_loss = F.binary_cross_entropy_with_logits(
        overlapelse_boundary_logits[torch.arange(labels.shape[0], device=labels.device), labels], overlapelse_edge_gt_new

    )
    # pt4 = torch.exp(-overlapelse_edgeBCE_loss)
    # F4_loss = alpha * (1 - pt4) ** gamma * overlapelse_edgeBCE_loss
    # overlapelse_edge_loss = torch.mean(F4_loss)

    return overlap_mask_loss, overlapelse_mask_loss, overlap_edge_loss ,overlapelse_edge_loss





def overlap_cls_loss(overlap_cls_logits, proposals, gt_overlaps, mask_matched_idxs):
    # match = mask_matched_idxs.tolist()

    discretization_size = 28

    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_overlaps, proposals, mask_matched_idxs)
    ]
    mask_targets = torch.cat(mask_targets, dim=0)
    b = mask_targets.size(0)
    cls_labels = []
    for i in range (0, b):
        if mask_targets[i].sum() == 0:
            cls_label = 0
        else:
            cls_label = 1
        cls_labels.append(cls_label)
    cls_labels = torch.tensor(cls_labels, dtype=torch.int64)
    cls_labels = cls_labels.to(device=torch.device("cuda"if torch.cuda.is_available() else 'cpu'))

    overlap_loss = F.cross_entropy(overlap_cls_logits, cls_labels)
    loss_dict.update(overlap_loss)

    # stride: 64,32,16,8,4 -> 4, 8, 16, 32
    fpn_fms = fpn_fms[1:][::-1]
    stride = [4, 8, 16, 32]
    pool_features, rcnn_rois, labels, bbox_targets = roi_pool(
        fpn_fms, rcnn_rois, stride, (7, 7), 'roi_align',
        labels, bbox_targets)
    flatten_feature = F.flatten(pool_features, start_axis=1)
    roi_feature = F.relu(self.fc1(flatten_feature))
    roi_feature = F.relu(self.fc2(roi_feature))
    pred_emd_pred_cls_0 = self.emd_pred_cls_0(roi_feature)
    pred_emd_pred_delta_0 = self.emd_pred_delta_0(roi_feature)
    pred_emd_pred_cls_1 = self.emd_pred_cls_1(roi_feature)
    pred_emd_pred_delta_1 = self.emd_pred_delta_1(roi_feature)
    if self.training:
        loss0 = emd_loss(
            pred_emd_pred_delta_0, pred_emd_pred_cls_0,
            pred_emd_pred_delta_1, pred_emd_pred_cls_1,
            bbox_targets, labels)
        loss1 = emd_loss(
            pred_emd_pred_delta_1, pred_emd_pred_cls_1,
            pred_emd_pred_delta_0, pred_emd_pred_cls_0,
            bbox_targets, labels)
        loss = F.concat([loss0, loss1], axis=1)
        indices = F.argmin(loss, axis=1)
        loss_emd = F.indexing_one_hot(loss, indices, 1)
        loss_emd = loss_emd.sum() / loss_emd.shapeof()[0]
        loss_dict = {}
        loss_dict['loss_rcnn_emd'] = loss_emd
        return loss_dict
    else:
        pred_scores_0 = F.softmax(pred_emd_pred_cls_0)[:, 1:].reshape(-1, 1)
        pred_scores_1 = F.softmax(pred_emd_pred_cls_1)[:, 1:].reshape(-1, 1)
        pred_delta_0 = pred_emd_pred_delta_0[:, 4:].reshape(-1, 4)
        pred_delta_1 = pred_emd_pred_delta_1[:, 4:].reshape(-1, 4)
        target_shape = (rcnn_rois.shapeof()[0], config.num_classes - 1, 4)
        base_rois = F.add_axis(rcnn_rois[:, 1:5], 1).broadcast(target_shape).reshape(-1, 4)
        pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
        pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)
        pred_bbox_0 = F.concat([pred_bbox_0, pred_scores_0], axis=1)
        pred_bbox_1 = F.concat([pred_bbox_1, pred_scores_1], axis=1)
        # [{head0, pre1, tag1}, {head1, pre1, tag1}, {head0, pre1, tag2}, ...]
        pred_bbox = F.concat((pred_bbox_0, pred_bbox_1), axis=1).reshape(-1, 5)
        return pred_bbox
    #
    # return overlap_loss



def emd_loss(p_b0, p_c0, p_b1, p_c1, targets, labels):
    pred_box = F.concat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shapeof()[-1])
    pred_box = pred_box.reshape(-1, config.num_classes, 4)
    pred_score = F.concat([p_c0, p_c1], axis=1).reshape(-1, p_c0.shapeof()[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.reshape(-1).astype(np.int32)
    fg_masks = F.greater(labels, 0)
    non_ignore_masks = F.greater_equal(labels, 0)
    # mulitple class to one
    indexing_label = (labels * fg_masks).reshape(-1,1)
    indexing_label = indexing_label.broadcast((labels.shapeof()[0], 4))
    pred_box = F.indexing_one_hot(pred_box, indexing_label, 1)
    # loss for regression
    loss_box_reg = smooth_l1_loss(
        pred_box,
        targets,
        config.rcnn_smooth_l1_beta)
    # loss for classification
    loss_cls = softmax_loss(pred_score, labels)
    loss = loss_cls*non_ignore_masks + loss_box_reg * fg_masks
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = mge.tensor(config.bbox_normalize_stds[None, :])
        mean_opr = mge.tensor(config.bbox_normalize_means[None, :])
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox



class OverlapRoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img,   # default: 100
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 overlap_mask_head=None,
                 overlap_cls_head=None,
                 ):
        super(OverlapRoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.overlap_mask_head = overlap_mask_head
        self.overlap_cls_head = overlap_cls_head


    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        # if self.overlap_mask_head is None:
        #     return False
        # if self.mask_predictor is None:
        #     return False
        # if self.overlap_cls_head is None:
        #     return False
        # if self.overlap_cls_predictor is None:
        #     return False
        return True



    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """

        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands

                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)



                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)


                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)

                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)

                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):

            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]

        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]

        self.check_targets(targets)
        if targets is None:
            raise ValueError("target should not be None.")

        dtype = proposals[0].dtype
        device = proposals[0].device


        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal

        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals

        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)


        for img_id in range(num_images):

            img_sampled_inds = sampled_inds[img_id]

            proposals[img_id] = proposals[img_id][img_sampled_inds]

            labels[img_id] = labels[img_id][img_sampled_inds]

            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])


        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]

        device = class_logits.device

        num_classes = class_logits.shape[-1]


        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_boxes = self.box_coder.decode(box_regression, proposals)


        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label

            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes

            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes

            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class

            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep1 = box_ops.soft_nms_pytorch(boxes, scores)
            # keep2 = box_ops.soft_nms(boxes,scores)

            # keep only topk scoring predictions

            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # box_keep = box_keep[:self.detection_per_img]
            # boxes, scores, labels = boxes[box_keep], scores[box_keep], labels[box_keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """


        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:

            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None


        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)


        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)


        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )



        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:

                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features_overlap = self.mask_roi_pool(features,mask_proposals,image_shapes)
            mask_features_cls = self.mask_roi_pool(features, mask_proposals, image_shapes)
            overlap_cls_logits = self.overlap_cls_head(mask_features_cls)
            overlap_logits, overlap_boundary_logits,overlapelse_logits, overlapelse_boundary_logits, x_feature = self.overlap_mask_head(mask_features_overlap)

            mask_features = mask_features + x_feature
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)# [82,2,28,28]


            # mask_logits_down = F.interpolate(mask_logits,scale_factor=0.5)
            # overlapelse_logits_down = F.interpolate(overlapelse_logits,scale_factor=0.5)
            # mask_cat = torch.cat([mask_logits_down,overlapelse_logits_down],1)
            # model1 = torch.nn.Conv2d(kernel_size=3,in_channels=4,out_channels=256)
            # model1.cuda()
            # mask_logits_1 = model1(mask_cat)
            # mask_logits_gai = self.mask_predictor(mask_logits_1)





            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or overlap_logits is None :
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                gt_overlaps = [t["overlaps"] for t in targets]

                if not os.path.exists("{}_codebook.npy".format(dataset_name)):
                    if not self.SPRef:
                        loss_recon = 0
                        for i in range(len(mask_logits)):
                            loss_recon += mask_recon_loss(mask_logits[i][0], proposals, self.recon_net, self.targets,
                                                          mask_ths=self.recon_mask_ths, iter=self.iter)
                        loss_mask.update({"loss_recon": loss_recon / len(mask_logits)})

                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask":rcnn_loss_mask}
                rcnn_loss_cls = overlap_cls_loss(overlap_cls_logits, mask_proposals, gt_overlaps, pos_matched_idxs)
                overlap_mask_loss, overlapelse_mask_loss, overlap_edge_loss ,overlapelse_edge_loss = \
                    overlap_maskrcnn_loss(overlap_logits, overlap_boundary_logits,targets, pos_matched_idxs,mask_proposals,overlapelse_logits,
                                          overlapelse_boundary_logits)


                loss_mask = {"loss_mask":rcnn_loss_mask, "loss_cls":rcnn_loss_cls,"loss_overlap": overlap_mask_loss,
                             "loss_overlapelse":overlapelse_mask_loss,"loss_overlap_edge":overlap_edge_loss,"loss_overlapelse_edge":overlapelse_edge_loss}

            else:
                labels = [r["labels"] for r in result]
                mask_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(mask_probs, result):
                    r["masks"] = mask_prob
            #
                pred_overlap_scores = F.softmax(overlap_cls_logits, -1)
                overlap_inds = pred_overlap_scores.argmax(-1)

                # inds = torch.where(torch.gt(pred_overlap_scores, 0.5))[0]
                # overlap_proposals = [mask_proposals[ind] for ind in overlap_inds]
                if overlap_inds.any()==0:
                    pass
                else:
                    overlap_inds = torch.nonzero(overlap_inds)
                    inds = overlap_inds.squeeze()
                    overlap_proposals = [torch.stack([mask_proposals[0][i,:] for i in inds])]
                    for overlap_proposal, r in zip(overlap_proposals, result):
                        r["overlap_proposals"] = overlap_proposal
                    overlape_features = self.mask_roi_pool(features,overlap_proposals,image_shapes)
                    overlap_pred, overlap_boundary_pred, overlapelse_pred, overlapelse_boundary_pred , mask_features  = self.overlap_mask_head(
                        overlape_features)

                    overlap_labels = [torch.stack([labels[0][ind] for ind in inds])]

                    overlap_preds = maskrcnn_inference(overlap_pred, overlap_labels)
                    overlap_boundary_preds = maskrcnn_inference(overlap_boundary_pred,overlap_labels)
                    overlapelse_preds = maskrcnn_inference(overlapelse_pred,overlap_labels)
                    overlapelse_boundary_preds = maskrcnn_inference(overlapelse_boundary_pred,overlap_labels)
                    for overlap_prob, r in zip(overlap_preds, result):
                        r["overlap_prob"] = overlap_prob
                    for overlap_boundary_prob, r in zip(overlap_boundary_preds, result):
                        r["overlap_boundary_prob"] = overlap_boundary_prob
                    for overlapelse_prob, r in zip(overlapelse_preds, result):
                        r["overlapelse_prob"] = overlapelse_prob
                    for overlapelse_boundary_prob, r in zip(overlapelse_boundary_preds, result):
                        r["overlapelse_boundary_prob"] = overlapelse_boundary_prob



            losses.update(loss_mask)

        return result, losses



