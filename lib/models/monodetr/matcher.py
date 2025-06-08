# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_cxcylrtb_to_xyxy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 2]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            # C_g.split(sizes, -1)[i][i][indices_g[i][0],indices_g[i][1]]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                     np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                indices], C_g.split(sizes, -1)


class HungarianMatcherV2(HungarianMatcher):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        """
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 6]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [batch_size * num_target_boxes, 6]
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the confidence cost
        out_prob_max = out_prob.max(dim=-1)[0]
        cost_confidence = -out_prob_max.unsqueeze(1).expand(-1, cost_giou.shape[1])

        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_giou * cost_giou + cost_confidence
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes_3d"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return the matching indices for predictions and targets
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], [
            c[i] for i, c in enumerate(C.split(sizes, -1))]


def build_matcher(cfg):
    return HungarianMatcher(
        cost_class=cfg['set_cost_class'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_giou=cfg['set_cost_giou'])


def build_matcher_v2(cfg):
    return HungarianMatcherV2(
        cost_class=cfg['set_cost_class'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_giou=cfg['set_cost_giou'])


def filter_high_cost_pairs_and_sort(indices, C_splits, cost_threshold):
    """
    Filters out high-cost prediction-target pairs based on the given cost threshold and sorts the remaining pairs
    by the order of target indices (index_j).
    
    Params:
        indices: A list of tuples (index_i, index_j), where:
            - index_i is a tensor of indices for selected predictions
            - index_j is a tensor of indices for the corresponding selected targets
        C_splits: A list of tensors, each representing the cost matrix for a batch element split by target sizes
        cost_threshold: The threshold above which the cost is considered too high

    Returns:
        sorted_filtered_indices: A list of tuples (filtered_sorted_index_i, filtered_sorted_index_j) containing pairs with cost below the threshold, sorted by index_j.
        sorted_masks: A list of boolean tensors, each indicating whether a sorted pair is below the cost threshold (True) or not (False), sorted by index_j.
    """
    sorted_indices = []
    sorted_filtered_indices = []
    sorted_masks = []

    for idx, (index_i, index_j) in enumerate(indices):
        C = C_splits[idx]  # Get the cost matrix for the current batch element
        costs = C[index_i, index_j]  # Extract the costs for the matched indices

        # Sort index_j and associated index_i and costs
        sorted_index_j = index_j.argsort()
        sorted_index_i = index_i[sorted_index_j]
        sorted_index_j = index_j[sorted_index_j]
        sorted_costs = costs[sorted_index_j]

        sorted_indices.append((sorted_index_i.tolist(), sorted_index_j.tolist()))

        # Determine which sorted costs are below the threshold
        mask = -sorted_costs >= cost_threshold
        sorted_masks.append(mask)

        # Filter sorted indices based on the mask
        filtered_sorted_i = sorted_index_i[mask]
        filtered_sorted_j = sorted_index_j[mask]

        sorted_filtered_indices.append((filtered_sorted_i, filtered_sorted_j))

    return sorted_indices, sorted_filtered_indices, sorted_masks
