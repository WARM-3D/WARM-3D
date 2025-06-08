"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import copy
import cv2
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from lib.losses.focal_loss import sigmoid_focal_loss
from torch import nn
from utils import box_ops
from utils.misc import (NestedTensor, accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid)

from .additional_constraints import calculate_3d_bbox_size, calculate_plane_constraint_loss
from .backbone import build_backbone
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from .depthaware_transformer import build_depthaware_transformer
from .matcher import build_matcher


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """

    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11,
                 two_stage_dino=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.feature_extrinsics_embed = MLP(hidden_dim, hidden_dim, 7, 2)  # extrinsics

        self.use_dab = use_dab

        if init_box:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
                depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, images, calibs, targets, img_sizes):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight  # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

            else:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight[:self.num_queries]
                refanchor = self.refpoint_embed.weight[:self.num_queries]
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip, depth_encoding = self.depth_predictor(
            srcs,
            masks[1],
            pos[1])

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact, visual_encoding = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)  # , attn_mask)

        depth_image_encoding = torch.cat([depth_encoding, visual_encoding], dim=1)
        outputs_feature_extrinsic = self.feature_extrinsics_embed(depth_image_encoding)
        outputs_feature_extrinsic = outputs_feature_extrinsic.mean(dim=1)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat(
                [((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_map) / 2,
                 depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_3d_dim': outputs_3d_dim[-1],
               'pred_depth': outputs_depth[-1], 'pred_angle': outputs_angle[-1],
               'pred_depth_map_logits': pred_depth_map_logits,
               'pred_feature_extrinsic': outputs_feature_extrinsic}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth,
                outputs_feature_extrinsic)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out  # , mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth,
                      outputs_feature_extrinsic):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e,
                 'pred_feature_extrinsic': outputs_feature_extrinsic}
                for a, b, c, d, e, in zip(outputs_class[:-1], outputs_coord[:-1],
                                          outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, cfg, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict_source = weight_dict[0]
        self.weight_dict_target = weight_dict[1]
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

        self.dataloader = None
        self.info = None
        self.mask = None
        self.resoluition: list[int] = [960, 600]

        self.batch_count = 0
        self.model_name = cfg['model_name']

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        if self.mask is not None:
            mask = [t[i] for t, (_, i) in zip(self.mask, indices)]
            indices = [(i[mask[t]], j[mask[t]]) for (i, j), t in zip(indices, range(len(mask)))]
            num_boxes = sum([len(i) for (i, j) in indices])

        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {'loss_center': loss_3dcenter.sum() / num_boxes}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = \
            torch.stack(
                (outputs['pred_boxes'][:, :, 2: 4].sum(dim=-1), outputs['pred_boxes'][:, :, 4: 6].sum(dim=-1)),
                dim=-1)[idx]
        target_2dboxes = torch.cat([t['boxes'][:, 2: 4][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):
        if self.mask is not None:
            mask = [t[i] for t, (_, i) in zip(self.mask, indices)]
            indices = [(i[mask[t]], j[mask[t]]) for (i, j), t in zip(indices, range(len(mask)))]
            num_boxes = sum([len(i) for (i, j) in indices])

        idx = self._get_src_permutation_idx(indices)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(
            depth_input - target_depths) + depth_log_variance
        losses = {'loss_depth': depth_loss.sum() / num_boxes}
        return losses

    def loss_dims(self, outputs, targets, indices, num_boxes):
        if self.mask is not None:
            mask = [t[i] for t, (_, i) in zip(self.mask, indices)]
            indices = [(i[mask[t]], j[mask[t]]) for (i, j), t in zip(indices, range(len(mask)))]
            num_boxes = sum([len(i) for (i, j) in indices])

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {'loss_dim': dim_loss.sum() / num_boxes}
        if torch.isnan(losses['loss_dim']):
            losses['loss_dim'] = torch.tensor(0.0, device=dim_loss.device)

        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):
        if self.mask is not None:
            mask = [t[i] for t, (_, i) in zip(self.mask, indices)]
            indices = [(i[mask[t]], j[mask[t]]) for (i, j), t in zip(indices, range(len(mask)))]
            num_boxes = sum([len(i) for (i, j) in indices])

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1,
                                                                                  index=heading_target_cls.view(-1, 1),
                                                                                  value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')

        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        resolution = self.resoluition
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor(
            [resolution[0] / 16, resolution[1] / 16, resolution[0] / 16, resolution[1] / 16], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = {"loss_depth_map": self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)}
        return losses

    def loss_constraint(self, outputs, targets, indices, num_boxes):
        # if self.mask is not None:
        #     mask = [t[i] for t, (_, i) in zip(self.mask, indices)]
        #     indices = [(i[mask[t]], j[mask[t]]) for (i, j), t in zip(indices, range(len(mask)))]
        #     num_boxes = sum([len(i) for (i, j) in indices])
        
        idx = self._get_src_permutation_idx(indices)

        # giou 2d
        src_3d_boxes, verts3d_list = calculate_3d_bbox_size(outputs=outputs, info=self.info, dataloader=self.dataloader,
                                                            idx=idx)
        target_boxes_2d = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes_2d[:, 2:4] = target_boxes_2d[:, 2:4] * 1.1
        target_boxes_2d_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes_2d)

        target_boxes_3d = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes_3d_xyxy = box_ops.box_cxcylrtb_to_xyxy(target_boxes_3d)

        # Visualize boxes
        with torch.no_grad():
            if self.batch_count == 0 or self.batch_count % 400 == 0:
                self.visualize_boxes_on_images(idx, src_3d_boxes, target_boxes_2d_xyxy, target_boxes_3d_xyxy,
                                               verts3d_list)
            self.batch_count += 1

        # Calculate GIoU loss
        loss_giou_2d = 1 - torch.diag(box_ops.generalized_box_iou(src_3d_boxes, target_boxes_2d_xyxy))

        # Calculate L1 loss for box centers
        src_centers = (src_3d_boxes[:, :2] + src_3d_boxes[:, 2:]) / 2
        target_centers = (target_boxes_2d_xyxy[:, :2] + target_boxes_2d_xyxy[:, 2:]) / 2
        loss_l1_centers = F.l1_loss(src_centers, target_centers, reduction='sum') / num_boxes

        # Combine losses
        losses = {'loss_constraint': loss_giou_2d.sum() / num_boxes + loss_l1_centers}

        return losses

    def loss_extrinsics(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        loss_plane_constraint = calculate_plane_constraint_loss(outputs=outputs, info=self.info,
                                                                dataloader=self.dataloader, idx=idx)
        losses = {'loss_extrinsics': F.l1_loss(torch.tensor(loss_plane_constraint), torch.tensor(0),
                                               reduction='none').sum()}
        return losses

    def loss_feature_extrinsics(self, outputs, targets, indices, num_boxes):
        src_feature_extrinsics = outputs['pred_feature_extrinsic'].reshape(-1)
        target_feature_extrinsics = torch.cat([t['feature_extrinsic'] for t in targets], dim=0)
        loss_feature_extrinsic = F.l1_loss(src_feature_extrinsics, target_feature_extrinsics, reduction='none')
        losses = {'loss_feature_extrinsic': loss_feature_extrinsic.sum()}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            # 'constraint': self.loss_constraint,
            # 'extrinsics': self.loss_extrinsics,
            # 'feature_extrinsic': self.loss_feature_extrinsics
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask=None, info=None, data_domain='source'):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             mask:
             info:
             data_domain:
        """
        self.info = info
        self.resoluition = info['resolution'][0].tolist()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses_set = []
        if data_domain == 'source':
            losses_set = self.losses[0]
        elif data_domain == 'target':
            losses_set = self.losses[1]
        else:
            raise ValueError('data_domain should be source or target')

        self.mask = mask

        for loss in losses_set:
            # ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            losses_set = [loss for loss in losses_set if loss != 'feature_extrinsic']
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in losses_set:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def visualize_boxes_on_images(self, idx, src_3d_boxes, target_boxes_2d, target_boxes_3d, verts3d_list=None):
        scaling_factors = self.info['img_size'][0, :2].repeat(2).detach().cpu().numpy() / 2
        src_3d_boxes = src_3d_boxes.detach().cpu().numpy()
        target_boxes_2d = target_boxes_2d.detach().cpu().numpy()
        target_boxes_3d = target_boxes_3d.detach().cpu().numpy()

        # Calculate 3D bbox converted to 2D
        src_3d_boxes = src_3d_boxes * scaling_factors
        target_boxes_2d = target_boxes_2d * scaling_factors
        target_boxes_3d = target_boxes_3d * scaling_factors

        # Fetch images for visualization
        # images = [self.dataloader.dataset.get_image(index) for index in self.info['img_id']]
        std = self.dataloader.dataset.std
        mean = self.dataloader.dataset.mean
        images = ((self.info['img_weak_augmented'].detach().cpu().numpy().transpose(0, 2, 3,
                                                                                    1) * std + mean) * 255).astype(
            np.uint8)
        # images = [np.array(image) for image in images]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        save_dir = f"./visual/{self.model_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, image in enumerate(images):
            idx_i = np.where(idx[0] == i)

            def draw_line(img, start_point, end_point, color, thickness):
                start_point = tuple(int(round(x)) for x in start_point)
                end_point = tuple(int(round(x)) for x in end_point)
                cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)

            if verts3d_list is not None:
                selected_verts3d = [np.asarray(verts3d_list[i].cpu()) for i in idx_i[0].tolist()]

                for j, verts3d_points in enumerate(selected_verts3d):
                    edges = [(2, 1), (1, 0), (0, 3), (2, 3), (7, 4), (4, 5), (5, 6), (6, 7), (7, 3), (1, 5), (0, 4),
                             (2, 6)]
                    for start, end in edges:
                        start_point = verts3d_points[start] / 2
                        end_point = verts3d_points[end] / 2
                        draw_line(image, start_point, end_point, (0, 0, 255), 2)
            # Draw source boxes
            # for j, box in enumerate(src_3d_boxes[idx_i]):
            #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            # cv2.putText(image, f"{j}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
            #             2)

            # Draw target boxes
            for j, box in enumerate(target_boxes_2d[idx_i]):
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # cv2.putText(image, f"{j}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                #             2)

            for j, box in enumerate(target_boxes_3d[idx_i]):
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
                # cv2.putText(image, f"{j}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                #             2)

            cv2.imwrite(f"./visual/{self.model_name}/image_Epoch_{self.info['epoch']}_step_{self.batch_count + i}.png", image)


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


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab=cfg['use_dab'],
        two_stage_dino=cfg['two_stage_dino'])

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict_source = compute_weight_dict_source(cfg)
    weight_dict_target = compute_weight_dict_target(cfg)
    weight_dict = [weight_dict_source, weight_dict_target]

    losses_source = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map',
                     'feature_extrinsic']
    # losses_target = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map',
    #                  'extrinsics', 'constraint']

    # losses_target = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map',
    #                  'constraint']
    losses_target = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
    
    losses = [losses_source, losses_target]

    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        cfg=cfg)

    device = torch.device(cfg['device'])
    criterion.to(device)

    return model, criterion, matcher


def compute_weight_dict_source(cfg):
    """
    Compute the weight dictionary for the source losses based on the given configuration.

    Args:
        cfg (dict): Configuration dictionary containing the coefficients for different losses.

    Returns:
        dict: Weight dictionary for the source losses.
    """

    weight_dict_source = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef'],
                          'loss_giou': cfg['giou_loss_coef'], 'loss_dim': cfg['dim_loss_coef'],
                          'loss_angle': cfg['angle_loss_coef'], 'loss_depth': cfg['depth_loss_coef'],
                          'loss_center': cfg['3dcenter_loss_coef'], 'loss_depth_map': cfg['depth_map_loss_coef']}

    if cfg['use_dn']:
        weight_dict_source['tgt_loss_ce'] = cfg['cls_loss_coef']
        weight_dict_source['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict_source['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict_source['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict_source['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict_source.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict_source.items()})
        weight_dict_source.update(aux_weight_dict)

    return weight_dict_source


def compute_weight_dict_target(cfg):
    """
    Compute the weight dictionary for target losses based on the given configuration.

    Args:
        cfg (dict): Configuration dictionary containing the loss coefficients.

    Returns:
        dict: Weight dictionary for target losses.

    """
    weight_dict_target = {
        'loss_ce': cfg.get('target_cls_loss_coef', cfg['cls_loss_coef']),
        'loss_bbox': cfg.get('target_bbox_loss_coef', cfg['bbox_loss_coef']),
        'loss_giou': cfg.get('target_giou_loss_coef', cfg['giou_loss_coef']),
        'loss_dim': cfg.get('target_dim_loss_coef', cfg['dim_loss_coef']),
        'loss_angle': cfg.get('target_angle_loss_coef', cfg['angle_loss_coef']),
        'loss_depth': cfg.get('target_depth_loss_coef', cfg['depth_loss_coef']),
        'loss_center': cfg.get('target_3dcenter_loss_coef', cfg['3dcenter_loss_coef']),
        'loss_depth_map': cfg.get('target_depth_map_loss_coef', cfg['depth_map_loss_coef']),
        'loss_constraint': cfg['target_constraint_loss_coef'],
        'loss_extrinsics': cfg['target_extrinsic_loss_coef']
    }

    if cfg.get('use_target_dn', False):
        weight_dict_target['tgt_loss_ce'] = cfg.get('target_cls_loss_coef', cfg['cls_loss_coef'])
        weight_dict_target['tgt_loss_bbox'] = cfg.get('target_bbox_loss_coef', cfg['bbox_loss_coef'])
        weight_dict_target['tgt_loss_giou'] = cfg.get('target_giou_loss_coef', cfg['giou_loss_coef'])
        weight_dict_target['tgt_loss_angle'] = cfg.get('target_angle_loss_coef', cfg['angle_loss_coef'])
        weight_dict_target['tgt_loss_center'] = cfg.get('target_3dcenter_loss_coef', cfg['3dcenter_loss_coef'])

    if cfg.get('target_aux_loss', False):
        aux_weight_dict = {}
        for i in range(cfg.get('target_dec_layers', cfg['dec_layers']) - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict_target.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict_target.items()})
        weight_dict_target.update(aux_weight_dict)

    return weight_dict_target
