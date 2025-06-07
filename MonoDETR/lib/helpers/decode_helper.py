import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from scipy.spatial.transform import Rotation as R
from utils import box_ops
# from mmdet3d.ops import nms

def decode_detect_detections(dets, info, calibs, cls_mean_size, extrinsic, threshold, NMS):
    """
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    """
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][0]
            y = dets[i, j, 3] * info['img_size'][1]
            w = dets[i, j, 4] * info['img_size'][0]
            h = dets[i, j, 5] * info['img_size'][1]
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            # print(dimensions) 
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][0]
            y3d = dets[i, j, 35] * info['img_size'][1]
            locations = img_to_rect(calibs, x3d, y3d, depth)[0]
            # locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            # locations[1] += dimensions[0] / 2

            # heading angle decoding
            ry = get_heading_angle(dets[i, j, 7:31])
            feature_extrinsic = dets[i, j, 37:44]
            feature_extrinsic_matrix = feature_extrinsic_to_transformation(feature_extrinsic)


            # rotation_matrix = cfg['detect']['ext']
            rotation_matrix  = np.array(extrinsic).reshape(4,4)[:3,:3]
            ground_rotation = R.from_matrix(
                rotation_matrix)
            # ground_rotation = R.from_matrix(
            #     (camera_transformation() @ feature_extrinsic_matrix)[:3, :3])
            
            yaw_ground, pitch_ground, roll_ground = ground_rotation.as_euler(
                'zyx', degrees=False)
            alpha = ry2alpha(calibs, ry, (bbox[0] + bbox[2]) / 2)
            preds.append(
                [cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, pitch_ground, roll_ground,
                                                                                     score])
        # results[info['img_id'][i]] = preds
        if NMS:
            # boxes = np.array([pred[2:6] for pred in preds])
            # scores = np.array([pred[-1] for pred in preds])
            pass
            
    return preds


def decode_detections(dets, info, calibs, cls_mean_size, threshold, NMS):
    """
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    """
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            # locations[1] += dimensions[0] / 2

            # heading angle decoding
            ry = get_heading_angle(dets[i, j, 7:31])
            feature_extrinsic = dets[i, j, 37:44]
            feature_extrinsic_matrix = feature_extrinsic_to_transformation(feature_extrinsic)
            ground_rotation = R.from_matrix(
                (camera_transformation() @ feature_extrinsic_matrix)[:3, :3])
            yaw_ground, pitch_ground, roll_ground = ground_rotation.as_euler(
                'zyx', degrees=False)
            alpha = calibs[i].ry2alpha(ry, (bbox[0] + bbox[2]) / 2)

            preds.append(
                [cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, pitch_ground, roll_ground,
                                                                                     score])
        results[info['img_id'][i]] = preds
    return results

def img_to_rect(calib, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return:
    """
    P2 = calib.reshape(3,4).cpu().numpy()
    cu, cv, fu, fv = P2[0, 2], P2[1, 2], P2[0, 0], P2[1, 1]
    tx, ty =  P2[0, 3] / (-fu), P2[1, 3] / (-fv)
    x = ((u - cu) * depth_rect) / fu + tx
    y = ((v - cv) * depth_rect) / fv + ty
    pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect

def ry2alpha(calib, ry, u):
    P2 = calib.reshape(3,4).cpu().numpy()
    cu, cv, fu, fv = P2[0, 2], P2[1, 2], P2[0, 0], P2[1, 1]
    alpha = ry - np.arctan2(u - cu, fu)

    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi

    return alpha


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]

    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)
    feature_extrinsic = outputs['pred_feature_extrinsic'].unsqueeze(1).expand(-1, 50, 7)

    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    xs3d = boxes[:, :, 0: 1]
    ys3d = boxes[:, :, 1: 2]

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes)
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]

    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat(
        [labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma, feature_extrinsic], dim=2)

    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)  # type: ignore
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    """
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    """
    dim = feat.size(2)  # get channel dim
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()  # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))  # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)  # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


def camera_transformation(roll: float = -90) -> np.ndarray:
    """
    Generate a camera transformation matrix.

    Args:
        roll (float): Roll angle.

    Returns:
        np.ndarray: Camera transformation matrix.
    """
    if roll == -90:
        T = np.array([[1, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
    else:
        theta = np.radians(roll)
        T = np.array([[1, 0, 0, 0],
                      [0, np.cos(theta), -np.sin(theta), 0],
                      [0, np.sin(theta), np.cos(theta), 0],
                      [0, 0, 0, 1]])
    return T


def feature_extrinsic_to_transformation(feature_extrinsic: list) -> np.ndarray:
    r = feature_extrinsic[3:]
    t = feature_extrinsic[:3]
    q = R.from_quat(r)
    m = R.as_matrix(q)
    t = np.matrix(t).T
    matrix = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    return np.asarray(np.linalg.inv(matrix))


def project_3d_to_ground_plane(point, plane):
    """
    Projects a 3D point onto a plane with improved numerical stability.
    Returns the projected point as a list, or [None, None, None] if the calculation is unstable.

    :param point: A list [x, y, z] representing the 3D point to be projected.
    :param plane: A list [a, b, c, d] representing the plane equation coefficients in ax + by + cz + d = 0.
    :return: A list representing the projected point on the plane, or [None, None, None] for unstable cases.
    """

    # Unpack the point and plane coefficients
    x, y, z = point
    a, b, c, d = plane

    # Check for a degenerate plane (a, b, c all zero) which doesn't define a valid plane
    if a == 0 and b == 0 and c == 0:
        return [None, None, None]

    # Calculate the denominator and handle very small values to improve stability
    denom = a ** 2 + b ** 2 + c ** 2
    if abs(denom) < 1e-8:  # A small threshold to prevent division by a very small number
        return [None, None, None]

    # Calculate the scale factor for projection
    scale = -(a * x + b * y + c * z + d) / denom

    # Project the point onto the plane
    x_proj = x + a * scale
    y_proj = y + b * scale
    z_proj = z + c * scale

    return [x_proj, y_proj, z_proj]
