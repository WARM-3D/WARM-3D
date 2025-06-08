import numpy as np
import time
import torch
from scipy.spatial.transform import Rotation as R


def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


def calculate_plane_constraint_loss(outputs, info, dataloader, idx):
    # get corresponding calibs & transform tensor to numpy
    calibs = [dataloader.dataset.get_calib(index) for index in info['img_id']]
    extrinsics = [dataloader.dataset.get_extrinsics(
        index) for index in info['img_id']]
    cls_mean_size = torch.tensor(
        dataloader.dataset.cls_mean_size, device='cuda')

    location_dets = extract_plane_detections(outputs=outputs, idx=idx, calibs=calibs,
                                             cls_mean_size=cls_mean_size, info=info)

    loss = 0
    discard = 0
    for points_in_one_image in location_dets:
        if len(points_in_one_image) < 3:
            discard += 1
            continue
        _, variance_ratio = fit_plane_with_PCA_torch(points_in_one_image)
        loss += variance_ratio[2]
    loss = loss / len(location_dets) if len(location_dets) != discard else 0
    return loss


def extract_plane_detections(outputs, idx, calibs, cls_mean_size, info):
    # Extract necessary components from outputs
    boxes, depth = outputs['pred_boxes'][idx], outputs['pred_depth'][idx][:, 0:1]
    with torch.no_grad():
        out_logits, heading, size_3d, = outputs['pred_logits'][idx], \
            outputs['pred_angle'][idx], outputs['pred_3d_dim'][idx],
        img_size = info['img_size'][0, :2]
        cls_id = out_logits.argmax(dim=1)
        euler_angles_list = []
        feature_extrinsic = outputs['pred_feature_extrinsic']
        feature_extrinsic_matrixs = [feature_extrinsic_to_transformation(
            extrinsic) for extrinsic in feature_extrinsic]
        for feature_extrinsic_matrix in feature_extrinsic_matrixs:
            ground_rotation = (camera_transformation() @
                               feature_extrinsic_matrix)[:3, :3]
            yaw_ground, pitch_ground, roll_ground = R.from_matrix(ground_rotation.cpu().numpy()).as_euler('zyx',
                                                                                                          degrees=False)
            euler_angles = torch.tensor([yaw_ground, pitch_ground, roll_ground], device='cuda',
                                        dtype=torch.float32).unsqueeze(0)
            euler_angles_list.append(euler_angles)
    euler_angles_list = torch.concat(euler_angles_list, dim=0)

    dimensions = size_3d + cls_mean_size[cls_id]
    yaw = batch_get_heading_angle_torch(heading)
    _, pitch_ground, roll_ground = euler_angles_list[idx[0]].split(1, dim=1)

    locations = []
    P2 = []
    assert torch.equal(idx[0], torch.sort(idx[0])[0])

    for id_cab, cab in enumerate(calibs):
        mask = idx[0] == id_cab
        location = cab.img_to_rect_torch(
            boxes[:, 0] * img_size[0], boxes[:, 1] * img_size[1], depth.squeeze())
        p2 = torch.eye(4, device='cuda')
        p2[:3, :4] = torch.tensor(cab.P2, device='cuda', dtype=torch.float32)
        p2 = p2.repeat(len(boxes), 1, 1)

        locations.append(location[mask])
        P2.append(p2[mask])

    Trans = torch.cat(locations, dim=0)
    P2 = torch.cat(P2, dim=0)
    Rot = batch_euler_angles_to_rotation_matrix(
        yaw, pitch_ground.squeeze(), roll_ground.squeeze())

    Agent_transformation_matrix = torch.eye(
        4, device='cuda', requires_grad=True).unsqueeze(0).repeat(len(Rot), 1, 1)
    Agent_transformation_matrix[:, :3, :3] = Rot
    Agent_transformation_matrix[:, :3, 3] = Trans

    Bottom_center_in_camera = get_bottom_center_in_camera_frame_batch(
        Agent_transformation_matrix, dimensions[:, 0])

    def organize_by_image_simplified_torch(d3_locations, idx):
        # Get the maximum index to determine the number of images
        max_index = idx[0].max() + 1
        result = [None] * max_index

        # Iterate over each index, create a mask and extract locations for that index
        for i in range(max_index):
            mask = (idx[0] == i)
            result[i] = d3_locations[mask]

        return result

    return organize_by_image_simplified_torch(Bottom_center_in_camera, idx)


def fit_plane_with_PCA_torch(points):
    """
    Fit a plane in a 3D space using PCA (implemented via SVD) with PyTorch and calculate variance on each component.

    :param points: A torch tensor of shape (n_samples, 3) representing the 3D points.
    :return: A tuple containing the PCA components and the variance ratio of each component.
    """

    # Center the points by subtracting the mean
    points_mean = points.mean(dim=0)
    points_centered = points - points_mean

    # Compute the SVD of the centered points
    U, S, V = torch.svd(points_centered)

    # The right singular vectors (V) are the principal components
    components = V.t()

    # The singular values (S) are related to the explained variance
    variance = torch.pow(S, 2) / (points.size(0) - 1)
    total_variance = variance.sum()
    variance_ratio = variance / total_variance

    return components, variance_ratio


def get_bottom_center_in_camera_frame_batch(object_center_transform, object_height, height_scale=0.5):
    """
    Calculate the bottom center of a rectangular object in the camera reference frame using PyTorch.

    :param object_center_transform: Batch of 4x4 transformation matrices representing the position and orientation of the object center in the camera coordinate system.
    :param object_height: Batch of heights of the object. It could be a tensor with the same batch size or a single tensor if all objects have the same height.
    :param height_scale: Scalar or tensor that scales the height to compute the offset from the center to the bottom center. Defaults to 0.5.
    :return: Batch of coordinates of the bottom center of the object in the camera coordinate system.
    """

    if object_height.ndim == 0:
        object_height = object_height.expand(object_center_transform.shape[0])

    # Extract rotation (R) and translation (t) components from the transformation matrices
    R = object_center_transform[:, :3, :3]
    t = object_center_transform[:, :3, 3]

    # The height vector (in object coordinates) pointing from the center to the bottom
    height_vector_object_coords = torch.zeros((object_center_transform.shape[0], 3),
                                              device=object_center_transform.device)
    height_vector_object_coords[:, 2] = -object_height * height_scale

    # Transform the height vector to camera coordinates
    # Since it's a direction vector, we only apply the rotation
    height_vector_camera_coords = torch.bmm(
        R, height_vector_object_coords.unsqueeze(-1)).squeeze(-1)

    # Calculate the bottom center position in camera coordinates
    bottom_center_camera_coords = t + height_vector_camera_coords

    return bottom_center_camera_coords


def calculate_3d_bbox_size(outputs, info, dataloader, idx):
    # Pre-fetch and batch process calibration and extrinsic data
    img_ids = info['img_id']
    calibs = [dataloader.dataset.get_calib(index) for index in img_ids]
    extrinsics = [dataloader.dataset.get_extrinsics(
        index) for index in img_ids]
    cls_mean_size = torch.tensor(
        dataloader.dataset.cls_mean_size, device='cuda')

    # Processing detections
    dets, verts3d = extract_detections(outputs=outputs, idx=idx, calibs=calibs,
                                       cls_mean_size=cls_mean_size, info=info)
    # dets = torch.stack(dets)
    return dets, verts3d


# @time_func
def extract_detections(outputs, idx, calibs, cls_mean_size, info):
    # Unpack outputs directly to avoid redundant operations
    out_logits, boxes, heading, size_3d, depth = outputs['pred_logits'][idx], outputs['pred_boxes'][idx], \
        outputs['pred_angle'][idx], outputs['pred_3d_dim'][idx], outputs['pred_depth'][idx][:, 0:1]
    with torch.no_grad():
        feature_extrinsic = outputs['pred_feature_extrinsic']
        feature_extrinsic_matrixs = [feature_extrinsic_to_transformation(
            extrinsic) for extrinsic in feature_extrinsic]
    img_size = info['img_size'][0, :2]
    scaling_factors = torch.cat([img_size, img_size]).to('cuda')
    cls_id = out_logits.argmax(dim=1)

    # d3_boxes, verts3d_list = [], []
    # with torch.no_grad():
    #     euler_angles_list = []
    #     for feature_extrinsic_matrix in feature_extrinsic_matrixs:
    #         # yaw_ground, pitch_ground, roll_ground = get_euler_angles_from_rotation_matrix(ground_rotation)
    #         ground_rotation = (camera_transformation() @
    #                            feature_extrinsic_matrix)[:3, :3]
    #         yaw_ground, pitch_ground, roll_ground = R.from_matrix(ground_rotation.cpu().numpy()).as_euler('zyx',
    #                                                                                                       degrees=False)
    #         euler_angles = torch.tensor([yaw_ground, pitch_ground, roll_ground], device='cuda',
    #                                     dtype=torch.float32).unsqueeze(0)
    #         euler_angles_list.append(euler_angles)

    # euler_angles_list = torch.concat(euler_angles_list, dim=0)
    euler_angles_list = info['view'].to('cuda')

    dimensions = size_3d + cls_mean_size[cls_id]
    yaw = batch_get_heading_angle_torch(heading)
    _, pitch_ground, roll_ground = euler_angles_list[idx[0]].split(1, dim=1)

    locations = []
    P2 = []
    assert torch.equal(idx[0], torch.sort(idx[0])[0])

    for id_cab, cab in enumerate(calibs):
        mask = idx[0] == id_cab
        location = cab.img_to_rect_torch(
            boxes[:, 0] * img_size[0], boxes[:, 1] * img_size[1], depth.squeeze())
        p2 = torch.eye(4, device='cuda')
        p2[:3, :4] = torch.tensor(cab.P2, device='cuda', dtype=torch.float32)
        p2 = p2.repeat(len(boxes), 1, 1)

        locations.append(location[mask])
        P2.append(p2[mask])

    Trans = torch.cat(locations, dim=0)
    P2 = torch.cat(P2, dim=0)
    Rot = batch_euler_angles_to_rotation_matrix(
        yaw, pitch_ground.squeeze(), roll_ground.squeeze())

    Agent_transformation_matrix = torch.eye(
        4, device='cuda', requires_grad=True).unsqueeze(0).repeat(len(Rot), 1, 1)
    Agent_transformation_matrix[:, :3, :3] = Rot
    Agent_transformation_matrix[:, :3, 3] = Trans

    Bottom_center_in_camera = Trans
    # h, w, l = dimensions[:,0], dimensions[:,1], dimensions[:,2]
    g2c_trans = torch.eye(4, device='cuda').repeat(len(boxes), 1, 1)
    verts3d = batch_get_camera_3d_8points(
        dimensions, Bottom_center_in_camera, g2c_trans, P2,
        Agent_transformation_matrix)
    bounding_box = batch_calc_projected_2d_bbox(verts3d)
    bounding_box = bounding_box / scaling_factors
    verts3d = verts3d.permute(0, 2, 1)

    return bounding_box, verts3d


def batch_calc_projected_2d_bbox(vertices_pos2d):
    """
    Calculates the 2D bounding box from vertex positions using PyTorch.
    """
    x_coords = vertices_pos2d[:, 0, :]
    y_coords = vertices_pos2d[:, 1, :]
    min_x, _ = torch.min(x_coords, dim=1)
    max_x, _ = torch.max(x_coords, dim=1)
    min_y, _ = torch.min(y_coords, dim=1)
    max_y, _ = torch.max(y_coords, dim=1)
    return torch.cat([min_x.unsqueeze(1), min_y.unsqueeze(1), max_x.unsqueeze(1), max_y.unsqueeze(1)], dim=1)


# @time_func
def feature_extrinsic_to_transformation(feature_extrinsic: list):
    t, q = torch.tensor(feature_extrinsic[:3]), torch.tensor(
        feature_extrinsic[3:])
    # m = quat_to_rotation_matrix(q)
    q = q.detach().cpu().numpy()
    m = torch.tensor(R.from_quat(q).as_matrix(),
                     device='cuda', dtype=torch.float32)
    matrix = torch.cat([torch.cat([m, t.view(3, 1)], dim=1), torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda')],
                       dim=0)
    return torch.linalg.inv(matrix)


# @time_func
def camera_transformation(roll: float = -90):
    """
    Generate a camera transformation matrix.
    """
    if roll == -90:
        T = torch.tensor([[1, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]], dtype=torch.float32, device='cuda')
    else:
        theta = torch.radians(torch.tensor(
            roll, dtype=torch.float32, device='cuda'))
        T = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(theta), -torch.sin(theta), 0],
            [0, torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device='cuda')
    return T


def batch_euler_angles_to_rotation_matrix(yaw, pitch, roll):
    """
    Converts Euler angles to a rotation matrix.
    """
    N = yaw.size(0)
    device = yaw.device

    # Precompute cosines and sines of the angles
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)

    R_x = torch.stack([
        torch.stack([torch.ones(N, device=device), torch.zeros(N, device=device), torch.zeros(N, device=device)],
                    dim=1),
        torch.stack([torch.zeros(N, device=device),
                    cos_roll, -sin_roll], dim=1),
        torch.stack([torch.zeros(N, device=device), sin_roll, cos_roll], dim=1)
    ], dim=1)

    R_y = torch.stack([
        torch.stack([cos_pitch, torch.zeros(
            N, device=device), sin_pitch], dim=1),
        torch.stack([torch.zeros(N, device=device), torch.ones(N, device=device), torch.zeros(N, device=device)],
                    dim=1),
        torch.stack(
            [-sin_pitch, torch.zeros(N, device=device), cos_pitch], dim=1)
    ], dim=1)

    R_z = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, torch.zeros(N, device=device)], dim=1),
        torch.stack([sin_yaw, cos_yaw, torch.zeros(N, device=device)], dim=1),
        torch.stack([torch.zeros(N, device=device), torch.zeros(
            N, device=device), torch.ones(N, device=device)], dim=1)
    ], dim=1)

    # Compute the full rotation matrix by matrix multiplication
    R = R_x @ R_y @ R_z
    return R


def batch_get_camera_3d_8points(dimensions, center_ground, g2c_trans, P2, agent_in_camera):
    """
    Converts 3D bounding box corners to 2D using PyTorch tensors.
    """
    h, w, l = dimensions[:, 0], dimensions[:, 1], dimensions[:, 2]
    N = w.size(0)
    device = w.device
    agent_rot = agent_in_camera[:, :3, :3]  # Extract rotation part [N, 3, 3]

    # Create the corners relative to the center
    x_corners = (l / 2)[:, None] * torch.tensor([1,
                                                 1, -1, -1, 1, 1, -1, -1], device=device)
    y_corners = (w / 2)[:, None] * torch.tensor([1, -
                                                 1, -1, 1, 1, -1, -1, 1], device=device)
    z_corners = (h / 2)[:, None] * torch.tensor([-1, -
                                                 1, -1, -1, 1, 1, 1, 1], device=device)

    # Stack and replicate the corners across the batch
    corners = torch.stack(
        (x_corners, y_corners, z_corners), dim=1)  # [N, 3, 8]

    corners_final = torch.bmm(agent_rot, corners) + \
        center_ground[:, :, None].repeat(1, 1, 8)

    ones = torch.ones((N, 1, 8), device=device)
    corners_homogeneous = torch.cat((corners_final, ones), dim=1)
    corners_cam = torch.bmm(g2c_trans, corners_homogeneous)

    # Project corners to 2D
    # Broadcasting p2 across batches
    corners_2d_homogeneous = torch.bmm(P2, corners_cam)
    corners_2d = corners_2d_homogeneous[:, :2, :] / \
        corners_2d_homogeneous[:, 2, :].unsqueeze(1)  # Normalize by Z

    return corners_2d


def batch_get_heading_angle_torch(heading):
    """
    Convert heading information into an angle using PyTorch operations.
    """
    heading_bin, heading_res = heading[:, :12], heading[:, 12:24]
    cls = torch.argmax(heading_bin, dim=1)
    res = heading_res[torch.arange(heading_res.size(0)), cls]
    return class2angle(cls, res, to_label_format=True)


# @time_func
def class2angle(cls, res, to_label_format=True):
    """
    Convert class and residual to an angle.
    Assumes that bin angles and increments are predefined.
    This function should be adapted to use PyTorch if it involves tensors.
    """
    num_bins = 12
    bin_size = 2 * torch.pi / num_bins  # Total of 360 degrees divided into num_bins
    angle_center = cls * bin_size

    if to_label_format:
        angle = angle_center + res
    else:
        angle = angle_center - res

    return angle
