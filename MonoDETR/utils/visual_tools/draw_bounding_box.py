import cv2
import glob as gb
import json
import math
import numpy as np
import os
import sys
import yaml
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# stop python from writing so much bytecode
sys.dont_write_bytecode = True

np.set_printoptions(suppress=True)

# import config
import utils.visual_tools.config as config
from utils.visual_tools.make_video import create_video_from_images


class Data:
    """ class Data """

    def __init__(self, obj_type="unset", truncation=-1, occlusion=-1,
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1,
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, pitch=-10, roll=-10, score=-1000, detect_id=-1,
                 vx=0, vy=0, vz=0):
        """init object data"""
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.score = score
        self.ignored = False
        self.valid = False
        self.detect_id = detect_id

    def __str__(self):
        """ str """
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.

    Args:
        calfile (str): path to single calibration file
    """
    text_file = open(calfile, 'r')
    for line in text_file:
        parsed = line.split('\n')[0].split(' ')
        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None and parsed[0] == 'P2:':
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed[1]
            p2[0, 1] = parsed[2]
            p2[0, 2] = parsed[3]
            p2[0, 3] = parsed[4]
            p2[1, 0] = parsed[5]
            p2[1, 1] = parsed[6]
            p2[1, 2] = parsed[7]
            p2[1, 3] = parsed[8]
            p2[2, 0] = parsed[9]
            p2[2, 1] = parsed[10]
            p2[2, 2] = parsed[11]
            p2[2, 3] = parsed[12]
            p2[3, 3] = 1
    text_file.close()
    return p2


def load_detect_data(filename):
    """
    load detection data of kitti format
    """
    data = []
    with open(filename) as infile:
        index = 0
        for line in infile:
            # KITTI detection benchmark data format:
            # (objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields = line.split(" ")
            t_data = Data()
            # get fields from table
            t_data.obj_type = fields[
                0].lower()  # object type [car, pedestrian, cyclist, ...]
            t_data.truncation = float(fields[1])  # truncation [0..1]
            t_data.occlusion = int(float(fields[2]))  # occlusion  [0,1,2]
            t_data.obs_angle = float(fields[3])  # observation angle [rad]
            t_data.x1 = int(float(fields[4]))  # left   [px]
            t_data.y1 = int(float(fields[5]))  # top    [px]
            t_data.x2 = int(float(fields[6]))  # right  [px]
            t_data.y2 = int(float(fields[7]))  # bottom [px]
            t_data.h = float(fields[8])  # height [m]
            t_data.w = float(fields[9])  # width  [m]
            t_data.l = float(fields[10])  # length [m]
            t_data.X = float(fields[11])  # X [m]
            t_data.Y = float(fields[12])  # Y [m]
            t_data.Z = float(fields[13])  # Z [m]
            if config.ground_project:
                t_data.X = float(fields[-3])  # X [m]
                t_data.Y = float(fields[-2])  # Y [m]
                t_data.Z = float(fields[-1])  # Z [m]
            t_data.yaw = float(fields[14])  # yaw angle [rad]
            if len(fields) == 16:
                t_data.score = float(fields[15])  # detection score
            elif len(fields) == 17:
                t_data.pitch = float(fields[15])  # pitch angle [rad]
                t_data.roll = float(fields[16])  # roll angle [rad]
                t_data.score = 1
            elif len(fields) > 17:
                t_data.pitch = float(fields[15])  # pitch angle [rad]
                t_data.roll = float(fields[16])  # roll angle [rad]
                t_data.score = float(fields[17])  # detection score

            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


def project_3d_world(p2, de_center_in_world, w3d, h3d, l3d, ry3d, camera2world, isCenter=1):
    """
    help with world
    Projects a 3D box into 2D vertices using the camera2world tranformation
    Note: Since the roadside camera contains pitch and roll angle w.r.t. the ground/world,
    simply adopting KITTI-style projection not works. We first compute the 3D bounding box in ground-coord and then convert back to camera-coord.

    Args:
        p2 (nparray): projection matrix of size 4x3
        de_bottom_center: bottom center XYZ-coord of the object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        camera2world: camera_to_world translation
    """
    center_world = np.array(de_center_in_world)  # bottom center in world
    theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = camera2world[:3, :3] * theta  # first column
    world2camera = np.linalg.inv(camera2world)
    yaw_world_res = math.atan2(theta0[1], theta0[0])
    verts3d, agent_in_camera = get_camera_3d_8points_g2c(w3d, h3d, l3d,
                                                         yaw_world_res, center_world[:3, :], world2camera, p2,
                                                         isCenter=isCenter)

    verts3d = np.array(verts3d)
    return verts3d, agent_in_camera


def read_kitti_ext(extfile):
    """read extrin"""
    text_file = open(extfile, 'r')
    cont = text_file.read()
    x = yaml.safe_load(cont)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    p1 = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    return np.array(p1.I)


def read_a9_ext(extfile):
    with open(extfile, "r") as json_file:
        data = json.load(json_file)
        transformation1 = np.array(data['lidar_to_camera_matrix'])
        transformation1 = np.array(transformation1).reshape((4, 4))
        transformation2 = np.array(
            data['base_to_camera_matrix']).reshape((4, 4))
        transformation3 = np.array(
            data['base_to_lidar_matrix']).reshape((4, 4))
        return transformation1, transformation2, transformation3


def get_camera_3d_8points_g2c(w3d, h3d, l3d, yaw_ground, center_ground,
                              g2c_trans, p2,
                              isCenter=1):
    """
    function: projection 3D to 2D
    w3d: width of object
    h3d: height of object
    l3d: length of object
    yaw_world: yaw angle in world coordinate
    center_world: the center or the bottom-center of the object in world-coord
    g2c_trans: ground2camera / world2camera transformation
    p2: projection matrix of size 4x3 (camera intrinsics)
    isCenter:
        1: center,
        0: bottom
    """
    ground_r = np.matrix([[math.cos(yaw_ground), -math.sin(yaw_ground), 0],
                          [math.sin(yaw_ground), math.cos(yaw_ground), 0],
                          [0, 0, 1]])
    # l, w, h = obj_size
    w = w3d
    l = l3d
    h = h3d

    if isCenter:
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                       [w / 2, -w / 2, -w / 2, w / 2,
                                        w / 2, -w / 2, -w / 2, w / 2],
                                       [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    else:  # bottom center, ground: z axis is up
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                       [w / 2, -w / 2, -w / 2, w / 2,
                                        w / 2, -w / 2, -w / 2, w / 2],
                                       [0, 0, 0, 0, h, h, h, h]])

    corners_3d_ground = np.matrix(
        ground_r) * np.matrix(corners_3d_ground) + np.matrix(center_ground)  # [3, 8]

    if g2c_trans.shape[0] == 4:  # world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * \
                         np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  # only consider the rotation
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground  # [3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)
    if True in np.isnan(corners_2d_all):
        print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])

    agent_in_camera = ground_r @ np.eye(3)
    agent_in_camera = np.hstack((agent_in_camera, center_ground))
    agent_in_camera = np.vstack((agent_in_camera, np.array([0, 0, 0, 1])))
    return corners_2d, g2c_trans @ agent_in_camera


def get_camera_3d_8points_g2c_a9(w3d, h3d, l3d, center_ground,
                                 g2c_trans, p2,
                                 isCenter=1, agent_in_lidar=None):
    """
    function: projection 3D to 2D
    w3d: width of object
    h3d: height of object
    l3d: length of object
    center_world: the center or the bottom-center of the object in world-coord
    g2c_trans: ground2camera / world2camera transformation
    p2: projection matrix of size 4x3 (camera intrinsics)
    isCenter:
        1: center,
        0: bottom
    """

    ground_r = agent_in_lidar[:3, :3]
    # l, w, h = obj_size
    w = w3d
    l = l3d
    h = h3d

    if isCenter:
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                       [w / 2, -w / 2, -w / 2, w / 2,
                                        w / 2, -w / 2, -w / 2, w / 2],
                                       [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    else:  # bottom center, ground: z axis is up
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                       [w / 2, -w / 2, -w / 2, w / 2,
                                        w / 2, -w / 2, -w / 2, w / 2],
                                       [0, 0, 0, 0, h, h, h, h]])

    center_ground = np.array(center_ground)
    center_ground = center_ground[:3].reshape((3, 1))

    corners_3d_ground = np.matrix(
        ground_r) * np.matrix(corners_3d_ground) + np.matrix(center_ground)  # [3, 8]

    if g2c_trans.shape[0] == 4:  # world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * \
                         np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  # only consider the rotation
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground  # [3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)
    if True in np.isnan(corners_2d_all):
        print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])

    corners_2d = np.array(corners_2d)
    return corners_2d

def draw_detection(img, result, calibs, cfg):   
    """Show 2D box and 3D box.

    Args:
        name_list (list): List of image names.
        thresh (float, optional): Threshold for object detection. Defaults to 0.5.
        projectMethod (str, optional): Projection method ('Ground' or 'World'). Defaults to 'Ground'.
    """
    for result_index in range(len(result)):
        t = result[result_index]
        obj_type_id = t[0]
        x1 = int(t[2])
        y1 = int(t[3])
        x2 = int(t[4])
        y2 = int(t[5])
        h = t[6]
        w = t[7]
        l = t[8]
        X = t[9]
        Y = t[10]
        Z = t[11]
        yaw = t[12]
        pitch = t[13]
        roll = t[14]
        score = t[15]
        
        obj_type_map = cfg['detect']['class_name']
        obj_type = obj_type_map[obj_type_id].lower()
        # if obj_type_map[obj_type].low not in config.color_list:
        #     continue

        color_type = config.color_list[obj_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if w <= 0.05 and l <= 0.05 and h <= 0.05:  # Invalid annotation
            continue

        cam_bottom_center = [X, Y, Z]

        agent_rotation = R.from_euler(
            'zyx', [yaw, pitch, roll], degrees=False)
        agent_translation = np.array(cam_bottom_center)
        agent_rotation_matrix = agent_rotation.as_matrix()
        agent_transformation_matrix = np.eye(4)
        agent_transformation_matrix[:3, :3] = agent_rotation_matrix
        agent_transformation_matrix[:3, 3] = agent_translation

        bottom_center_in_camera = np.matrix(
            cam_bottom_center + [1.0]).T

        isCenter = 0 if config.ground_project else 1
        P2 = calibs.reshape(3,4).cpu().numpy()

        verts3d = get_camera_3d_8points_g2c_a9(
            w, h, l, bottom_center_in_camera, np.eye(4), P2, isCenter=isCenter,
            agent_in_lidar=agent_transformation_matrix)

        if verts3d is None:
            continue
        verts3d = verts3d.astype(np.int32)

        # Function to draw 3D bounding box lines
        def draw_line(img, start_point, end_point, color, thickness):
            cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)

        # Improved function to draw labels with background for better visibility
        def draw_label(img, label, label_pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_thickness=2,
                        background_color=(255, 255, 255), text_color=(0, 0, 0)):
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_rect_end = (label_pos[0] + label_size[0] + 2, label_pos[1] - label_size[1] - 2)
            label_rect_start = (label_pos[0] - 2, label_pos[1] + 2)

            # cv2.rectangle(img, label_rect_start, label_rect_end, background_color, cv2.FILLED)

            cv2.putText(img, label, (label_pos[0], label_pos[1] - 5), font, font_scale, text_color, font_thickness)

        # Draw the 3D bounding box
        edges = [(2, 1), (1, 0), (0, 3), (2, 3), (7, 4), (4, 5), (5, 6), (6, 7), (7, 3), (1, 5), (0, 4), (2, 6)]
        for start, end in edges:
            draw_line(img, verts3d[start], verts3d[end], color_type, 2)

        # Calculate label position
        label_pos = ((np.amin(verts3d[:, 0]) + np.amax(verts3d[:, 0])) // 2, np.amin(verts3d[:, 1]) - 10)
        if obj_type.lower() == 'bigcar':
            obj_type = 'BigVehicle'
        
        object_label = f'{obj_type} {score:.2f}'
        depth_label = f'Depth: {Z:.2f}'

        # Draw object label
        draw_label(img, object_label, label_pos, font_scale=0.5, background_color=(0, 0, 0),
                    text_color=(255, 255, 255))

        # Adjust depth label position below the object label
        # depth_label_pos = (label_pos[0], label_pos[1] + 20)
        # draw_label(img, depth_label, depth_label_pos, font_scale=0.5, background_color=(0, 0, 0),
        #            text_color=(255, 255, 255))

    return img

def draw_detection_and_tracking(img, result, calibs, cfg, tracked_states):
    """Show 2D box and 3D box with tracked states.

    Args:
        img (numpy.ndarray): The image on which to draw the detections.
        result (list): The detections from the model.
        calibs (torch.Tensor): Calibration data.
        cfg (dict): Configuration dictionary.
        tracked_states (list): List of tracked object states.
    """
    for result_index in range(len(result)):
        t = result[result_index]
        obj_type_id = t[0]
        x1 = int(t[2])
        y1 = int(t[3])
        x2 = int(t[4])
        y2 = int(t[5])
        h = t[6]
        w = t[7]
        l = t[8]
        X = t[9]
        Y = t[10]
        Z = t[11]
        yaw = t[12]
        pitch = t[13]
        roll = t[14]
        score = t[15]
        
        # Use the tracked state if available
        if result_index < len(tracked_states):
            # tracked_X, tracked_Y, tracked_Z = tracked_states[result_index]
            tracked_X, tracked_Y, tracked_Z, track_id = tracked_states[result_index]

            # X, Y, Z = tracked_X, tracked_Y, tracked_Z
        
        obj_type_map = cfg['detect']['class_name']
        obj_type = obj_type_map[obj_type_id].lower()

        color_type = config.color_list[obj_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if w <= 0.05 and l <= 0.05 and h <= 0.05:  # Invalid annotation
            continue

        cam_bottom_center = np.array([X, Y, Z])

        agent_rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        agent_translation = cam_bottom_center.flatten()
        agent_rotation_matrix = agent_rotation.as_matrix()
        agent_transformation_matrix = np.eye(4)
        agent_transformation_matrix[:3, :3] = agent_rotation_matrix
        agent_transformation_matrix[:3, 3] = agent_translation

        bottom_center_in_camera = np.append(cam_bottom_center, 1.0).reshape(4, 1)

        isCenter = 0 if config.ground_project else 1
        P2 = calibs.reshape(3, 4).cpu().numpy()

        verts3d = get_camera_3d_8points_g2c_a9(
            w, h, l, bottom_center_in_camera, np.eye(4), P2, isCenter=isCenter,
            agent_in_lidar=agent_transformation_matrix)

        if verts3d is None:
            continue
        verts3d = verts3d.astype(np.int32)

        def draw_line(img, start_point, end_point, color, thickness):
            cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)

        def draw_label(img, label, label_pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_thickness=2,
                        background_color=(255, 255, 255), text_color=(0, 0, 0)):
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_rect_end = (label_pos[0] + label_size[0] + 2, label_pos[1] - label_size[1] - 2)
            label_rect_start = (label_pos[0] - 2, label_pos[1] + 2)

            cv2.putText(img, label, (label_pos[0], label_pos[1] - 5), font, font_scale, text_color, font_thickness)

        edges = [(2, 1), (1, 0), (0, 3), (2, 3), (7, 4), (4, 5), (5, 6), (6, 7), (7, 3), (1, 5), (0, 4), (2, 6)]
        for start, end in edges:
            draw_line(img, verts3d[start], verts3d[end], color_type, 2)

        label_pos = ((np.amin(verts3d[:, 0]) + np.amax(verts3d[:, 0])) // 2, np.amin(verts3d[:, 1]) - 10)
        if obj_type.lower() == 'bigcar':
            obj_type = 'BigVehicle'
        
        # object_label = f'{obj_type} {score:.2f}'
        object_label = f'{obj_type} {score:.2f} ID: {track_id}'

        # depth_label = f'Depth: {Z:.2f}'

        draw_label(img, object_label, label_pos, font_scale=0.5, background_color=(0, 0, 0),
                    text_color=(255, 255, 255))

    return img

def draw_detection_and_tracking_centerpoint(img, result, calibs, cfg, outputs):
    """Show 2D box and 3D box with tracked states.

    Args:
        img (numpy.ndarray): The image on which to draw the detections.
        result (list): The detections from the model.
        calibs (torch.Tensor): Calibration data.
        cfg (dict): Configuration dictionary.
        tracked_states (list): List of tracked object states.
    """
    for result_index in range(len(result)):
        t = result[result_index]
        obj_type_id = t[0]
        x1 = int(t[2])
        y1 = int(t[3])
        x2 = int(t[4])
        y2 = int(t[5])
        h = t[6]
        w = t[7]
        l = t[8]
        X = t[9]
        Y = t[10]
        Z = t[11]
        yaw = t[12]
        pitch = t[13]
        roll = t[14]
        score = t[15]
        
        # Use the tracked state if available
        if result_index < len(tracked_states):
            # tracked_X, tracked_Y, tracked_Z = tracked_states[result_index]
            tracked_X, tracked_Y, tracked_Z, track_id = tracked_states[result_index]

            # X, Y, Z = tracked_X, tracked_Y, tracked_Z
        
        obj_type_map = cfg['detect']['class_name']
        obj_type = obj_type_map[obj_type_id].lower()

        color_type = config.color_list[obj_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if w <= 0.05 and l <= 0.05 and h <= 0.05:  # Invalid annotation
            continue

        cam_bottom_center = np.array([X, Y, Z])

        agent_rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        agent_translation = cam_bottom_center.flatten()
        agent_rotation_matrix = agent_rotation.as_matrix()
        agent_transformation_matrix = np.eye(4)
        agent_transformation_matrix[:3, :3] = agent_rotation_matrix
        agent_transformation_matrix[:3, 3] = agent_translation

        bottom_center_in_camera = np.append(cam_bottom_center, 1.0).reshape(4, 1)

        isCenter = 0 if config.ground_project else 1
        P2 = calibs.reshape(3, 4).cpu().numpy()

        verts3d = get_camera_3d_8points_g2c_a9(
            w, h, l, bottom_center_in_camera, np.eye(4), P2, isCenter=isCenter,
            agent_in_lidar=agent_transformation_matrix)

        if verts3d is None:
            continue
        verts3d = verts3d.astype(np.int32)

        def draw_line(img, start_point, end_point, color, thickness):
            cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)

        def draw_label(img, label, label_pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_thickness=2,
                        background_color=(255, 255, 255), text_color=(0, 0, 0)):
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_rect_end = (label_pos[0] + label_size[0] + 2, label_pos[1] - label_size[1] - 2)
            label_rect_start = (label_pos[0] - 2, label_pos[1] + 2)

            cv2.putText(img, label, (label_pos[0], label_pos[1] - 5), font, font_scale, text_color, font_thickness)

        edges = [(2, 1), (1, 0), (0, 3), (2, 3), (7, 4), (4, 5), (5, 6), (6, 7), (7, 3), (1, 5), (0, 4), (2, 6)]
        for start, end in edges:
            draw_line(img, verts3d[start], verts3d[end], color_type, 2)

        label_pos = ((np.amin(verts3d[:, 0]) + np.amax(verts3d[:, 0])) // 2, np.amin(verts3d[:, 1]) - 10)
        if obj_type.lower() == 'bigcar':
            obj_type = 'BigVehicle'
        
        # object_label = f'{obj_type} {score:.2f}'
        object_label = f'{obj_type} {score:.2f} ID: {track_id}'

        # depth_label = f'Depth: {Z:.2f}'

        draw_label(img, object_label, label_pos, font_scale=0.5, background_color=(0, 0, 0),
                    text_color=(255, 255, 255))

    return img


def show_box_with_roll(name_list):
    """Show 2D box and 3D box.

    Args:
        name_list (list): List of image names.
        thresh (float, optional): Threshold for object detection. Defaults to 0.5.
        projectMethod (str, optional): Projection method ('Ground' or 'World'). Defaults to 'Ground'.
    """
    # Configurations
    image_root = config.image_dir
    label_dir = config.label_dir
    # label_dir = config.label_dir
    cal_dir = config.cal_dir
    out_dir = config.out_box_dir

    for i, name in tqdm(enumerate(name_list), total=len(name_list)):
        img_path = os.path.join(image_root, name)
        name = os.path.splitext(os.path.basename(name))[0]
        # Load detection data
        detection_file = os.path.join(label_dir, f'{name}.txt')
        try:
            result = load_detect_data(detection_file)
        except Exception as e:
            print(f"Error reading {detection_file}: {e}")
            continue

        calfile = os.path.join(cal_dir, f'{name}.txt')
        p2 = read_kitti_cal(calfile)
        img = cv2.imread(img_path)

        for result_index in range(len(result)):
            t = result[result_index]

            if t.obj_type not in config.color_list:
                continue

            color_type = config.color_list[t.obj_type]
            cv2.rectangle(img, (t.x1, t.y1), (t.x2, t.y2), (255, 255, 255), 1)

            if t.w <= 0.05 and t.l <= 0.05 and t.h <= 0.05:  # Invalid annotation
                continue

            cam_bottom_center = [t.X, t.Y, t.Z]

            agent_rotation = R.from_euler(
                'zyx', [t.yaw, t.pitch, t.roll], degrees=False)
            agent_translation = np.array(cam_bottom_center)
            agent_rotation_matrix = agent_rotation.as_matrix()
            agent_transformation_matrix = np.eye(4)
            agent_transformation_matrix[:3, :3] = agent_rotation_matrix
            agent_transformation_matrix[:3, 3] = agent_translation

            bottom_center_in_camera = np.matrix(
                cam_bottom_center + [1.0]).T

            isCenter = 0 if config.ground_project else 1
            verts3d = get_camera_3d_8points_g2c_a9(
                t.w, t.h, t.l, bottom_center_in_camera, np.eye(4), p2, isCenter=isCenter,
                agent_in_lidar=agent_transformation_matrix)

            if verts3d is None:
                continue
            verts3d = verts3d.astype(np.int32)

            # Function to draw 3D bounding box lines
            def draw_line(img, start_point, end_point, color, thickness):
                cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)

            # Improved function to draw labels with background for better visibility
            def draw_label(img, label, label_pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_thickness=2,
                           background_color=(255, 255, 255), text_color=(0, 0, 0)):
                label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                label_rect_end = (label_pos[0] + label_size[0] + 2, label_pos[1] - label_size[1] - 2)
                label_rect_start = (label_pos[0] - 2, label_pos[1] + 2)

                # cv2.rectangle(img, label_rect_start, label_rect_end, background_color, cv2.FILLED)
                cv2.putText(img, label, (label_pos[0], label_pos[1] - 5), font, font_scale, text_color, font_thickness)

            # Draw the 3D bounding box
            edges = [(2, 1), (1, 0), (0, 3), (2, 3), (7, 4), (4, 5), (5, 6), (6, 7), (7, 3), (1, 5), (0, 4), (2, 6)]
            for start, end in edges:
                draw_line(img, verts3d[start], verts3d[end], color_type, 4)

            # Calculate label position
            label_pos = ((np.amin(verts3d[:, 0]) + np.amax(verts3d[:, 0])) // 2, np.amin(verts3d[:, 1]) - 10)
            object_label = f'{t.obj_type} {t.score:.2f}'
            depth_label = f'Depth: {t.Z:.2f}'

            # Draw object label
            draw_label(img, object_label, label_pos, font_scale=0.5, background_color=(0, 0, 0),
                       text_color=(255, 255, 255))

            # Adjust depth label position below the object label
            # depth_label_pos = (label_pos[0], label_pos[1] + 20)
            # draw_label(img, depth_label, depth_label_pos, font_scale=0.5, background_color=(0, 0, 0),
            #            text_color=(255, 255, 255))

            # Save the modified image
        cv2.imwrite(f'{out_dir}/{name}.jpg', img)



def camera_transformation(roll=-90):
    roll_angle = -90
    theta = np.radians(roll_angle)
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(theta), -np.sin(theta), 0],
                  [0, np.sin(theta), np.cos(theta), 0],
                  [0, 0, 0, 1]])
    return T


def matrix_to_xyzrpy(matrix, degrees=True):
    rotation = matrix[:3, :3]
    r = R.from_matrix(rotation)
    xyz = matrix[:3, 3].astype(float)
    euler = r.as_euler('xyz', degrees=degrees).astype(float)

    result_dict = {
        'x': float(xyz[0]),
        'y': float(xyz[1]),
        'z': float(xyz[2]),
        'roll': float(euler[0]),
        'pitch': float(euler[1]),
        'yaw': float(euler[2])
    }

    return result_dict


if __name__ == '__main__':
    if config.val_list is None:
        name_list = gb.glob(config.label_dir + "/*")
    else:
        val_part_list = open(config.val_list).readlines()
        name_list = []
        for name in val_part_list:
            name_list.append(name.split('\n')[0] + '.jpg')
        name_list.sort()
        # name_list = ['008628.png', '023338.png', '016843.png']

    if not os.path.isdir(config.out_box_dir):
        os.makedirs(config.out_box_dir)

    # -----------------------------------------------------------------------------------------------------
    # --------------Two approaches can be adopted for projection and visualization------------------------
    # 'Ground': using the ground to camera transformation (denorm: the ground plane equation), default
    # 'World': using the extrinsics (world to camera transformation)
    # show_box_with_roll(name_list, projectMethod='Ground')
    show_box_with_roll(name_list=name_list)

    image_folder = config.out_box_dir
    output_video = os.path.join(image_folder, 'video.mp4')
    create_video_from_images(image_folder, output_video)
