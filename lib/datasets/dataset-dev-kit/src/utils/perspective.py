from typing import Tuple, Optional, Union, Dict
import numpy as np
from pathlib import Path
import sys
import os
import json
from scipy.spatial.transform import Rotation as R

PERCEPTION_SHARED_DIRECTORY = Path(__file__).absolute().parent


class Perspective:
    """
    This class represents a single camera perspective including
    utility functions for translation between camera- and ground frame.

    This class relies on:
    1. rotation matrix, from ground- to camera frame (3x3)
    2. translation (i.e. camera position) in ground frame (3x1)
    3. intrinsic matrix of the camera (3x3)

    Additionally, the image shape (height, width) is stored.
    """

    def __init__(
            self,
            rotation_matrix: np.ndarray,
            translation: np.ndarray,
            intrinsic_matrix: np.ndarray,
            image_shape: Tuple[int, int],
            projection_from_lidar_south: Optional[np.ndarray] = None,
            projection_from_lidar_north: Optional[np.ndarray] = None,
            transformation_matrix_s110_lidar_ouster_south_to_s110_base: Optional[np.ndarray] = None,
            transformation_matrix_from_camera_to_lidar: Optional[np.ndarray] = None
    ):
        assert rotation_matrix.shape == (3, 3)
        assert translation.shape == (3, 1)
        self.rotation_matrix = rotation_matrix
        self.translation = translation
        self.intrinsic_matrix = intrinsic_matrix
        self.image_shape = image_shape
        self.projection_from_lidar_south = projection_from_lidar_south
        self.projection_from_lidar_north = projection_from_lidar_north
        self.transformation_matrix_s110_liar_ouster_south_to_s110_base = transformation_matrix_s110_lidar_ouster_south_to_s110_base
        self.transformation_matrix_from_camera_to_lidar = transformation_matrix_from_camera_to_lidar
        self.initialize_matrices()

    def initialize_matrices(self):
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        self.extrinsic_matrix = np.hstack((self.rotation_matrix, -self.rotation_matrix @ self.translation))
        self.projection_matrix = self.intrinsic_matrix @ self.extrinsic_matrix

    def project_to_ground(self, image_points: np.ndarray, ground_plane_height=0) -> np.ndarray:
        """Project image points (2xn) into ground frame (3xn) i.e. z=0."""
        assert image_points.shape[0] == 2
        # "Transform" points into ground frame.
        # The real ground point is somewhere on the line going through the camera position and the respective point.
        augmented_points = np.vstack((image_points, np.ones(image_points.shape[1])))
        ground_points = self.rotation_matrix.T @ self.inv_intrinsic_matrix @ augmented_points
        # Find intersection of line with ground plane i.e. z=0.
        ground_points *= -(self.translation[2] - ground_plane_height) / ground_points[2]
        ground_points += self.translation
        return ground_points

    def project_from_base_to_image(self, ground_points: np.ndarray, filter_behind: bool = False):
        """Project ground points (3xn) into image frame (2xn)."""
        # Transform points using the projection matrix.
        augmented_points = np.vstack((ground_points, np.ones(ground_points.shape[1])))
        transformed_points = self.projection_matrix @ augmented_points

        # Filter out points that are behind camera
        if filter_behind:
            transformed_points = transformed_points[:, transformed_points[2] > 0]

        # Divide x and y values by z (camera pinhole model).
        image_points = transformed_points[:2] / transformed_points[2]

        return image_points

    def project_from_lidar_south_to_image(self, lidar_pts: np.ndarray):
        """Project lidar xyz to image frame."""
        assert self.projection_from_lidar_south is not None
        # Transform points using the projection matrix.
        augmented_points = np.vstack((lidar_pts, np.ones(lidar_pts.shape[1])))
        transformed_points = self.projection_from_lidar_south @ augmented_points
        # Divide x and y values by z (camera pinhole model).
        image_points = transformed_points[:2] / transformed_points[2]
        return image_points

    def project_from_lidar_north_to_image(self, lidar_pts: np.ndarray):
        """Project lidar xyz to image frame."""
        assert self.projection_from_lidar_north is not None
        # Transform points using the projection matrix.
        augmented_points = np.vstack((lidar_pts, np.ones(lidar_pts.shape[1])))
        transformed_points = self.projection_from_lidar_north @ augmented_points
        # Divide x and y values by z (camera pinhole model).
        image_points = transformed_points[:2] / transformed_points[2]
        return image_points

    def transform_from_s110_base_to_s110_lidar_ouster_south(self, location: np.ndarray, roll: float = 0.0,
                                                            pitch: float = 0.0, yaw: float = 0.0):
        """Transform location from s110 base frame to s110 lidar ouster south frame."""
        assert self.transformation_matrix_s110_liar_ouster_south_to_s110_base is not None
        box_pose = np.eye(4)
        box_pose[0:3, 3] = location[:, 0]
        box_pose[0:3, 0:3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
        box_pose = np.linalg.inv(self.transformation_matrix_s110_liar_ouster_south_to_s110_base) @ box_pose
        location_transformed = box_pose[0:3, 3]
        # make location_transformed 3x1 instead of 3,
        location_transformed = np.expand_dims(location_transformed, axis=1)
        return location_transformed, R.from_matrix(box_pose[0:3, 0:3]).as_euler("xyz", degrees=False)

    def transform_from_camera_to_lidar(self, location: np.ndarray, roll: float = 0.0, pitch: float = 0.0,
                                       yaw: float = 0.0):
        """Transform location from camera frame to lidar frame."""

        transformation_matrix = self.transformation_matrix_from_camera_to_lidar

        # Transform location from camera frame to ground frame.

        agent_rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        agent_rotation = agent_rotation.as_matrix()
        agent_transformation = np.eye(4)
        agent_transformation[:3, :3] = agent_rotation
        agent_transformation[:3, 3] = location

        agent_in_lidar = transformation_matrix @ agent_transformation

        agent_pose_rotation = agent_in_lidar[:3, :3]
        # agent_yaw, agent_pitch, agent_roll = R.from_matrix(agent_pose_rotation).as_euler('zyx')
        # agent_roll, agent_pitch, agent_yaw = R.from_matrix(agent_pose_rotation).as_euler('xyz')
        quat = R.from_matrix(agent_pose_rotation).as_quat()
        agent_pose_translation = agent_in_lidar[:3, 3]

        return agent_pose_translation, quat


def parse_perspective(perspective_file_path: str) -> Optional[Perspective]:
    """Parse single perspective from JSON file."""

    print(f"Loading perspective from {perspective_file_path}...")

    with open(perspective_file_path) as f:
        data = json.load(f)

    # First type of calibrated json file (using on the Highway)
    keys = {"rotation_matrix", "translation", "intrinsic_matrix", "extrinsic_matrix"}

    # Second type of calibrated json file (used at the S110 intersection)
    key2 = {"rotation_matrix", "translation_matrix", "calibrated_intrinsic_camera_matrix", "projection_matrix"}

    # Third type of calibrated json file (using two different projection matrices)
    key3 = {"intrinsic", "R", "t", "projection_matrix", "image_width", "image_height"}

    # Fourth type of calibrated json file (keys are used in lidar calibration files)
    key4 = {
        "projection_matrix_into_s110_camera_basler_south1_8mm",
        "projection_matrix_into_s110_camera_basler_south2_8mm",
        "transformation_matrix_into_s110_base",
    }

    # Optional projection from lidar
    proj_from_lidar_south = None
    proj_from_lidar_north = None

    if "projections" in data:
        if data["cam"] == "s110_s1_cam_8_b":
            data = data["projections"][1]
        elif data["cam"] == "s110_s2_cam_8_b":
            data = data["projections"]

    intrinsic_matrix = None
    image_shape = None
    transformation_matrix_s110_liar_ouster_south_to_s110_base = None
    if keys.issubset(data):
        rotation_matrix = np.array(data["rotation_matrix"]).T
        translation = np.array(data["extrinsic_matrix"])[:, -1:]
        translation = -rotation_matrix.T @ translation
        intrinsic_matrix = np.array(data["intrinsic_matrix"])
        image_shape = data["image_height"], data["image_width"]

    elif key2.issubset(data):

        rotation_matrix = np.array(data["rotation_matrix"])
        intrinsic_matrix = np.array(data["calibrated_intrinsic_camera_matrix"])[0:3, 0:3]
        projection_matrix = np.array(data["projection_matrix"])
        translation_matrix = np.array(data["translation_matrix"])[:, np.newaxis]  # convert to column vector
        if "transformation_matrix_s110_lidar_ouster_south_to_s110_base" in data:
            transformation_matrix_s110_liar_ouster_south_to_s110_base = np.array(
                data["transformation_matrix_s110_lidar_ouster_south_to_s110_base"]
            )
        else:
            transformation_matrix_s110_liar_ouster_south_to_s110_base = np.eye(4)

        image_shape, proj_from_lidar_south, proj_from_lidar_north, translation = get_original_translation(
            data,
            intrinsic_matrix,
            proj_from_lidar_south,
            proj_from_lidar_north,
            projection_matrix,
            rotation_matrix,
            translation_matrix,
        )
    elif key3.issubset(data):
        rotation_matrix = np.array(data["R"])
        intrinsic_matrix = np.array(data["intrinsic"])
        projection_matrix = np.array(data["projection_matrix"])
        translation_matrix = np.array(data["t"])[:, np.newaxis]  # convert to column vector

        image_shape, proj_from_lidar_south, proj_from_lidar_north, translation = get_original_translation(
            data,
            intrinsic_matrix,
            proj_from_lidar_south,
            proj_from_lidar_north,
            projection_matrix,
            rotation_matrix,
            translation_matrix,
        )
    elif key4.issubset(data):
        transformation_matrix = np.array(data["transformation_matrix_into_s110_base"])
        rotation_matrix = transformation_matrix[:3, :3]
        translation = transformation_matrix[:3, 3]
        # convert translation 3, to 3,1
        translation = translation[:, np.newaxis]
    else:
        print("Could not parse camera calibration parameters (.json): ", perspective_file_path, "\nExiting...")
        sys.exit()

    perspective = Perspective(
        rotation_matrix, translation, intrinsic_matrix, image_shape, proj_from_lidar_south, proj_from_lidar_north,
        transformation_matrix_s110_liar_ouster_south_to_s110_base
    )
    return perspective


def get_original_translation(
        data,
        intrinsic_matrix,
        proj_from_lidar_south,
        proj_from_lidar_north,
        projection_matrix,
        rotation_matrix,
        translation_matrix,
):
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
    # TODO: remove duplicate projection matrix from config file
    # np.testing.assert_array_less(np.dot(intrinsic_matrix, extrinsic_matrix) - projection_matrix, 1e-6)
    if "projection_from_s110_lidar_ouster_south" in data:
        proj_from_lidar_south = np.array(data["projection_from_s110_lidar_ouster_south"])
    if "projection_from_s110_lidar_ouster_north" in data:
        proj_from_lidar_north = np.array(data["projection_from_s110_lidar_ouster_north"])
    translation = -rotation_matrix.T @ translation_matrix
    image_shape = data["image_height"], data["image_width"]
    return image_shape, proj_from_lidar_south, proj_from_lidar_north, translation
