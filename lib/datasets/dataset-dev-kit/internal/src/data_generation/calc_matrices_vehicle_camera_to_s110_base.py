import argparse
import glob
import json
import os
from os.path import join

import numpy as np

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '--input_folder_path_transformation_matrices',
        default='input/transformation_matrices',
        type=str,
        help='Input folder path to transformation matrices',
    )
    arg_parser.add_argument(
        '--output_folder_path_transformation_matrices',
        default='output/transformation_matrices',
        type=str,
        help='Output folder path to transformation matrices',
    )
    args = arg_parser.parse_args()
    input_folder_path_transformation_matrices = args.input_folder_path_transformation_matrices
    output_folder_path_transformation_matrices = args.output_folder_path_transformation_matrices

    # create output folder if not exists
    if not os.path.exists(output_folder_path_transformation_matrices):
        os.makedirs(output_folder_path_transformation_matrices)

    transformation_matrix_vehicle_camera_to_vehicle_lidar_robosense = np.array([
        [0.126729, -0.991225, 0.0375934, 0.177805],
        [0.123777, -0.0218005, -0.992071, -0.0357065],
        [0.984184, 0.130377, 0.119928, -0.166477],
        [0, 0, 0, 1]
    ])
    transformation_matrix_vehicle_lidar_robosense_to_vehicle_camera = np.linalg.inv(
        transformation_matrix_vehicle_camera_to_vehicle_lidar_robosense)

    transformation_matrix_s110_lidar_ouster_south_to_s110_base = np.array([
        [0.21479485, -0.9761028, 0.03296187, -15.87257873],
        [0.97627128, 0.21553835, 0.02091894, 2.30019086],
        [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
        [0, 0, 0, 1]
    ])
    # 1) read all transformation matrices (infra_lidar to vehicle_lidar)
    for input_file_path in sorted(glob.glob(join(input_folder_path_transformation_matrices, '*.json'))):
        input_file_name = input_file_path.split('/')[-1]
        with open(input_file_path, "r") as json_file:
            json_data = json.load(json_file)
        transformation_matrix_vehicle_lidar_robosense_to_s110_lidar_ouster_south = np.array(
            json_data["transformation_matrix"])
        transformation_matrix_s110_lidar_ouster_south_to_vehicle_lidar_robosense = np.linalg.inv(
            transformation_matrix_vehicle_lidar_robosense_to_s110_lidar_ouster_south)

        del json_data["transformation_matrix"]
        json_data["transformation_matrix_vehicle_lidar_robosense_to_s110_lidar_ouster_south"] = \
            transformation_matrix_vehicle_lidar_robosense_to_s110_lidar_ouster_south.tolist()

        json_data["transformation_matrix_s110_lidar_ouster_south_to_vehicle_lidar_robosense"] = \
            transformation_matrix_s110_lidar_ouster_south_to_vehicle_lidar_robosense.tolist()

        # 2) calculate transformation matrices from vehicle_camera to s110_base (vehicle_camera -> vehicle_lidar -> s110_lidar_ouster_south -> s110_base)
        transformation_matrix_vehicle_camera_to_s110_base = transformation_matrix_s110_lidar_ouster_south_to_s110_base @ \
                                                            transformation_matrix_vehicle_lidar_robosense_to_s110_lidar_ouster_south @ \
                                                            transformation_matrix_vehicle_camera_to_vehicle_lidar_robosense
        transformation_matrix_s110_base_to_vehicle_camera = np.linalg.inv(
            transformation_matrix_vehicle_camera_to_s110_base)
        # 3) add both transformation matrices to json_data and store json content to file
        json_data[
            "transformation_matrix_vehicle_camera_to_s110_base"] = transformation_matrix_vehicle_camera_to_s110_base.tolist()
        json_data[
            "transformation_matrix_s110_base_to_vehicle_camera"] = transformation_matrix_s110_base_to_vehicle_camera.tolist()
        json.dump(json_data, open(input_file_path, 'w'), indent=4)

        # 4) calculate projection matrices from vehicle_camera to s110_base
        optimal_intrinsic_camera_matrix_for_lidar_projection = np.array([[2726.55, 0, 685.235],
                                                                         [0, 2676.64, 262.745],
                                                                         [0, 0, 1]])
        extrinsic_matrix_s110_base_to_vehicle_camera_16mm = transformation_matrix_s110_base_to_vehicle_camera[:3, :]
        projection_matrix_s110_base_to_vehicle_camera_basler_16mm = optimal_intrinsic_camera_matrix_for_lidar_projection @ extrinsic_matrix_s110_base_to_vehicle_camera_16mm
        # 5) add projection matrix to json_data and store json content to file
        json_data[
            "projection_matrix_s110_base_to_vehicle_camera_basler_16mm"] = projection_matrix_s110_base_to_vehicle_camera_basler_16mm.tolist()
        json.dump(json_data, open(join(output_folder_path_transformation_matrices, input_file_name), 'w'), indent=4)
