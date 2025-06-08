"""
This script is for interpolation between two transformation matrices

# Usage:
# python 02_interpolate_transformation_matrix.py \
                        --input_folder_path_transformation_matrices input/transformation_matrices \
                        --input_folder_path_point_clouds_source input/point_clouds_source \
                        --output_folder_path_interpolated_transformation_matrices output/interpolated_transformation_matrices \
                        --nb_interpolated_transformation_matrices [10, 100]

# NOTE:
# 1) use --nb_interpolated_transformation_matrices 10 if every 10th frame is used for registration (= 11 key transformations calculated in total out of 100 frames)
# 2) use --nb_interpolated_transformation_matrices 100 if every 100th frame is used for registration (= 2 key transformations calculated in total out of 100 frames)
"""
import argparse
import glob
import os.path
from os.path import join
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input_folder_path_transformation_matrices",
        default="input/transformation_matrices",
        type=str,
        help="Input folder path to transformation matrices",
    )
    argparser.add_argument(
        "--input_folder_path_point_clouds_source",
        default="input/point_clouds_source",
        type=str,
        help="Input folder path to source point clouds",
    )
    argparser.add_argument(
        "--output_folder_path_interpolated_transformation_matrices",
        default="output/interpolated_transformation_matrices",
        type=str,
        help="Output folder path to interpolated transformation matrices",
    )
    argparser.add_argument(
        "--nb_interpolated_transformation_matrices",
        default="10",
        type=int,
        help="Number of interpolated transformation matrices between two transformation matrices",
    )
    # read transformation matrices
    args = argparser.parse_args()
    input_folder_path_transformation_matrices = args.input_folder_path_transformation_matrices
    input_folder_path_point_clouds_source = args.input_folder_path_point_clouds_source
    output_folder_path_interpolated_transformation_matrices = (
        args.output_folder_path_interpolated_transformation_matrices
    )
    nb_interpolated_transformation_matrices = args.nb_interpolated_transformation_matrices

    # create output folder if not exists
    if not os.path.exists(output_folder_path_interpolated_transformation_matrices):
        os.makedirs(output_folder_path_interpolated_transformation_matrices)

    file_names_output = []
    for file_name in sorted(glob.glob(join(input_folder_path_point_clouds_source, "*.pcd"))):
        file_names_output.append(os.path.basename(file_name))

    # read transformation matrices
    transformation_matrices_input = []
    transformation_matrices_interpolated = []
    for file_path in sorted(glob.glob(join(input_folder_path_transformation_matrices, "*.json"))):
        # load transformation matrix from json file
        transformation_matrix = np.array(json.load(open(file_path))["transformation_matrix"])
        transformation_matrices_input.append(transformation_matrix)

    for idx_fname in range(len(transformation_matrices_input) - 1):
        transformation_matrix_1 = transformation_matrices_input[idx_fname]
        transformation_matrix_2 = transformation_matrices_input[idx_fname + 1]

        # extract rotation matrix
        rotation_matrix_1 = transformation_matrix_1[:3, :3]
        rotation_matrix_2 = transformation_matrix_2[:3, :3]

        # extract translation vector
        translation_vector_1 = transformation_matrix_1[:3, 3]
        translation_vector_2 = transformation_matrix_2[:3, 3]

        # extract quaternion
        quaternion_1 = R.from_matrix(rotation_matrix_1).as_quat()
        quaternion_2 = R.from_matrix(rotation_matrix_2).as_quat()

        key_rotations = R.from_quat([quaternion_1, quaternion_2])

        print("key rotation of start and end frame (euler angles in degrees): ",
              key_rotations.as_euler("xyz", degrees=True))

        # interpolate only 8 frames instead of 9 frames inbetween when using last frame
        if idx_fname == len(transformation_matrices_input) - 2:
            key_times = [0, nb_interpolated_transformation_matrices - 1]
            times = np.arange(1, nb_interpolated_transformation_matrices - 1)
        else:
            key_times = [0, nb_interpolated_transformation_matrices]
            times = np.arange(1, nb_interpolated_transformation_matrices)

        slerp = Slerp(key_times, key_rotations)
        interp_rots = slerp(times)

        print("interpolated quaternions:", interp_rots.as_euler("xyz", degrees=True))

        interpolated_rotations = []
        for idx, tf in enumerate(interp_rots):
            interpolated_rotations.append(tf.as_matrix())

        # interpolate translation vectors
        key_tsl = np.vstack([translation_vector_1, translation_vector_2])

        print("key translations (start and end): ", key_tsl)

        if idx_fname == len(transformation_matrices_input) - 2:
            key_times = [0, nb_interpolated_transformation_matrices - 1]
            times = np.arange(1, nb_interpolated_transformation_matrices - 1)
        else:
            key_times = [0, nb_interpolated_transformation_matrices]
            times = np.arange(1, nb_interpolated_transformation_matrices)

        linfit = interp1d(key_times, key_tsl, axis=0)
        interp_tsl = linfit(times)

        print("interpolated translations: ", interp_tsl)

        transformation_matrices_interpolated.append(transformation_matrix_1)
        for idx, tsl in enumerate(interp_tsl):
            # create 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = interpolated_rotations[idx]
            transformation_matrix[:3, 3] = tsl
            transformation_matrices_interpolated.append(transformation_matrix)
        if idx_fname == len(transformation_matrices_input) - 2:
            transformation_matrices_interpolated.append(transformation_matrix_2)

    for output_file_name, transformation_matrix_interpolated in zip(
            file_names_output, transformation_matrices_interpolated
    ):
        # save interpolated transformation matrices to json files (output_folder_path_interpolated_transformation_matrices)
        json.dump(
            {"transformation_matrix": np.around(transformation_matrix_interpolated, decimals=7).tolist()},
            open(
                join(
                    output_folder_path_interpolated_transformation_matrices, output_file_name.replace(".pcd", ".json")
                ),
                "w",
            ),
        )
