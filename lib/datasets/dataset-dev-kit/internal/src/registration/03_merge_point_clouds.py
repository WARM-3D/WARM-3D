import argparse
import glob
import json
import os
import open3d as o3d
import numpy as np

# This script is for merging two point clouds
# Usage:
# python 03_merge_point_clouds.py \
#                         --input_folder_path_transformation_matrices input/transformation_matrices \
#                         --input_folder_path_point_clouds_source input/point_clouds_source \
#                         --input_folder_path_point_clouds_target input/point_clouds_target \
#                         --output_folder_path_point_clouds_registered output/point_clouds_registered


from src.utils.point_cloud_registration_utils import (
    read_point_cloud_with_intensity,
    write_point_cloud_with_intensity,
    filter_point_cloud,
)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_transformation_matrices",
        default="input/transformation_matrices",
        type=str,
        help="Input folder path to transformation matrices",
    )
    arg_parser.add_argument(
        "--input_folder_path_point_clouds_source",
        default="input/point_clouds_source",
        type=str,
        help="Input folder path to source point clouds",
    )
    arg_parser.add_argument(
        "--input_folder_path_point_clouds_target",
        default="input/point_clouds_target",
        type=str,
        help="Input folder path to target point clouds",
    )
    arg_parser.add_argument(
        "--output_folder_path_point_clouds_registered",
        default="output/point_clouds_registered",
        type=str,
        help="Output folder path to registered point clouds",
    )
    args = arg_parser.parse_args()
    input_folder_path_transformation_matrices = args.input_folder_path_transformation_matrices
    input_folder_path_point_clouds_source = args.input_folder_path_point_clouds_source
    input_folder_path_point_clouds_target = args.input_folder_path_point_clouds_target
    output_folder_path_point_clouds_registered = args.output_folder_path_point_clouds_registered

    if not os.path.exists(output_folder_path_point_clouds_registered):
        os.makedirs(output_folder_path_point_clouds_registered)

    # read all transformation matrices
    transformation_matrices = []
    for file_path in sorted(glob.glob(os.path.join(input_folder_path_transformation_matrices, "*.json"))):
        transformation_matrices.append(np.array(json.load(open(file_path))["transformation_matrix"]))
    # read all point clouds using open3d
    point_clouds_source = []
    intensities_source = []
    for file_path_source in sorted(glob.glob(os.path.join(input_folder_path_point_clouds_source, "*.pcd"))):
        # read point clouds with intensities
        point_cloud_source_array, header = read_point_cloud_with_intensity(file_path_source)
        point_cloud_source_array = filter_point_cloud(point_cloud_source_array, 200)
        intensity_source = point_cloud_source_array[:, 3]
        # normalize intensity
        intensity_source = intensity_source / np.max(intensity_source)
        intensities_source.append(intensity_source)
        point_clouds_source.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_source_array[:, :3])))

    point_clouds_target = []
    intensities_target = []
    output_file_names = []
    for file_path in sorted(glob.glob(os.path.join(input_folder_path_point_clouds_target, "*.pcd"))):
        # read point clouds with intensities
        point_cloud_array, header = read_point_cloud_with_intensity(file_path)
        point_cloud_array = filter_point_cloud(point_cloud_array, 120)
        intensity_target = point_cloud_array[:, 3]
        # normalize intensity
        intensity_target = intensity_target / np.max(intensity_target)
        intensities_target.append(intensity_target)
        point_clouds_target.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_array[:, :3])))
        output_file_names.append(os.path.basename(file_path))
    # register all point clouds
    point_clouds_registered = []
    for i in range(len(point_clouds_source)):
        point_cloud_source_transformed = point_clouds_source[i].transform(transformation_matrices[i])
        # merge point clouds (source and target)
        point_cloud_merged = point_cloud_source_transformed + point_clouds_target[i]
        # merge intensities (source and target)
        intensity_merged = np.concatenate((intensities_source[i], intensities_target[i]))
        # add intensity to point cloud using numpy
        point_cloud_merged_array = np.concatenate(
            (np.asarray(point_cloud_merged.points), intensity_merged[:, None]), axis=1
        )
        point_clouds_registered.append(point_cloud_merged_array)

    # save all registered point clouds. Use output file name from target point clouds
    for point_cloud_registered, output_file_name in zip(point_clouds_registered, output_file_names):
        # write point clouds with intensities
        write_point_cloud_with_intensity(
            os.path.join(output_folder_path_point_clouds_registered, output_file_name),
            point_cloud_registered,
        )
