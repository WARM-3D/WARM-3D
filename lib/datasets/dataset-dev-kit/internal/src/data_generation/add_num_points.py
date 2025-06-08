import argparse
import glob
import json
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def get_num_points(point_cloud, position_3d, rotation_yaw, dimensions):
    # TODO: check whether the other sin needs to be negated instead
    obb = o3d.geometry.OrientedBoundingBox(
        position_3d,
        np.array(
            [
                [np.cos(rotation_yaw), -np.sin(rotation_yaw), 0],
                [np.sin(rotation_yaw), np.cos(rotation_yaw), 0],
                [0, 0, 1],
            ]
        ),
        dimensions,
    )
    num_points = len(obb.get_point_indices_within_bounding_box(point_cloud.points))
    return num_points


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_labels", type=str, help="Path to labels input folder", default="")
    arg_parser.add_argument(
        "--input_folder_path_point_clouds", type=str, help="Folder Path to point clouds", default=""
    )
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    args = arg_parser.parse_args()
    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    label_file_paths = sorted(glob.glob(args.input_folder_path_labels + "/*.json"))
    point_cloud_file_paths = sorted(glob.glob(args.input_folder_path_point_clouds + "/*.pcd"))
    # iterate over all files in input folder
    for file_path_label, file_path_point_cloud in zip(label_file_paths, point_cloud_file_paths):
        file_name_label = os.path.basename(file_path_label)
        file_name_point_cloud = os.path.basename(file_path_point_cloud)
        # load point cloud using open3d
        point_cloud = o3d.io.read_point_cloud(file_path_point_cloud)

        # load json file
        data_json = json.load(open(file_path_label))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # iterate over all objects
            for object_id, label in frame_obj["objects"].items():
                cuboid = label["object_data"]["cuboid"]["val"]
                quaternion = np.array(cuboid[3:7])
                rotation_yaw = Rotation.from_quat(quaternion).as_euler("xyz")[2]
                num_lidar_points = get_num_points(point_cloud, cuboid[:3], rotation_yaw, cuboid[7:10])
                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")
                # check if num_points exist in num
                if attribute is None:
                    num_lidar_points_attribute = {"name": "num_points", "val": num_lidar_points}
                    label["object_data"]["cuboid"]["attributes"]["num"].append(num_lidar_points_attribute)
                else:
                    # num_points already exist, overwrite it
                    attribute["val"] = num_lidar_points
        # write json file
        with open(os.path.join(args.output_folder_path_labels, file_name_label), "w") as f:
            json.dump(data_json, f)
