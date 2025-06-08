import argparse
import glob
import os
import json
import numpy as np

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r02_s01 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_file_path_lidar_north_calibration_data",
        type=str,
        help="File path to lidar north calibration data.",
        default="",
    )
    arg_parser.add_argument(
        "--input_file_path_lidar_south_calibration_data",
        type=str,
        help="File path to lidar south calibration data.",
        default="",
    )
    args = arg_parser.parse_args()

    input_folder_paths_all = []

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)

    calib_data_lidar_north = json.load(open(args.input_file_path_lidar_north_calibration_data, "r"))
    calib_data_lidar_south = json.load(open(args.input_file_path_lidar_south_calibration_data, "r"))
    transformation_matrix_lidar_north_to_s110_base = np.array(
        calib_data_lidar_north["transformation_matrix_into_s110_base"]
    )
    transformation_matrix_lidar_south_to_s110_base = np.array(
        calib_data_lidar_south["transformation_matrix_into_s110_base"]
    )
    for input_folder_path in input_folder_paths_all:
        label_file_paths = sorted(glob.glob(input_folder_path + "/*.json"))
        for file_path_label in label_file_paths:
            file_name_label = os.path.basename(file_path_label)
            label_data_json = json.load(open(file_path_label))
            if "s110_lidar_ouster_south" in input_folder_path:
                label_data_json["openlabel"]["coordinate_systems"]["s110_lidar_ouster_south"]["pose_wrt_parent"][
                    "matrix4x4"
                ] = transformation_matrix_lidar_south_to_s110_base.flatten().tolist()
            elif "s110_lidar_ouster_north" in input_folder_path:
                label_data_json["openlabel"]["coordinate_systems"]["s110_lidar_ouster_north"]["pose_wrt_parent"][
                    "matrix4x4"
                ] = transformation_matrix_lidar_north_to_s110_base.flatten().tolist()
            else:
                raise ValueError("Unknown lidar")

            for frame_id, frame_obj in label_data_json["openlabel"]["frames"].items():
                if "s110_lidar_ouster_south" in input_folder_path:
                    frame_obj["frame_properties"]["transforms"]["s110_lidar_ouster_south_to_s110_base"][
                        "transform_src_to_dst"
                    ]["matrix4x4"] = (np.linalg.inv(transformation_matrix_lidar_south_to_s110_base).flatten().tolist())
                elif "s110_lidar_ouster_north" in input_folder_path:
                    frame_obj["frame_properties"]["transforms"]["s110_lidar_ouster_north_to_s110_base"][
                        "transform_src_to_dst"
                    ]["matrix4x4"] = (np.linalg.inv(transformation_matrix_lidar_north_to_s110_base).flatten().tolist())
                else:
                    raise ValueError("Unknown lidar")

            output_folder_path = input_folder_path.replace("_backup", "")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # write json file
            with open(os.path.join(output_folder_path, file_name_label), "w") as f:
                json.dump(label_data_json, f)
