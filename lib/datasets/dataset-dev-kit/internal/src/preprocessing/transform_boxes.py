import os
import json
from decimal import Decimal
import argparse

import numpy as np
from scipy.spatial.transform import Rotation


# Example usage:
# python merge_boxes_late_fusion.py --input_folder_path_source1_boxes <INPUT_FOLDER_PATH_SOURCE1_BOXES>
#                              --input_folder_path_source2_boxes <INPUT_FOLDER_PATH_SOURCE2_BOXES>
#                              --output_folder_path_fused_boxes <OUTPUT_FOLDER_PATH_FUSED_BOXES>

# Note:
# The result will be stored in source1.


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        # 👇️ if passed in object is instance of Decimal
        # convert it to a string
        if isinstance(obj, Decimal):
            return str(obj)
        # 👇️ otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def parse_parameters():
    parser = argparse.ArgumentParser(description="Merge 3D boxes (late fusion)")
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        default="",
        help="input folder path to source1 boxes",
    )
    parser.add_argument(
        "--output_folder_path_boxes",
        type=str,
        default="",
        help="output folder path to fused boxes.",
    )
    parser.add_argument(
        "--source_cs",
        type=str,
        default="",
        help="source coordinate system.",
    )
    parser.add_argument(
        "--target_cs",
        type=str,
        default="",
        help="target coordinate system.",
    )
    args = parser.parse_args()
    return args


def store_boxes_in_open_label(json_data, output_folder_path, output_file_name):
    with open(os.path.join(output_folder_path, output_file_name), "w", encoding="utf-8") as json_writer:
        json_string = json.dumps(json_data, ensure_ascii=True, indent=4, cls=DecimalEncoder)
        json_writer.write(json_string)


if __name__ == "__main__":
    args = parse_parameters()
    input_folder_path_boxes = args.input_folder_path_boxes
    output_folder_path_boxes = args.output_folder_path_boxes
    source_cs = args.source_cs
    target_cs = args.target_cs

    if not os.path.exists(output_folder_path_boxes):
        os.mkdir(output_folder_path_boxes)

    input_files_source = sorted(os.listdir(input_folder_path_boxes))

    frame_id = 0
    for boxes_file_name_source in input_files_source:
        north_objects = []
        json_label_source = json.load(
            open(
                os.path.join(input_folder_path_boxes, boxes_file_name_source),
            )
        )
        # get all 3d objects
        objects_3d_source_list = []
        for frame_idx, frame_obj in json_label_source["openlabel"]["frames"].items():
            for obj_id, objects_3d_source in frame_obj["objects"].items():
                objects_3d_source_list.append(objects_3d_source)
                # get 3d position from south and north
                position_3d_source = np.array(objects_3d_source["object_data"]["cuboid"]["val"])[:3]
                # TODO: use calib files
                # transformation lidar south to s110 base
                transformation_matrix = np.array(
                    [
                        [9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e00],
                        [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e01],
                        [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ],
                    dtype=float,
                )
                # get quaternion from source1
                quaternion_source = np.array(objects_3d_source["object_data"]["cuboid"]["val"])[3:7]
                roll_source, pitch_source, yaw_source = Rotation.from_quat(quaternion_source).as_euler(
                    "xyz", degrees=True
                )
                delta_yaw = 177.90826842 - 163.58774077
                if source_cs == "s110_lidar_ouster_south" and target_cs == "s110_lidar_ouster_north":
                    # transformation matrix from lidar north to lidar south
                    transformation_matrix = np.linalg.inv(transformation_matrix)

                position_3d_source = np.matmul(
                    transformation_matrix,
                    np.array(
                        [
                            position_3d_source[0],
                            position_3d_source[1],
                            position_3d_source[2],
                            1,
                        ]
                    ),
                )
                # convert to quaternion
                quaternion = Rotation.from_euler("xyz", [roll_source, pitch_source, yaw_source], degrees=True).as_quat()
                # extract object dimensions
                dimensions = np.array(objects_3d_source["object_data"]["cuboid"]["val"])[7:10]
                # store values in object_3d_south
                objects_3d_source["object_data"]["cuboid"]["val"] = [
                    position_3d_source[0],
                    position_3d_source[1],
                    position_3d_source[2],
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                    dimensions[0],
                    dimensions[1],
                    dimensions[2],
                ]
            # use input file name as output file name and use target sensor ID as name
            output_file_name = boxes_file_name_source.replace(source_cs, target_cs)
            store_boxes_in_open_label(json_label_source, output_folder_path_boxes, output_file_name)
            frame_id += 1
