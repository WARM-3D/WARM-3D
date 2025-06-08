import argparse
import glob
import json
import os
from uuid import uuid4

import numpy as np
import torch
import torchvision.ops.boxes as bops
from scipy.optimize import linear_sum_assignment
from src.utils.utils import get_cuboid_corners
from src.utils.vis_utils import VisualizationUtils


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


def assignment(objects_box3d, objects_box2d, iou_threshold, utils=None, camera_id=None, lidar_id=None,
               boxes_coordinate_system_origin=None):
    iou_dst = np.ones((len(objects_box3d), len(objects_box2d)))
    for idx_target, object_box3d in enumerate(objects_box3d):
        points_3d = get_cuboid_corners(object_box3d["object_data"]["cuboid"]["val"])
        # project 3D box to 2D image plane
        points_2d = utils.project_3d_box_to_2d(points_3d, camera_id, lidar_id, boxes_coordinate_system_origin)
        # TODO: check shape
        if points_2d is None:
            continue
        if points_2d.shape[0] != 2 or points_2d.shape[1] != 8:
            continue
        # get 2D bounding box
        x_min = min(points_2d[0, :])
        y_min = min(points_2d[1, :])
        x_max = max(points_2d[0, :])
        y_max = max(points_2d[1, :])

        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        height, width, x_center, y_center = crop_to_image(height, width, x_center, y_center)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        box3d_projected_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float)

        for idx_source, object_2d in enumerate(objects_box2d):
            # source
            boxes2d = object_2d["object_data"]["bbox"]
            # get full bbox
            full_bbox = None
            for box2d in boxes2d:
                if box2d["name"] == "full_bbox" and box2d["attributes"]["text"][0][
                    "val"] == camera_id:
                    full_bbox = box2d["val"]
                    break

            x_center = full_bbox[0]
            y_center = full_bbox[1]
            width = full_bbox[2]
            height = full_bbox[3]

            height, width, x_center, y_center = crop_to_image(height, width, x_center, y_center)

            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

            box2d_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float)
            iou = bops.box_iou(box3d_projected_tensor, box2d_tensor)
            iou_dst[idx_target, idx_source] = 1 - iou

    indices_target, indices_source = linear_sum_assignment(iou_dst)
    indices = np.column_stack((indices_target, indices_source))

    unmatched_indices_target, unmatched_indices_source = [], []
    for idx_target, object_box3d in enumerate(objects_box3d):
        if idx_target not in indices[:, 0]:
            unmatched_indices_target.append(idx_target)

    for idx_source, object_2d in enumerate(objects_box2d):
        if idx_source not in indices[:, 1]:
            unmatched_indices_source.append(idx_source)

    matched_indices = []
    for idx in indices:
        if iou_dst[idx[0], idx[1]] > iou_threshold:
            unmatched_indices_target.append(idx[0])
            unmatched_indices_source.append(idx[1])
        else:
            matched_indices.append(idx.reshape(1, 2))

    return unmatched_indices_target, unmatched_indices_source, matched_indices


def crop_to_image(height, width, x_center, y_center):
    # crop box to image ROI
    if int(x_center - width / 2.0) < 0:
        # update x_center, in case it is negative
        x_max = int(x_center + width / 2.0)
        x_center = int(x_max / 2.0)
        width = x_max
    if int(x_center + width / 2.0) > 1920:
        # update x_center, in case it is larger than image width
        x_min = int(x_center - width / 2.0)
        width = 1920 - x_min
        x_center = x_min + int(width / 2.0)
    if int(y_center - height / 2.0) < 0:
        # update y_center, in case it is negative
        y_max = int(y_center + height / 2.0)
        y_center = int(y_max / 2.0)
        height = y_max
    if int(y_center + height / 2.0) > 1200:
        # update y_center, in case it is larger than image height
        y_min = int(y_center - height / 2.0)
        height = 1200 - y_min
        y_center = y_min + int(height / 2.0)
    return height, width, x_center, y_center


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_2d_box_labels", type=str, help="Path to 2d box labels input folder",
                            default="")
    arg_parser.add_argument("--input_folder_path_3d_box_labels", type=str,
                            help="Path to 3d box labels input folder",
                            default="")
    arg_parser.add_argument("--camera_id", type=str, help="Sensor ID")
    arg_parser.add_argument("--lidar_id", type=str, help="Sensor ID")
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")

    args = arg_parser.parse_args()
    camera_id = args.camera_id
    lidar_id = args.lidar_id

    utils = VisualizationUtils()

    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    input_file_paths_2d_box_labels = sorted(glob.glob(args.input_folder_path_2d_box_labels + "/*.json"))
    input_file_paths_3d_box_labels = sorted(
        glob.glob(args.input_folder_path_3d_box_labels + "/*.json"))

    frame_idx = None
    for input_file_path_2d_box_labels, input_file_path_3d_box_labels in zip(input_file_paths_2d_box_labels,
                                                                            input_file_paths_3d_box_labels):
        print("input_file_path_3d_box_labels: ", input_file_path_3d_box_labels)
        box3d_labels = []
        box3d_data_json = json.load(open(input_file_path_3d_box_labels))
        for frame_id, frame_obj in box3d_data_json["openlabel"]["frames"].items():
            print("num ob 3d boxes: ", len(frame_obj["objects"]))
            for object_id, label in frame_obj["objects"].items():
                box3d_labels.append(label)

        box2d_labels = []
        box2d_data_json = json.load(open(input_file_path_2d_box_labels))
        for frame_id, frame_obj in box2d_data_json["openlabel"]["frames"].items():
            frame_idx = frame_id
            print("num ob 2d boxes: ", len(frame_obj["objects"]))
            for object_id, label in frame_obj["objects"].items():
                box2d_labels.append(label)

        # 3. step: associate box3d labels with box2d labels based on 2D IoU
        unmatched_indices_target, unmatched_indices_source, matched_indices = assignment(
            objects_box3d=box3d_labels, objects_box2d=box2d_labels, iou_threshold=0.7, utils=utils, camera_id=camera_id,
            lidar_id=lidar_id, boxes_coordinate_system_origin="s110_lidar_ouster_south")

        fused_object_list = {}

        print("num. of unmatched 3d boxes: ", len(unmatched_indices_target))
        for unmatched_idx in unmatched_indices_target:
            box3d_object = box3d_labels[unmatched_idx]
            # generate new uuid
            # TODO: use existing uuid
            uuid = str(uuid4())
            fused_object_list[uuid] = box3d_object

        print("num. of unmatched 2d boxes: ", len(unmatched_indices_source))
        for unmatched_idx in unmatched_indices_source:
            box2d_object = box2d_labels[unmatched_idx]
            uuid = str(uuid4())
            # TODO: use existing uuid
            fused_object_list[uuid] = box2d_object

        print("num. of matched 2d/3d boxes: ", len(matched_indices))
        for matched_idx in matched_indices:
            box2d_object = box2d_labels[matched_idx[0, 1]]
            box3d_object = box3d_labels[matched_idx[0, 0]]
            # add 3d cuboid data with attributes to box2d object
            box2d_object["object_data"]["cuboid"] = box3d_object["object_data"]["cuboid"]
            # TODO: use existing uuid
            uuid = str(uuid4())
            fused_object_list[uuid] = box2d_object

        # 4. step: save new labels
        # update objects:
        print("num. of fused 2d+3d boxes: ", len(fused_object_list))
        box2d_data_json["openlabel"]["frames"][str(frame_idx)]["objects"] = fused_object_list
        with open(args.output_folder_path_labels + "/" + os.path.basename(input_file_path_3d_box_labels),
                  "w") as f:
            json.dump(box2d_data_json, f, indent=4, sort_keys=True)
