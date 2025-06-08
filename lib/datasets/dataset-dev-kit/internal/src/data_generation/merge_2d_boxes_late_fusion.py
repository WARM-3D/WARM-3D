import argparse
import glob
import json
import os
from uuid import uuid4

import numpy as np
import torch
import torchvision.ops.boxes as bops
from scipy.optimize import linear_sum_assignment


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


def assignment(objects_target, objects_source, iou_threshold):
    iou_dst = np.zeros((len(objects_target), len(objects_source)))
    for idx_target, object_target in enumerate(objects_target):
        # target
        boxes2d_manually_labeled = object_target["object_data"]["bbox"]
        # get full bbox
        full_bbox = None
        for box2d_manually_labeled in boxes2d_manually_labeled:
            if box2d_manually_labeled["name"] == "full_bbox" and box2d_manually_labeled["attributes"]["text"][0][
                "val"] == sensor_id:
                full_bbox = box2d_manually_labeled["val"]
                break

        x_center, y_center, width, height = full_bbox
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        box2d_target_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float)

        for idx_source, object_source in enumerate(objects_source):
            # source
            box2d_labeling_compnay = object_source["object_data"]["bbox"][0]["val"]
            x_center = box2d_labeling_compnay[0]
            y_center = box2d_labeling_compnay[1]
            width = box2d_labeling_compnay[2]
            height = box2d_labeling_compnay[3]

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

            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

            box2d_source_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float)
            iou = bops.box_iou(box2d_target_tensor, box2d_source_tensor)
            iou_dst[idx_target, idx_source] = 1 - iou

    indices_target, indices_source = linear_sum_assignment(iou_dst)
    indices = np.column_stack((indices_target, indices_source))

    unmatched_indices_target, unmatched_indices_source = [], []
    for idx_target, object_target in enumerate(objects_target):
        if idx_target not in indices[:, 0]:
            unmatched_indices_target.append(idx_target)

    for idx_source, object_source in enumerate(objects_source):
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


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_mask_labels", type=str, help="Path to mask labels input folder",
                            default="")
    arg_parser.add_argument("--input_folder_path_box2d_labeling_company_labels", type=str,
                            help="Path to box2d labels input folder",
                            default="")
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    # add sensor ID
    arg_parser.add_argument("--sensor_id", type=str, help="Sensor ID")
    args = arg_parser.parse_args()
    sensor_id = args.sensor_id

    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    mask_label_file_paths = sorted(glob.glob(args.input_folder_path_mask_labels + "/*.json"))
    box2d_labeling_company_label_file_paths = sorted(
        glob.glob(args.input_folder_path_box2d_labeling_company_labels + "/*.json"))

    # 1. step: load all mask labels into a list
    frame_idx = None
    for mask_label_file_path, box2d_labeling_company_label_file_path in zip(mask_label_file_paths,
                                                                            box2d_labeling_company_label_file_paths):
        print("box2d_labeling_company_label_file_path: ", box2d_labeling_company_label_file_path)
        manual_labeled_object_list = []
        mask_label_data_json = json.load(open(mask_label_file_path))
        for frame_id, frame_obj in mask_label_data_json["openlabel"]["frames"].items():
            print("num ob 2d boxes (manually labeled): ", len(frame_obj["objects"]))
            for object_id, label in frame_obj["objects"].items():
                manual_labeled_object_list.append(label)

        # 2. step: load all box2d labels (from labeling company) into map
        labeling_company_object_list = []
        box2d_labeling_company_label_data_json = json.load(open(box2d_labeling_company_label_file_path))
        for frame_id, frame_obj in box2d_labeling_company_label_data_json["openlabel"]["frames"].items():
            frame_idx = frame_id
            print("num ob 2d boxes (labeled by labeling company): ", len(frame_obj["objects"]))
            for object_id, label in frame_obj["objects"].items():
                labeling_company_object_list.append(label)

        # 3. step: associate mask labels with box2d labels based on 2D IoU
        # iterate all bbox2d labels and bbox_2d_labeling_company labels and calculate 2D IoU
        # if 2D IoU > 0.5, associate bbox2d label with box2d label
        # if 2D IoU < 0.5, do not associate bbox2d label with box2d label

        unmatched_indices_target, unmatched_indices_source, matched_indices = assignment(
            objects_target=manual_labeled_object_list, objects_source=labeling_company_object_list, iou_threshold=0.7)

        fused_object_list = {}

        print("num. of unmatched 2d boxes (labeled by labeling company): ", len(unmatched_indices_target))
        for unmatched_idx in unmatched_indices_target:
            manual_labeled_object = manual_labeled_object_list[unmatched_idx]
            # generate new uuid
            uuid = str(uuid4())
            fused_object_list[uuid] = manual_labeled_object

        print("num. of unmatched 2d boxes (manually labeled): ", len(unmatched_indices_source))
        for unmatched_idx in unmatched_indices_source:
            labeling_company_object = labeling_company_object_list[unmatched_idx]
            labeling_company_object["object_data"]["bbox"][0]["name"] = "full_bbox"
            # add sensor ID to full_bbox
            labeling_company_object["object_data"]["bbox"][0]["attributes"] = {
                "text": [
                    {
                        "name": "sensor_id",
                        "val": sensor_id
                    }
                ]
            }
            # TODO: keep old uuid
            uuid = str(uuid4())
            fused_object_list[uuid] = labeling_company_object

        print("num. of matched 2d boxes: ", len(matched_indices))
        for matched_idx in matched_indices:
            labeling_company_object = labeling_company_object_list[matched_idx[0, 1]]
            manual_labeled_object = manual_labeled_object_list[matched_idx[0, 0]]
            # get visible bounding box from manually labeled object and add it to labeling company object
            visible_bbox = None
            for bbox2d_match_item in manual_labeled_object["object_data"]["bbox"]:
                if bbox2d_match_item["name"] == "visible_bbox" and bbox2d_match_item["attributes"]["text"][0][
                    "val"] == sensor_id:
                    visible_bbox = bbox2d_match_item
                    break
            # change name from shape to full_bbox
            labeling_company_object["object_data"]["bbox"][0]["name"] = "full_bbox"
            if visible_bbox is not None:
                # add visible bbox to box2d_object_labeling_compnay
                labeling_company_object["object_data"]["bbox"].append(visible_bbox)
            # add sensor ID to full_bbox
            labeling_company_object["object_data"]["bbox"][0]["attributes"] = {
                "text": [
                    {
                        "name": "sensor_id",
                        "val": sensor_id
                    }
                ]
            }
            labeling_company_object["object_data"]["poly2d"] = manual_labeled_object["object_data"]["poly2d"]
            uuid = str(uuid4())
            fused_object_list[uuid] = labeling_company_object

        # 4. step: save new labels
        # update objects:
        print("num. of fused 2d boxes: ", len(fused_object_list))
        box2d_labeling_company_label_data_json["openlabel"]["frames"][str(frame_idx)]["objects"] = fused_object_list
        with open(args.output_folder_path_labels + "/" + os.path.basename(box2d_labeling_company_label_file_path),
                  "w") as f:
            json.dump(box2d_labeling_company_label_data_json, f, indent=4, sort_keys=True)
