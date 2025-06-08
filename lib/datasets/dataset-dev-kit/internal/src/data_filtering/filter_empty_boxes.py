import argparse
import glob
import json
import os

import numpy as np

from src.eval.evaluation import get_attribute_by_name

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_folder_path_boxes', type=str, help='Path to input folder', default='')
    # add output folder
    argparser.add_argument('--output_folder_path_boxes', type=str, help='Path to output folder', default='')
    args = argparser.parse_args()
    input_folder_path_boxes = args.input_folder_path_boxes
    # add output folder
    output_folder_path_boxes = args.output_folder_path_boxes

    # iterate over all files in input folder
    for file_path_label in glob.glob(os.path.join(input_folder_path_boxes, "*.json")):
        valid_objects = {}
        # load json file
        data_json = json.load(open(file_path_label))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # iterate over all objects
            for object_id, label in frame_obj["objects"].items():
                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")
                lidar_points = 0
                if attribute is not None:
                    lidar_points = attribute["val"]
                if lidar_points > 0:
                    # keep valid box with key value pair (object_id and label)
                    valid_objects[object_id] = label

        # update data_json with valid objects
        data_json["openlabel"]["frames"][frame_id]["objects"] = valid_objects
        file_name_label = os.path.basename(file_path_label)
        # write json file
        with open(os.path.join(output_folder_path_boxes, file_name_label), "w") as f:
            json.dump(data_json, f)
