import argparse
import glob
import json
import os
import numpy as np

default_dimensions = {
    "CAR": [
        4.70,
        2.09,
        1.45
    ],
    "TRUCK": [
        6.32,
        2.59,
        3.96
    ],
    "BUS": [
        11.95,
        2.55,
        2.99
    ],
    "VAN": [
        7.00,
        2.24,
        2.67
    ],
    "BICYCLE": [
        1.70,
        0.508,
        1.50
    ],
    "PEDESTRIAN": [
        1.75,
        0.50,
        0.50
    ],
    "MOTORCYCLE": [
        2.00,
        0.50,
        1.50
    ],
    "TRAILER": [
        16.00,
        2.59,
        3.96
    ],
    "SPECIAL_VEHICLE": [
        7.00,
        2.24,
        2.67
    ]
}

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_boxes", type=str, help="Path to boxes input folder", default="")
    arg_parser.add_argument("--output_folder_path_boxes", type=str, help="Path to output folder", default="")
    args = arg_parser.parse_args()
    # create output folder
    if not os.path.exists(args.output_folder_path_boxes):
        os.makedirs(args.output_folder_path_boxes)

    boxes_file_paths = sorted(glob.glob(args.input_folder_path_boxes + "/*.json"))
    # iterate over all files in input folder
    for file_path in boxes_file_paths:
        file_name = os.path.basename(file_path)
        data_json = json.load(open(file_path))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # iterate over all objects
            for object_id, label in frame_obj["objects"].items():
                cuboid = label["object_data"]["cuboid"]["val"]
                category = label["object_data"]["type"]
                if category in default_dimensions:
                    default_dimension = default_dimensions[category]
                    cuboid[7:10] = default_dimension
        # write json file
        with open(os.path.join(args.output_folder_path_boxes, file_name), "w") as f:
            json.dump(data_json, f)
