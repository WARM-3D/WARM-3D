import argparse
import glob
import os
import json
from pathlib import Path

import numpy as np

import uuid
import hashlib
from src.utils.detection import Detection, save_to_openlabel
from src.utils.utils import providentia_id_to_class_name_mapping

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_folder_path_labels', type=str, help='Path to labels input folder', default='')
    argparser.add_argument('--output_folder_path_labels', type=str, help='Output folder Path', default='')
    argparser.add_argument('--image_width', type=int, help='Image width', default=1920)
    argparser.add_argument('--image_height', type=int, help='Image height', default=1200)

    args = argparser.parse_args()
    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels
    image_width = args.image_width
    image_height = args.image_height

    # create output folder
    if not os.path.exists(output_folder_path_labels):
        os.makedirs(output_folder_path_labels)

    # iterate over all files in input folder
    for file_path_label in sorted(glob.glob(input_folder_path_labels + '/*.json')):
        file_name_label = os.path.basename(file_path_label)
        # load json file
        data_json = json.load(open(file_path_label))
        detections = []
        # iterate over all frames
        for label_obj in data_json['object_list']:
            object_class_id = label_obj["object_class"]  # integer [0, 12]
            object_id = label_obj["object_ID"]

            sub_type_id = label_obj["type"]
            if object_id == 2 and sub_type_id == 1:
                category = "BICYCLE"
            elif object_id == 2 and sub_type_id == 3:
                category = "MOTORCYCLE"

            yaw = label_obj["heading"][0]
            category = providentia_id_to_class_name_mapping[object_class_id]
            location = np.array([label_obj['position'][0], label_obj['position'][1],
                                 label_obj['position'][2]])
            # make location [[x], [y], [z]]
            location = np.expand_dims(location, axis=1)
            dimensions = np.array([label_obj['extent'][0],
                                   label_obj['extent'][1],
                                   label_obj['extent'][2]])
            detection = Detection(location=location, dimensions=dimensions, yaw=yaw, category=category)
            detection.uuid = uuid.uuid5(uuid.NAMESPACE_OID, str(object_id))
            detection.sensor_id = "dtwin_s40_s50"
            detection.score = label_obj["classification_confidence"][object_class_id]
            detection.existence_probability = label_obj["existence_probability"]
            detection.yaw_rate = label_obj["yaw_rate"]
            detection.speed = label_obj["speed"]  # speed (float): Speed of the object in m/s
            detection.velocity = np.array(label_obj["velocity"])  # velocity vector (vx, vy, vz) in m/s

            detections.append(detection)
        # sort detections by id
        detections.sort(key=lambda x: x.id)
        save_to_openlabel(detections, file_name_label, Path(output_folder_path_labels))
