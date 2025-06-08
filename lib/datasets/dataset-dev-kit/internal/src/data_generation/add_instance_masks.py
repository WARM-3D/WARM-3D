import argparse
import glob
import json
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_labels", type=str, help="Path to labels input folder", default="")
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    # add sensor ID
    arg_parser.add_argument("--sensor_id", type=str, help="Sensor ID", default="")
    args = arg_parser.parse_args()
    sensor_id = args.sensor_id
    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    label_file_paths = sorted(glob.glob(args.input_folder_path_labels + "/*.json"))
    # iterate over all files in input folder
    for file_path_label in label_file_paths:
        file_name_label = os.path.basename(file_path_label)
        # load json file
        data_json = json.load(open(file_path_label))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # iterate over all objects
            for object_id, label in frame_obj["objects"].items():
                a_bbox = label["object_data"]["a_bbox"]
                del label["object_data"]["a_bbox"]
                i_bbox = label["object_data"]["i_bbox"]
                del label["object_data"]["i_bbox"]

                # Step 1: store a_bbox and i_bbox under bbox
                label["object_data"]["bbox"] = [
                    {
                        "name": "full_bbox",
                        "val": np.array(a_bbox[0]["val"]).astype(int).tolist(),
                        "attributes": {
                            "text": [
                                {
                                    "name": "sensor_id",
                                    "val": sensor_id
                                }
                            ]
                        }
                    },
                    {
                        "name": "visible_bbox",
                        "val": np.array(i_bbox[0]["val"]).astype(int).tolist(),
                        "attributes": {
                            "text": [
                                {
                                    "name": "sensor_id",
                                    "val": sensor_id
                                }
                            ]
                        }
                    },
                ]

                amodal_mask = label["object_data"]["amodal_mask"]
                del label["object_data"]["amodal_mask"]
                imodal_mask = label["object_data"]["imodal_mask"]
                del label["object_data"]["imodal_mask"]

                # Step 2: store amodal mask and imodal mask under poly2d
                label["object_data"]["poly2d"] = [
                    {
                        "name": "full_mask",
                        "val": np.array(amodal_mask["attributes"]["mask"]).astype(int).tolist(),
                        "attributes": {
                            "text": [
                                {
                                    "name": "sensor_id",
                                    "val": sensor_id
                                }
                            ]
                        }
                    },
                    {
                        "name": "visible_mask",
                        "val": np.array(imodal_mask["attributes"]["mask"]).astype(int).tolist(),
                        "attributes": {
                            "text": [
                                {
                                    "name": "sensor_id",
                                    "val": sensor_id
                                }
                            ]
                        }
                    },
                ]

        # write json file
        with open(os.path.join(args.output_folder_path_labels, file_name_label), "w") as f:
            json.dump(data_json, f)
