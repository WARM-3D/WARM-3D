import argparse
import glob
import os
import json
import sys

lidar_north = {
    "d7d28da7-00a7-49cb-9feb-f6dda0f08184": {
        "start": "1651673050_054497235_s110_lidar_ouster_north.json",
        "end": "1651673059_955195369_s110_lidar_ouster_north.json",
        "old_category": "CAR",
        "new_category": "OTHER",
        "new_uuid": "4e216095-d484-494b-aafa-782209853df0",
        "has_flashing_lights": False,
    },
    "a91aeb87-f93b-47c6-803e-67a103967ba3": {
        "start": "1651673060_061669345_s110_lidar_ouster_north.json",
        "end": "1651673069_959030382_s110_lidar_ouster_north.json",
        "old_category": "CAR",
        "new_category": "OTHER",
        "new_uuid": "4e216095-d484-494b-aafa-782209853df0",
        "has_flashing_lights": False,
    },
    "4e216095-d484-494b-aafa-782209853df0": {
        "start": "1651673070_061175935_s110_lidar_ouster_north.json",
        "end": "1651673079_159189495_s110_lidar_ouster_north.json",
        "old_category": "CAR",
        "new_category": "OTHER",
        "new_uuid": "4e216095-d484-494b-aafa-782209853df0",
        "has_flashing_lights": False,
    },
}


def get_timestamp_by_filename(file_name):
    parts = file_name.split("_")
    timestamp_seconds = int(parts[0])
    timestamp_nano_seconds_remaining = int(parts[1])
    timestamp_nano_seconds = timestamp_seconds * 1e9 + timestamp_nano_seconds_remaining
    return timestamp_nano_seconds


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="dateset category rename")
    argparser.add_argument(
        "--input_folder_path_labels",
        help="Input folder path to lidar labels in OpenLABEL format.",
    )
    argparser.add_argument(
        "--output_folder_path_labels",
        default="output",
        help="Output folder path.",
    )

    args = argparser.parse_args()

    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels

    # create output folder if not exists
    if not os.path.exists(output_folder_path_labels):
        os.makedirs(output_folder_path_labels)

    label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))

    for label_file_path in label_file_paths:
        file_name = os.path.basename(label_file_path)
        timestamp_nano_seconds = get_timestamp_by_filename(file_name)
        valid_labels = {}
        labels = json.load(open(label_file_path))
        if "openlabel" in labels:
            num_detections_fused = 0
            objects_added = {}
            for frame_idx, frame_obj in labels["openlabel"]["frames"].items():
                for box_uuid, box in frame_obj["objects"].items():
                    category = box["object_data"]["type"]
                    if box_uuid in lidar_north and category == lidar_north[box_uuid]["old_category"]:
                        start_file_name = lidar_north[box_uuid]["start"]
                        timestamp_nano_seconds_start = get_timestamp_by_filename(start_file_name)
                        end_file_name = lidar_north[box_uuid]["end"]
                        timestamp_nano_seconds_end = get_timestamp_by_filename(end_file_name)
                        if timestamp_nano_seconds_start <= timestamp_nano_seconds <= timestamp_nano_seconds_end:
                            # rename object category

                            box["object_data"]["type"] = lidar_north[box_uuid]["new_category"]
                            # rename name
                            box["object_data"]["name"] = (
                                    lidar_north[box_uuid]["new_category"]
                                    + "_"
                                    + lidar_north[box_uuid]["new_uuid"].split("-")[0]
                            )
                            # update attributes
                            if lidar_north[box_uuid]["new_category"] == "OTHER":
                                # add flashing_emergency_lights attribute
                                if not "bool" in box["object_data"]["cuboid"]["attributes"]:
                                    box["object_data"]["cuboid"]["attributes"]["bool"] = []
                                box["object_data"]["cuboid"]["attributes"]["bool"].append(
                                    {"name": "has_flashing_lights", "val": lidar_north[box_uuid]["has_flashing_lights"]}
                                )
                            # update key with new uuid
                            frame_obj["objects"][lidar_north[box_uuid]["new_uuid"]] = box
                            if box_uuid != lidar_north[box_uuid]["new_uuid"]:
                                del frame_obj["objects"][box_uuid]
                            break
        else:
            print("No openlabel in {}".format(label_file_path))
            sys.exit()
        # write to file
        output_file_path = os.path.join(output_folder_path_labels, file_name)
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(labels, outfile, indent=4)
