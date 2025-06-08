import os
import json
import hashlib

input_files_labels = "annotations_backup/"
output_files_labels = "annotations/"
input_files_point_clouds = "point_clouds/"
idx = 201
for label_file_name, point_cloud_file_name in zip(sorted(os.listdir(input_files_labels)),
                                                  sorted(os.listdir(input_files_point_clouds))):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    # first_part=point_cloud_file_name.split(".")[0]
    # parts = first_part.split("_")
    # secs = parts[0]
    # json_data["timestamp_secs"] = secs
    # nsecs = parts[1]
    # json_data["timestamp_nsecs"] = nsecs
    json_data["index"] = idx
    idx = idx + 1
    # del json_data["pointcloud_file_name"]
    json_data["point_cloud_file_name"] = point_cloud_file_name
    del json_data["camera_channel"]
    del json_data["sequence"]
    del json_data["image_file_name"]
    del json_data["time_of_day"]
    for label in json_data["labels"]:
        if "box3d_projected" in label:
            del label["box3d_projected"]
        label["id"] = label["category"].upper() + "_" + str(label["id"]).zfill(6)
    with open(os.path.join(output_files_labels, point_cloud_file_name.replace(".pcd", ".json")), 'w',
              encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
