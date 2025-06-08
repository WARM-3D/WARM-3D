import os
import json
import hashlib

input_files_labels = "annotations/"
output_files_labels = "annotations_for_dataset/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    first_part = label_file_name.split(".")[0]
    parts = first_part.split("_")
    secs = parts[0]
    json_data["timestamp_secs"] = secs
    nsecs = parts[1]
    json_data["timestamp_nsecs"] = nsecs
    json_data["point_cloud_file_name"] = label_file_name.replace(".json", ".pcd")
    del json_data["camera_channel"]
    del json_data["sequence"]
    del json_data["image_file_name"]
    del json_data["time_of_day"]
    for label in json_data["labels"]:
        if "box3d_projected" in label:
            del label["box3d_projected"]
        label["id"] = label["category"].upper() + "_" + str(label["id"]).zfill(6)
    with open(os.path.join(output_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
