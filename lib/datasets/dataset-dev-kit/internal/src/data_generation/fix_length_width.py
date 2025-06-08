import os
import json
import hashlib

input_files_labels = "annotations_for_dataset/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    json_data["point_cloud_file_name"] = label_file_name.replace(".json", ".pcd")
    for label in json_data["labels"]:
        label["id"] = label["category"].upper() + "_" + str(label["id"]).zfill(6)
        label["category"] = label["category"].upper()
        tmp = label["box3d"]["dimension"]["width"]
        label["box3d"]["dimension"]["width"] = label["box3d"]["dimension"]["length"]
        label["box3d"]["dimension"]["length"] = tmp
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
