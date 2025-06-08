import os
import json
import hashlib

input_files_labels = "04_labels/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    part_one = label_file_name.split(".")[0]
    parts = part_one.split("_")
    nsecs = int(parts[1].lstrip("0"))
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    json_data["timestamp_nsecs"] = nsecs
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
