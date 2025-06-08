import os
import json
import hashlib

input_files_labels = "04_labels/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    del json_data["camera_channel"]
    del json_data["sequence"]
    for label in json_data["labels"]:
        label["id"] = hashlib.sha1(str(str(label["category"]) + str(label["id"])).encode('utf-8')).hexdigest()[:10]
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
