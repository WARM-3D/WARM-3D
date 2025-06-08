import os
import json

input_files_labels = "/home/walter/Downloads/a9_dataset_r0_dataset/r0_s2/04_labels/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    label_file_name_new = label_file_name.replace("south", "north")
    json_data["image_file_name"] = label_file_name_new
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
    os.rename(os.path.join(input_files_labels, label_file_name), os.path.join(input_files_labels, label_file_name_new))
