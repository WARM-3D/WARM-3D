import os
import json

input_files_labels = "labels/sequence/"

for label_file_name in sorted(os.listdir(input_files_labels)):
    data = open(os.path.join(input_files_labels, label_file_name), )
    labels = json.load(data)
    # labels["image_file_name"] = label_file_name.split(".")[0] + ".jpg"
    parts = label_file_name.split(".")[0].split("_")
    labels["timestamp_secs"] = str(labels["timestamp_secs"])
    labels["timestamp_nsecs"] = parts[9].zfill(9)
    labels_json = json.dumps(labels, indent=4)
    with open(os.path.join(input_files_labels, label_file_name), "w") as json_writer:
        json_writer.write(labels_json)
