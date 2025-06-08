import os
import json

input_files_labels = "04_labels/"
idx = 0
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    json_data["index"] = idx
    idx = idx + 1
    json_data["timestamp_secs"] = int(json_data["timestamp_secs"])
    json_data["timestamp_nsecs"] = int(json_data["timestamp_nsecs"].lstrip("0"))
    del json_data["camera_channel"]
    del json_data["sequence"]
    del json_data["time_of_day"]
    for label in json_data["labels"]:
        label["id"] = label["category"].upper() + "_" + str(label["id"]).zfill(6)

    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
