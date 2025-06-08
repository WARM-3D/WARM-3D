import os
import json

input_files_images = "03_images/"
input_files_labels = "04_labels/"

for label_file_name, image_file_name in zip(sorted(os.listdir(input_files_labels)),
                                            sorted(os.listdir(input_files_images))):

    json_file = open(os.path.join(input_files_labels, label_file_name), )
    labels = json.load(json_file)
    file_name_new = labels["image_file_name"].split(".")[0]
    parts = file_name_new.split("_")
    if len(parts) == 12:
        # use only timestamp (sec + nsec) and sensor ID
        file_name_new = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + "_" + parts[5] + "_" + parts[
            6] + "_" + parts[7] + "_" + parts[8] + "_" + parts[9] + "_" + parts[10].zfill(9)
    else:
        # file name has 10 parts
        # use only timestamp (sec + nsec) and sensor ID
        direction = ""
        if parts[5] == "n":
            direction = "north"
        else:
            direction = "south"
        lens_type = ""
        if parts[7] == "far":
            lens_type = "50mm"
        else:
            lens_type = "16mm"
        file_name_new = parts[8] + "_" + parts[9].zfill(9) + "_" + parts[
            4] + "_camera_basler_" + direction + "_" + lens_type
    labels["image_file_name"] = file_name_new + ".jpg"
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(labels, writer, ensure_ascii=True, indent=4)
    os.rename(os.path.join(input_files_images, image_file_name),
              os.path.join(input_files_images, file_name_new + ".jpg"))
    os.rename(os.path.join(input_files_labels, label_file_name),
              os.path.join(input_files_labels, file_name_new + ".json"))
