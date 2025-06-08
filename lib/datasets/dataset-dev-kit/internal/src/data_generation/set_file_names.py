import os
import json

# input_files_point_clouds = "point_clouds/"
input_files_images_anonym = "03_images_anonymized/"
for image_file_name_anonym in sorted(os.listdir(input_files_images_anonym)):
    first_part = image_file_name_anonym.split(".")[0]
    parts = first_part.split("_")
    print(parts)
    direction = ""
    if parts[5] == "n":
        direction = "north"
    else:
        direction = "south"
    type = ""
    if parts[7] == "near":
        type = "16mm"
    else:
        type = "50mm"
    image_file_name_new = parts[8] + "_" + parts[9].zfill(9) + "_" + parts[
        4] + "_camera_basler_" + direction + "_" + type + ".jpg"
    os.rename(os.path.join(input_files_images_anonym, image_file_name_anonym),
              os.path.join(input_files_images_anonym, image_file_name_new))
