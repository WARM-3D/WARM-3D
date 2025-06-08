import os
import json

input_files_labels = "/home/walter/Downloads/a9_r0_dataset/r0_s2/04_labels/"
idx = 0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    for label in json_data["labels"]:
        x_coordinates = []
        y_coordinates = []
        x_coordinates.append(int(label["box3d_projected"]["bottom_left_front"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["bottom_left_front"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["bottom_left_back"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["bottom_left_back"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["bottom_right_back"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["bottom_right_back"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["bottom_right_front"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["bottom_right_front"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["top_left_front"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["top_left_front"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["top_left_back"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["top_left_back"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["top_right_back"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["top_right_back"][1] * IMAGE_HEIGHT))
        x_coordinates.append(int(label["box3d_projected"]["top_right_front"][0] * IMAGE_WIDTH))
        y_coordinates.append(int(label["box3d_projected"]["top_right_front"][1] * IMAGE_HEIGHT))
        label["box2d"] = [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
