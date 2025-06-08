import os
import json

input_files_labels = "04_labels/"
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    for label in json_data["labels"]:
        label["box3d_projected"]["bottom_left_front"] = [value * 2.0 for value in
                                                         label["box3d_projected"]["bottom_left_front"]]
        label["box3d_projected"]["bottom_left_back"] = [value * 2.0 for value in
                                                        label["box3d_projected"]["bottom_left_back"]]
        label["box3d_projected"]["bottom_right_back"] = [value * 2.0 for value in
                                                         label["box3d_projected"]["bottom_right_back"]]
        label["box3d_projected"]["bottom_right_front"] = [value * 2.0 for value in
                                                          label["box3d_projected"]["bottom_right_front"]]
        label["box3d_projected"]["top_left_front"] = [value * 2.0 for value in
                                                      label["box3d_projected"]["top_left_front"]]
        label["box3d_projected"]["top_left_back"] = [value * 2.0 for value in label["box3d_projected"]["top_left_back"]]
        label["box3d_projected"]["top_right_back"] = [value * 2.0 for value in
                                                      label["box3d_projected"]["top_right_back"]]
        label["box3d_projected"]["top_right_front"] = [value * 2.0 for value in
                                                       label["box3d_projected"]["top_right_front"]]
    with open(os.path.join(input_files_labels, label_file_name), 'w', encoding='utf-8') as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)
