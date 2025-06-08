import argparse
import glob
import json
import os

if __name__ == "__main__":
    # add arg parser
    a = 1
    # add 1 to a
    #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels",
        type=str,
        help="Path to train labels",
        default="",
    )
    args = arg_parser.parse_args()
    input_folder_path = args.input_folder_path_labels
    input_label_files = sorted(glob.glob(input_folder_path + "/*.json"))
    image_file_names_south2 = []
    for input_label_file_path in input_label_files:
        json_file = open(input_label_file_path)
        json_data = json.load(json_file)
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            image_file_names_south2.append(frame_obj["frame_properties"]["image_file_names"][0])
    # write all file names to file
    with open("/home/walter/Downloads/file_names_s04_south1_correct.txt", "w") as file_writer:
        for image_file_name_south2 in image_file_names_south2:
            file_writer.write(image_file_name_south2 + "\n")
