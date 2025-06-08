import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_labels",
        type=str,
        help="Input directory path",
        default="",
    )
    parser.add_argument(
        "--output_folder_path_labels",
        type=str,
        help="Output directory path",
        default="",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    for file_name in sorted(os.listdir(args.input_folder_path_labels)):
        label_data = json.load(open(os.path.join(args.input_folder_path_labels, file_name)))
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            frame_obj["objects"] = {}
        with open(os.path.join(args.output_folder_path_labels, file_name), "w") as f:
            json.dump(label_data, f, indent=4)
