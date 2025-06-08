import argparse
import glob
import json
import os

from tqdm import tqdm

from src.visualization.visualize_image_with_3d_boxes import set_track_history

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels",
        type=str,
        help="Input folder path to labels",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_labels",
        type=str,
        help="Output folder path to labels",
        default="",
    )
    args = arg_parser.parse_args()
    input_folder_paths_labels = args.input_folder_path_labels
    output_folder_paths_labels = args.output_folder_path_labels

    if not os.path.exists(output_folder_paths_labels):
        os.makedirs(output_folder_paths_labels)

    input_file_paths = sorted(glob.glob(args.input_folder_path_labels + "/*.json"))
    current_frame_idx = 0
    for input_file_path in tqdm(input_file_paths):
        labels_json = json.load(open(input_file_path, "r"))
        set_track_history(labels_json, input_file_paths, current_frame_idx, use_boxes_in_s110_base=False)
        # store track history
        file_name = os.path.basename(input_file_path)
        json.dump(labels_json, open(os.path.join(output_folder_paths_labels, file_name), "w"),
                  sort_keys=True)
        current_frame_idx += 1
