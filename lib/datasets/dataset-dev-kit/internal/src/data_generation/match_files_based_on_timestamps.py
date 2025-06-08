#!/usr/bin/env python
import glob
import os
import re
import shutil
import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# This module optimizes the synchronization error by duplicating frames


def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split("([-+]?[0-9]*\.?[0-9]*)", key)]
    l.sort(key=alphanum)
    return l


def match_timestamps(source_file_paths, target_file_paths):
    matched_target_file_paths = []
    source_idx = 0
    print("num frames (source): ", str(len(source_file_paths)))
    frames_to_skip = 1
    print("frames to skip: ", str(frames_to_skip))
    num_matches = 0
    while source_idx < len(source_file_paths):
        time_differences = []
        source_file_path = source_file_paths[source_idx]
        source_filename = os.path.basename(source_file_path)
        parts = source_filename.split(".")[0].split("_")
        if len(parts) > 0:
            source_seconds = int(parts[0])
            source_nano_seconds = int(parts[1])
            source_nano_seconds_full = source_seconds * 1000000000 + source_nano_seconds
        else:
            print("Could not extract seconds and nano seconds from source file name: ", str(source_filename))
            sys.exit()

        for target_idx, target_file_path in enumerate(target_file_paths):
            # extract file name from path
            target_filename = os.path.basename(target_file_path)
            parts = target_filename.split(".")[0].split("_")

            if len(parts) > 0:
                target_seconds_current = int(parts[0])
                target_nano_seconds_current = int(parts[1])
                target_nano_seconds_current_full = target_seconds_current * 1000000000 + target_nano_seconds_current
            else:
                print("Could not extract seconds and nano seconds from target file name: ", str(target_file_path))
                sys.exit()

            time_differences.append(abs(source_nano_seconds_full - target_nano_seconds_current_full))

        target_idx_smallest = np.array(time_differences).argmin()
        target_file_path = target_file_paths[target_idx_smallest]
        num_matches = num_matches + 1
        source_idx = source_idx + frames_to_skip
        # add target file path to list
        matched_target_file_paths.append(target_file_path)
    return matched_target_file_paths


def parse_arguments():
    parser = argparse.ArgumentParser(description="Match frames of source and target folder based on timestamps.")
    # parser.add_argument("--root-dir", type=str, default=None, help="Specify the root path of the dataset")
    parser.add_argument("--out-dir", type=str, default=None, help="Specify the save directory")
    parser.add_argument("--folder_path_images", type=str, default=None, help="Specify the save directory")
    parser.add_argument("--folder_path_point_clouds", type=str, default=None, help="Specify the save directory")
    args = parser.parse_args()
    return args


def store_files_with_duplicates(input_file_paths, output_folder_path_raw_data):
    for input_file_path in tqdm(input_file_paths):
        input_file_name = os.path.basename(input_file_path)
        # extract sensor id from input file name
        # remove extension from input file name
        input_file_name_no_extension = os.path.splitext(input_file_name)[0]
        sensor_id = "_".join(input_file_name_no_extension.split("_")[2:])
        exists = os.path.exists(os.path.join(output_folder_path_raw_data, input_file_name))
        if exists:
            while os.path.exists(os.path.join(output_folder_path_raw_data, input_file_name)):
                print(f"File {input_file_name} already exists in {output_folder_path_raw_data}")
                # split file to extract timestamp in seconds and nano seconds
                parts = input_file_name.split(".")[0].split("_")
                if len(parts) >= 2:
                    seconds = int(parts[0])
                    nano_seconds = int(parts[1])
                nano_seconds = nano_seconds + 1
                input_file_name = (
                        str(seconds)
                        + "_"
                        + str(nano_seconds).zfill(9)
                        + "_"
                        + str(sensor_id)
                        + "."
                        + input_file_name.split(".")[1]
                )
                print(f"New file name: {input_file_name}")
            output_file_path_raw_data = os.path.join(output_folder_path_raw_data, input_file_name)
        else:
            output_file_path_raw_data = os.path.join(output_folder_path_raw_data, input_file_name)

        # copy image/point cloud file to output folder
        shutil.copy2(input_file_path, output_file_path_raw_data)


if __name__ == "__main__":
    args = parse_arguments()
    image_file_paths = sorted(glob.glob(args.folder_path_images + "/*"))
    point_cloud_file_paths = sorted(glob.glob(args.folder_path_point_clouds + "/*"))
    matched_target_file_paths = match_timestamps(point_cloud_file_paths, image_file_paths)
    sensor_id = "_".join(matched_target_file_paths[0].split("/")[-1].split(".")[0].split("_")[2:])
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "point_clouds"), exist_ok=True)
    store_files_with_duplicates(matched_target_file_paths, os.path.join(args.out_dir))
