import argparse
import glob
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import json
import sys
import numpy as np

# random.seed(42) # seed for debug


IMAGE_HZ = 10
LIDAR_HZ = 10


def match_timestamps(input_source_file_paths, input_target_file_paths, matched_before):
    matched_target_file_paths = []
    file_idx = 0
    while file_idx < len(input_source_file_paths):
        time_differences = []
        # extract file name from file path
        source_file_name = os.path.basename(input_source_file_paths[file_idx])
        parts = source_file_name.split(".")[0].split("_")
        if len(parts) > 2:
            source_seconds = int(parts[0])
            source_nano_seconds = int(parts[1])
            source_nano_seconds_full = source_seconds * 1000000000 + source_nano_seconds
        else:
            print("Could not extract seconds and nano seconds from source file name: ", str(source_file_name))
            sys.exit()

        for target_idx, target_file_path in enumerate(input_target_file_paths):
            # extract file name from file path
            target_file_name = os.path.basename(target_file_path)
            parts = target_file_name.split(".")[0].split("_")
            if len(parts) > 2:
                target_seconds_current = int(parts[0])
                target_nano_seconds_current = int(parts[1])
                target_nano_seconds_current_full = target_seconds_current * 1000000000 + target_nano_seconds_current
            else:
                print("Could not extract seconds and nano seconds from target file name: ", str(target_file_name))
                sys.exit()

            time_differences.append(abs(source_nano_seconds_full - target_nano_seconds_current_full))

        target_idx_smallest = np.array(time_differences).argmin()
        target_file_path = input_target_file_paths[target_idx_smallest]
        # check if frame was matched before in train set
        found_nearest_neighbor = False
        idx = 0
        while not found_nearest_neighbor:
            if target_file_path not in matched_before:
                matched_target_file_paths.append(target_file_path)
                found_nearest_neighbor = True
            else:
                print("Index: ", idx)
                print("Frame already matched before: ", source_file_name, target_file_path)
                # remove this frame from time_differences list
                time_differences.pop(target_idx_smallest)
                # use next frame as nearest neighbor
                target_idx_smallest = np.array(time_differences).argmin()
                target_file_path = input_target_file_paths[target_idx_smallest]
            idx += 1

        file_idx = file_idx + 1
    return input_source_file_paths, matched_target_file_paths


def process_set(input_file_paths_split, split_type, modal, out_dir, revert, replace):
    split_type_set_folder_path = os.path.join(out_dir, split_type, modal)
    output_folder_path_labels = os.path.join(out_dir, split_type, "labels")
    os.makedirs(split_type_set_folder_path, exist_ok=True)
    os.makedirs(output_folder_path_labels, exist_ok=True)

    for input_file_path in tqdm(input_file_paths_split):
        output_folder_path_sensor = split_type_set_folder_path
        output_folder_path_labels_new = output_folder_path_labels
        input_label_file_path = input_file_path.replace(modal, "labels").replace(
            "png" if modal == "images" else "pcd", "json"
        )
        if not os.path.exists(input_label_file_path):
            print(f"Label file {input_label_file_path} does not exist")
            continue
        # copy or move file
        input_file_name = os.path.basename(input_file_path)
        output_folder_path_sensor = os.path.join(output_folder_path_sensor, input_file_name)
        output_folder_path_labels_new = os.path.join(output_folder_path_labels_new, input_file_name)

        dest = replace(input_file_path, output_folder_path_sensor)
        revert.update({input_file_path: dest})

        dest = replace(
            input_label_file_path, output_folder_path_labels_new.replace("png" if modal == "images" else "pcd", "json")
        )
        revert.update({input_label_file_path: dest})


def create_data_split(
        operation,
        version,
        in_place,
        root_path,
        out_dir,
        split=None
):
    # init variables
    is_image = False
    is_pcd = False

    image_dirs = []
    point_cloud_dirs = []
    splits = []
    revert = dict()
    revert_dir = []

    if in_place:
        replace = shutil.move
    else:
        replace = shutil.copy2

    if operation == "split":

        dirs = sorted(glob.glob(os.path.join(root_path, "*")))
        if version == "full":
            is_image = True
            is_pcd = True
        elif version == "image":
            is_image = True
        elif version == "point_cloud":
            is_pcd = True

        # check split compatible with version
        for dir in dirs:
            if dir in ["train", "val"]:
                continue
            image_path = Path(os.path.join(dir, "images"))
            if is_image and Path.exists(image_path):
                image_dirs.append(image_path)
            pcd_path = Path(os.path.join(dir, "point_clouds"))
            if is_pcd and Path.exists(pcd_path):
                point_cloud_dirs.append(pcd_path)

        # Check splits global or individually
        if split is None:
            # default split globally
            split = [0.85, 0.15]
            if version == "full":
                splits.extend([split] * max(len(image_dirs), len(point_cloud_dirs)))
            elif version == "image":
                splits.extend([split] * len(image_dirs))
            elif version == "point_cloud":
                splits.extend([split] * len(point_cloud_dirs))

        elif len(split) > 2:
            # individual split
            splits = is_split_valid(split)
            # do sanity check for given version
            if version == "full":
                assert len(splits) == max(
                    len(image_dirs), len(point_cloud_dirs)
                ), f"Number of splits {len(splits)} is not equal the number of possible directories {max(len(image_dirs), len(point_cloud_dirs))}"
            elif version == "image":
                assert len(splits) == len(
                    image_dirs
                ), f"Number of splits {len(splits)} is not equal the number of possible directories {len(image_dirs)}"
            elif version == "point_cloud":
                assert len(splits) == len(
                    point_cloud_dirs
                ), f"Number of splits {len(splits)} is not equal the number of possible directories {len(point_cloud_dirs)}"
        else:
            # global split
            split = [i * 0.01 for i in split]
            if version == "full":
                splits.extend([split] * max(len(image_dirs), len(point_cloud_dirs)))
            elif version == "image":
                splits.extend([split] * len(image_dirs))
            elif version == "point_cloud":
                splits.extend([split] * len(point_cloud_dirs))

        Path(out_dir + "/train").mkdir(
            mode=0o777,
            parents=True,
            exist_ok=True,
        )
        Path(out_dir + "/val").mkdir(mode=0o777, parents=True, exist_ok=True)
        image_dirs.sort()
        point_cloud_dirs.sort()
        idx = 0
        for image_dir, point_cloud_dir in zip(image_dirs, point_cloud_dirs):
            split = splits[idx]
            idx += 1
            image_sub_dirs = sorted(glob.glob(os.path.join(image_dir, "*")))
            point_cloud_sub_dirs = sorted(glob.glob(os.path.join(point_cloud_dir, "*")))
            print(f"Processing images in {image_dir}")

            sensor_sub_dirs = image_sub_dirs + point_cloud_sub_dirs

            sensor_reference_dir = sensor_sub_dirs[0]
            images_len = len(glob.glob(os.path.join(sensor_reference_dir, "*")))
            train_len, val_len = calculate_split_len(images_len, split)
            sensor_reference_file_paths = sorted(glob.glob(os.path.join(sensor_reference_dir, "*")))
            file_paths_filtered = []
            for sensor_sub_dir in sensor_sub_dirs:
                sensor_sub_dir_file_paths = sorted(glob.glob(os.path.join(sensor_sub_dir, "*")))
                file_paths_filtered.append(sensor_sub_dir_file_paths)

            # shuffle files
            matched_file_paths = list(
                zip(
                    file_paths_filtered[0],
                    file_paths_filtered[1],
                    file_paths_filtered[2],
                    file_paths_filtered[3],
                )
            )
            random.shuffle(matched_file_paths)
            matched_file_paths_shuffled_all = [None for i in range(len(sensor_sub_dirs))]
            (
                matched_file_paths_shuffled_all[0],
                matched_file_paths_shuffled_all[1],
                matched_file_paths_shuffled_all[2],
                matched_file_paths_shuffled_all[3],
            ) = zip(*matched_file_paths)

            print("using x frames for training: ", str(len(matched_file_paths_shuffled_all[0][:train_len])))
            process_set(
                input_file_paths_split=matched_file_paths_shuffled_all[0][:train_len],
                split_type="train",
                modal="images",
                out_dir=out_dir,
                revert=revert,
                replace=replace,
            )
            print(
                "using x frames for val: ",
                str(len(matched_file_paths_shuffled_all[0][train_len: train_len + val_len])),
            )
            process_set(
                input_file_paths_split=matched_file_paths_shuffled_all[0][train_len:],
                split_type="val",
                modal="images",
                out_dir=out_dir,
                revert=revert,
                replace=replace,
            )
            # iterate over all other sensor dirs
            for matched_file_paths_shuffled in matched_file_paths_shuffled_all[1:]:
                if "images" in matched_file_paths_shuffled[0]:
                    modal = "images"
                elif "point_clouds" in matched_file_paths_shuffled[0]:
                    modal = "point_clouds"
                # move files to outdir
                process_set(
                    input_file_paths_split=matched_file_paths_shuffled[:train_len],
                    split_type="train",
                    modal=modal,
                    out_dir=out_dir,
                    revert=revert,
                    replace=replace,
                )
                process_set(
                    input_file_paths_split=matched_file_paths_shuffled[train_len: train_len + val_len],
                    split_type="val",
                    modal=modal,
                    out_dir=out_dir,
                    revert=revert,
                    replace=replace,
                )

        revert.update(
            {"target": [os.path.join(out_dir, "train"), os.path.join(out_dir, "val")]}
        )
        revert.update({"tree": revert_dir})
        if in_place:
            with open(out_dir + "revert.json", "w") as f:
                json.dump(revert, f)

    elif operation == "revert":
        assert os.path.isfile(root_path)
        with open(root_path, "r") as f:
            data = json.load(f)

        for dir in data["tree"]:
            os.makedirs(dir, exist_ok=True)
        for key in data:
            if key != "tree" and key != "target":
                shutil.move(data[key], key)
        os.remove(root_path)


def calculate_split_len(set_len, split):
    train_len = int(round(set_len * split[0]))
    print("train_len", train_len)
    val_len = int(round(set_len * split[1]))
    print("val_len", val_len)
    return train_len, val_len


def is_split_valid(split):
    splits = []
    current_split = []
    for val in split:
        if val == 0:
            current_split.append(0.0)
        else:
            current_split.append(val / 100.0)
        if len(current_split) == 2:
            assert np.isclose(np.sum(current_split), 1.0), f'This split {current_split} is not valid'
            splits.append(current_split)
            current_split = []
    return splits


def parse_arguments():
    parser = argparse.ArgumentParser(description="creating data split for A9 dataset")
    parser.add_argument("operation", metavar="split", type=str, choices=["split", "revert"])
    parser.add_argument(
        "--version",
        default="full",
        type=str,
        choices=["point_cloud", "image", "full"],
        help="Specify the version ['point_cloud', 'image', 'full']",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Specify whether to create a copy of the data or move them. Set --inplace to move the data. If --inplace is not set, the data will be copied.",
    )
    parser.add_argument("--root-dir", type=str, default=None, help="Specify the root path of the dataset")
    parser.add_argument("--out-dir", type=str, default=None, help="Specify the save directory")
    parser.add_argument("--split", nargs="+", type=float,
                        help="Specify split for each set '--split 90 10' means r01_s01 90% train 10% val")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    create_data_split(
        operation=args.operation,
        version=args.version,
        in_place=args.inplace,
        root_path=args.root_dir,
        out_dir=args.out_dir,
        split=args.split,
    )
