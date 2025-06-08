import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from plot_utils import PlotUtils

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels_train",
        type=str,
        help="Path to train labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_val",
        type=str,
        help="Path to val labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sampled",
        type=str,
        help="Path to test sampled labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sequence",
        type=str,
        help="Path to test sequence labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r01_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_statistic_plots",
        type=str,
        help="Output folder path to statistics",
        default="",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    input_folder_paths_all = []
    if args.input_folder_path_labels_train:
        input_folder_paths_all.append(args.input_folder_path_labels_train)
    if args.input_folder_path_labels_val:
        input_folder_paths_all.append(args.input_folder_path_labels_val)
    if args.input_folder_path_labels_test_sampled:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sampled)
    if args.input_folder_path_labels_test_sequence:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sequence)

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)

    rotations = []
    for input_files_labels in input_folder_paths_all:
        for label_file_name in sorted(os.listdir(input_files_labels)):
            json_file = open(
                os.path.join(input_files_labels, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_data = object_json["object_data"]
                    quaternion = np.asarray(object_data["cuboid"]["val"][3:7])
                    roll, pitch, yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)
                    rotations.append(np.degrees(yaw))

    # calculate average rotation
    avg_rotation = np.mean(rotations) + 180
    print("Average rotation: ", avg_rotation)
    plot = PlotUtils()
    ############################
    # 1. Histogram
    ############################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plt.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.11)
    print("Min", np.min(rotations), "Max", np.max(rotations))
    rotations = np.asarray(rotations) + 180
    bins_original = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    y_values, bins = np.histogram(rotations, bins=12)
    # y_max = np.max(y_values)
    print("Min", np.min(rotations), "Max", np.max(rotations))
    # generate y-axis color_bar string labels from 0k to 11k
    color_bar_labels = [str(i) + "k" for i in range(0, 7)]

    plot.plot_histogram(
        ax,
        rotations,
        num_bins=12,
        range_list=[0, 360],
        x_label=r"Rotation in degrees",
        y_label=r"\# 3D box labels",
        bin_labels=np.linspace(0, 360, 7),
        use_log_scale=False,
        y_max=None,
        step_size=6,
        color_bar_labels=color_bar_labels,
    )
    # show average
    plt.axvline(x=avg_rotation, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(avg_rotation + 10, 5000, str(round(avg_rotation)), color="r", fontsize=14)

    # show y values
    idx = 0
    for y_value, bin_current in zip(y_values, bins_original):
        x_pos = None
        if idx == 0 or idx == len(y_values) - 1:
            x_pos = bin_current
        elif idx + 1 < len(y_values) and y_value < y_values[idx + 1] and y_value > y_values[idx - 1]:
            x_pos = bin_current - 4
        elif idx + 1 < len(y_values) and y_value >= y_values[idx + 1] and y_value < y_values[idx - 1]:
            x_pos = bin_current + 4
        else:
            x_pos = bin_current + 3

        ax.text(x_pos, y_value + 100, str(y_value), color="black", fontweight="bold", fontsize=10)
        idx += 1

    # change the fontsize of minor ticks label
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "histogram_avg_rotation.pdf"))
