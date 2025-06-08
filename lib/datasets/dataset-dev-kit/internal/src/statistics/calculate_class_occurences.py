import argparse
import copy
import glob
import os
import json
from math import log10

import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import ListedColormap
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from internal.src.statistics.plot_utils import PlotUtils
from src.utils.vis_utils import VisualizationUtils

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
        help="Path to r02_s01 south lidar labels",
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
        help="Path to output folder",
        default="",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    # create output folder
    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)
    font_size = 9

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

    classes_list = ["CAR", "TRUCK", "TRAILER", "VAN", "MOTORCYCLE", "BUS", "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE",
                    "OTHER"]
    # iterate classes_list and create a dictionary with empty lists for each class
    classes = {}
    for class_name in classes_list:
        classes[class_name] = []

    class_colors = PlotUtils.get_class_colors(alpha=0.5)

    # automatically find out what object classes are present in the dataset
    classes_valid_set = set()
    valid_ids = set()
    for input_folder_path in input_folder_paths_all:
        label_file_paths = sorted(glob.glob(input_folder_path + "/*.json"))
        for label_file_path in label_file_paths:
            labels_json = json.load(open(label_file_path, "r"))
            for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
                for uuid, box in frame_obj["objects"].items():
                    object_class = box["object_data"]["type"]
                    classes_valid_set.add(object_class)
                    valid_ids.add(classes_list.index(object_class))

    # remove not valid classes from classes
    classes_valid_list = list(classes_valid_set)
    class_coler_ids_to_delete = []
    for class_name in list(classes):
        if class_name not in classes_valid_list:
            del classes[class_name]
            # delete not valid color from class_colors
            class_coler_ids_to_delete.append(classes_list.index(class_name))

    class_colors = np.delete(class_colors, class_coler_ids_to_delete, axis=0)

    distance_bins = {
        "0-10": [],
        "10-20": [],
        "20-30": [],
        "30-40": [],
        "40-50": [],
        "50-60": [],
        # "60-70": [],
        # "70-80": [],
    }
    difficulty_levels = {
        "EASY": 0,
        "MODERATE": 0,
        "HARD": 0,
    }
    # TODO: temp comment EMERGENCY_VEHICLE and OTHER because there are no occurences in the dataset
    classes_and_distances = {
        "CAR": copy.deepcopy(distance_bins),
        "TRUCK": copy.deepcopy(distance_bins),
        "TRAILER": copy.deepcopy(distance_bins),
        "VAN": copy.deepcopy(distance_bins),
        "MOTORCYCLE": copy.deepcopy(distance_bins),
        "BUS": copy.deepcopy(distance_bins),
        "PEDESTRIAN": copy.deepcopy(distance_bins),
        "BICYCLE": copy.deepcopy(distance_bins),
        "EMERGENCY_VEHICLE": copy.deepcopy(distance_bins),
        # "OTHER": copy.deepcopy(distance_bins),
    }
    # num_objects_per_num_points = np.zeros(18000)
    histogram = {
        "1-200": 0,
        "200-400": 0,
        "400-600": 0,
        "600-800": 0,
        "800-1000": 0,
        "1000-1200": 0,
        "1200-1400": 0,
        "1400-1600": 0,
        "1600-1800": 0,
        "1800-18000": 0
    }
    total_num_points = 0
    alpha = 0.5

    num_points_per_class = {}

    for input_folder_path_labels in input_folder_paths_all:
        for label_file_name in sorted(os.listdir(input_folder_path_labels)):
            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_data = object_json["object_data"]
                    cuboid = object_data["cuboid"]["val"]
                    location = cuboid[:3]
                    distance = np.linalg.norm(location)

                    occlusion_attribute = VisualizationUtils.get_attribute_by_name(
                        object_data["cuboid"]["attributes"]["text"], "occlusion_level"
                    )

                    num_points_attribute = VisualizationUtils.get_attribute_by_name(
                        object_data["cuboid"]["attributes"]["num"], "num_points"
                    )
                    number_points = 0 if num_points_attribute["val"] == -1 else num_points_attribute["val"]
                    if object_data["type"] in classes.keys():
                        classes[object_data["type"]].append(number_points)
                    else:
                        classes[object_data["type"]] = [number_points]

                    # num_objects_per_num_points[int(round(number_points))] += 1
                    # check into which bin the number of points falls of the histogram
                    for bin_name, num_items in histogram.items():
                        bin_range = bin_name.split("-")
                        if number_points >= int(bin_range[0]) and number_points < int(bin_range[1]):
                            histogram[bin_name] += 1
                            break
                    total_num_points += number_points

                    if occlusion_attribute is not None and occlusion_attribute["val"] == "MOSTLY_OCCLUDED":
                        difficulty_levels["HARD"] += 1
                    elif occlusion_attribute is not None and occlusion_attribute["val"] == "PARTIALLY_OCCLUDED":
                        difficulty_levels["MODERATE"] += 1
                    elif (distance > 0 and distance < 40) or number_points >= 50:
                        difficulty_levels["EASY"] += 1
                    elif (distance >= 40 and distance < 50) or (number_points >= 20 and number_points < 50):
                        difficulty_levels["MODERATE"] += 1
                    elif (distance >= 50 and distance < 64) or (number_points < 20 and number_points >= 5):
                        difficulty_levels["HARD"] += 1
                    else:
                        pass

                    # for bins in classes_and_distances[object_data["type"]].items():
                    # check in what bin the distance falls
                    for distance_bin, distance_bin_range in distance_bins.items():
                        distance_bin_range = distance_bin.split("-")
                        min_limit = int(distance_bin_range[0])
                        max_limit = int(distance_bin_range[1])
                        if max_limit == 60:
                            max_limit = 120
                        if distance >= min_limit and distance < max_limit:
                            classes_and_distances[object_data["type"]][distance_bin].append(number_points)
                            break

    # calculate statistics for 3 difficulty levels
    # print difficulty levels
    print("Difficulty levels")
    for difficulty_level, num_objects in difficulty_levels.items():
        print(f"{difficulty_level}: {num_objects}")

    tab = PrettyTable(
        ["Class", "Occurrences", "Min. points", "Average num points", "Max. points", "Standard deviation"]
    )
    # sort classes dict

    for object_class_obj, num_points in classes.items():
        tab.add_row(
            [
                object_class_obj,
                len(num_points),
                np.min(num_points),
                np.mean(num_points),
                np.max(num_points),
                np.std(num_points),
            ]
        )
    total_3d_box_labels = sum([len(num_points) for num_points in classes.values()])
    print("Total 3D box labels: ", total_3d_box_labels)
    # count total 3D box labels having more than 0 points
    total_3d_box_labels_not_empty = 0
    for num_points in classes.values():
        for num_points_obj in num_points:
            if num_points_obj > 0:
                total_3d_box_labels_not_empty += 1

    print("Total 3D box labels not empty: ", total_3d_box_labels_not_empty)
    tab.add_row(
        [
            "TOTAL",
            total_3d_box_labels,
            np.min([np.min(num_points) for num_points in classes.values()]),
            np.mean([np.mean(num_points) for num_points in classes.values()]),
            np.max([np.max(num_points) for num_points in classes.values()]),
            np.std([np.std(num_points) for num_points in classes.values()]),
        ]
    )
    print(tab)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    #########################################################
    # 1. Bar chart of number of labels for each class
    #########################################################
    # fig, ax = plt.subplots(figsize=(5, 4.0))
    # plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.27)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.41)
    class_names = classes.keys()
    # replace EMERGENCY_VEHICLE with EMERGENCY_VEH in class names
    class_names = [class_name.replace("EMERGENCY_VEHICLE", "EMERGENCY_VEH") for class_name in class_names]
    occurrences = [len(class_num_points) for class_num_points in classes.values()]
    average_class_occurrences = np.mean(occurrences)
    print("Average number of labels per class: ", average_class_occurrences)
    plt.bar(
        class_names,
        occurrences,
        color=class_colors,
        edgecolor="black",
        zorder=3,
    )
    # occurrences_night = [2833, 100, 100, 200, 0, 100, 600, 0]
    # plt.bar(
    #     class_names,
    #     occurrences_night,
    #     color=class_colors,
    #     edgecolor="black",
    #     zorder=3,
    #     hatch="///",
    # )
    # add legend
    # legend_elements = []
    # legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Night", hatch="///"))
    # ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=font_size)
    # add text labels
    for i, num_labels in enumerate(occurrences):
        ax.text(i - 0.25, num_labels + num_labels / 2, str(num_labels), color="black", fontweight="bold",
                fontsize=font_size)
    # for i, num_labels in enumerate(occurrences_night):
    #     ax.text(
    #         i - 0.25,
    #         num_labels + num_labels / 2,
    #         str(num_labels),
    #         color="black",
    #         fontweight="bold",
    #         fontsize=font_size,
    #     )
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5, zorder=0)
    plt.xticks(rotation=45, ha="right", fontsize=font_size)
    use_log_scale = True
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_yticks([0.8, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])


        # Define a function to format the y-axis tick labels
        def log_tick_formatter(val, pos=None):
            if val < 1:
                return 0
            else:
                return r"$10^{{{}}}$".format(int(np.log10(val)))


        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    else:
        ax.set_yticks([10, 100, 1000, 10000, 100000])
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel(r"\# 3D box labels")
    plt.axhline(y=average_class_occurrences, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(2, average_class_occurrences + 500, str(int(average_class_occurrences)), color="r", fontweight="bold",
             fontsize=font_size)
    # print total number of 3D object labels
    plt.text(0.5, average_class_occurrences + 10000, "Total 3D box detections: " + str(total_3d_box_labels),
             color="black",
             fontweight="bold")
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "bar_chart_class_occurrences_all_drives.pdf"))
    plt.close()
    plt.clf()

    #########################################################
    # 2. Bar chart of average and max. number of points for each class
    #########################################################
    fig, ax = plt.subplots(figsize=(5, 4.0))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.27)
    class_names = classes.keys()
    min_num_points_all_classes = [np.min(class_num_points) for class_num_points in classes.values()]
    avg_num_points_all_classes = [np.mean(class_num_points) for class_num_points in classes.values()]
    max_num_points_all_classes = [np.max(class_num_points) for class_num_points in classes.values()]
    total_average_num_points = np.mean([np.mean(class_num_points) for class_num_points in classes.values()])
    print(f"Total average number of points: {total_average_num_points}")
    # plot max. number of points
    plt.bar(
        class_names,
        max_num_points_all_classes,
        color=PlotUtils.get_class_colors(alpha=0.5),
        edgecolor="black",
    )
    # plot avg. number of points
    plt.bar(
        class_names,
        avg_num_points_all_classes,
        color=PlotUtils.get_class_colors(alpha=0.5),
        edgecolor="black",
        hatch="///",
    )
    # plot min. number of points
    plt.bar(
        class_names,
        min_num_points_all_classes,
        color=PlotUtils.get_class_colors(alpha=0.5),
        edgecolor="black",
        # hatch="///",
    )
    # add text labels for avg. number of points
    for i, num_points in enumerate(avg_num_points_all_classes):
        ax.text(
            i - 0.25, num_points + num_points / 5, str(int(num_points)), color="black", fontweight="bold",
            fontsize=font_size
        )
    # add text labels for max. number of points
    for i, num_points in enumerate(max_num_points_all_classes):
        ax.text(
            i - 0.25, num_points + num_points / 5, str(int(num_points)), color="black", fontweight="bold",
            fontsize=font_size
        )
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5, zorder=0)
    plt.xticks(rotation=45, ha="right")
    use_log_scale = True
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_yticks([0.8, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])


        # Define a function to format the y-axis tick labels
        def log_tick_formatter(val, pos=None):
            if val < 1:
                return 0
            else:
                return r"$10^{{{}}}$".format(int(np.log10(val)))


        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

    plt.ylabel(r"$\#$ 3D points", fontsize=font_size)
    # add horizontal line indicating the average across all classes
    # TODO: do not hard code average value
    plt.axhline(y=total_average_num_points, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(0 - 0.3, total_average_num_points + 100, str(int(round(total_average_num_points))), color="r",
             fontweight="bold", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "bar_chart_avg_num_points_per_class.pdf"))
    plt.close()
    plt.clf()

    ##################################################3
    # 3. Histogram for number of points
    ##################################################3
    # x-axis: number of points bins (0-100, 100-200, 200-300, 300-400, 400-500, 500-600, 600-700, 700-800, 800-900, 900-1000,1000-2000)
    # y-axis: number of 3d box labels
    # fig, ax = plt.subplots(figsize=(5, 4.5))
    # plt.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.12)

    # 16:10 = 8:5 = 4:2.5
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.2)

    # NOTE: either use num bins (int) or a bin range (list)
    # 20 bins
    # bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 7000]
    # generate bins from 0 to 18000 in steps of 500
    # bins = np.arange(0, 19000, 1000)
    # bins_original = copy.copy(bins)
    # bin_labels = bins
    # use 11 bin labels
    # bin_labels[-1] = 1000
    use_log_scale = True
    # num_bins = len(bins) - 1
    # y_values, histogram_bins = np.histogram(num_objects_per_num_points, bins=bins)

    # aggregate/accumulate array num_objects_per_num_points into 19 bins [0-1000, 1000-2000,..., 17000-18000]
    # y_values = np.zeros(len(bins))
    # bin_indices = np.digitize(num_objects_per_num_points, bins)
    # for i in range(num_objects_per_num_points.shape[0]):
    #     y_values[bin_indices[i] - 1] += int(num_objects_per_num_points[i])

    # convert y_values to integer
    # y_values = y_values.astype(int)

    # color_bar_labels = tuple([str(y_value / 1000) + "k" for y_value in y_values])
    # y_max = 5000
    # generate y-axis labels from 1k to 7k
    # color_bar_labels = [str(i) + "k" for i in range(0, 8)]
    step_size = 200

    x_values = np.arange(0, 2000, step_size)
    # add 18000 to x_values
    # x_values = np.append(x_values, 18000)

    # extract y_values for each bin of histogram
    # iterate all items of histogram dict
    y_values = []
    for bin_label, num_objects in histogram.items():
        y_values.append(num_objects)

    # TEMO set to 2000
    # y_values_first_bar = y_values[0]
    # remove first element from y_values
    # y_values = y_values[1:]
    # x_values_first_bar = x_values[0]
    # x_values = x_values[1:]

    # ax.bar(np.array(bins_original) + 100, y_values, width=200, color='b', edgecolor="black", zorder=3)

    ax.bar(x_values + 100, y_values, width=200, color='b', edgecolor="black", zorder=100)
    # ax.bar(x_values_first_bar + 100, y_values_first_bar, width=200, color='b', edgecolor="black", zorder=100)

    # fill first bar in red color
    # ax.patches[0].set_facecolor("r")

    y_max = max(y_values)
    # round y_max to the next 500
    y_max = int(y_max / 500) * 500 + 500
    cm = plt.cm.get_cmap("RdYlBu_r")
    # Get the colormap colors
    my_cmap = cm(np.arange(cm.N))
    # Set alpha
    my_cmap[:, -1] = np.linspace(0.5, 0.5, cm.N)

    # use last 50% of the colormap
    # my_cmap = my_cmap[int(cm.N / 2):, :]

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    # iterate all bars and set gradient color using the colormap
    for y_value, p in zip(y_values, ax.patches):
        # calculate exponential color value
        color = my_cmap((y_value) / (y_max))
        plt.setp(p, "facecolor", color)

    if use_log_scale:
        ax.set_yscale("log")
        # ax.set_ylim((0, 4))
        ax.set_yticks([0.8, 10, 100, 1000, 10000])


        # Define a function to format the y-axis tick labels
        def log_tick_formatter(val, pos=None):
            if val < 1:
                return 0
            # else:
            value = int(log10(int(val)))
            return str(r"$10^{%01d}$" % value)


        # Set the y-axis tick labels to show the actual values
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    else:
        ax.set_yticks(np.arange(0, 4500, 500))

    # show average (using all boxes incl. empty boxes)
    # avg_value = total_num_points / total_3d_box_labels
    # TODO: show average (using all boxes excl. empty boxes)
    avg_value = total_num_points / total_3d_box_labels_not_empty
    plt.axvline(x=avg_value, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(avg_value + 50, 15000, str(round(avg_value, 2)), color="r", fontsize=font_size)
    # # show total number of labels
    plt.text(800, 3800, "Total points: " + str(round(total_num_points)), color="black", fontsize=font_size)

    # show y values
    for idx, y_value in enumerate(y_values):
        y_pos = None
        if y_value == 0:
            y_pos = 100
        else:
            y_pos = y_value + y_value / 10
        ax.text(idx * 200 + 20, y_pos, str(y_value), color="black", fontweight="bold", fontsize=font_size)

    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)
    # make the x-axis values start on the very left (no padding, no white space)
    plt.margins(x=0)
    # make y-axis values start at 0
    plt.ylim((0, 50000))
    # make the x tick start at 0
    plt.xlim((0, 2000))
    # set x axis label to "# points"
    plt.xlabel(r"\# 3D points", fontsize=font_size)
    # set y axis label to "# 3D box labels"
    plt.ylabel(r"\# 3D box labels", fontsize=font_size)

    # add legend with color map
    Z = [[0, 0], [0, 0]]
    # levels = range(0, y_max, step_size)
    # generate 10 levels between 0 and 10000
    color_bar_step_size = 10

    levels = np.linspace(0, y_max, color_bar_step_size + 1)

    contour = plt.contourf(Z, levels, cmap=my_cmap)
    color_bar = plt.colorbar(contour)
    color_bar.ax.tick_params(labelsize=font_size)
    # generate y-axis labels from 1k to 10k
    # color_bar_labels = [str(i) + "k" for i in range(0, 9)]
    # color_bar_labels = tuple([str(y_value / 1000) + "k" for y_value in y_values])
    color_bar_labels = None
    if color_bar_labels is not None:
        color_bar.ax.set_yticklabels(color_bar_labels)

    ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # right now the x-axis contains these labels: 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 18000
    # we want to change it to: 0, 400, 800, 1200, 1600, 18000
    # ax.set_xticks([0, 400, 800, 1200, 1600, 18000])

    # ax.set_xticklabels(["0", "200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "18000"])
    ax.set_xticklabels(["1", "", "400", "", "800", "", "1200", "", "1600", "", "18000"])

    # show horizontal grid lines with transparency and move them behind the plot elements
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5, zorder=0)

    plt.savefig(
        os.path.join(output_folder_path_statistic_plots, "histogram_num_objects_per_num_points_test.pdf"))
    plt.close()
    plt.clf()

    ##################################################3
    # 4. Line chart for avg. number of points (depended on distance) for each class
    ##################################################3
    # x-axis: distance
    # y-axis: avg. number of points per objects
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans",
            "font.serif": "Computer Modern Roman",
        }
    )
    # fig, ax = plt.subplots(figsize=(5, 4.5))
    # plt.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.12)
    # 16:10 = 8:5 = 4:2.5
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.2)
    plt.xlabel(r"Distance [m]", fontsize=font_size)
    plt.ylabel(r"$\emptyset$ points", fontsize=font_size)
    distances = np.linspace(0, 60, 7).astype(int)
    plt.xticks(distances)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # ax.set_yscale("log")
    # after fusion
    plt.yticks(np.arange(0, 4800, 400))
    # before fusion
    # plt.yticks(np.arange(0, 2750, 250))

    class_colors = PlotUtils.get_class_colors(alpha=1.0)
    average_points_per_bin = {
        "0-10": [],
        "10-20": [],
        "20-30": [],
        "30-40": [],
        "40-50": [],
        "50-60": [],
        # "60-70": [],
        # "70-80": [],
    }
    for i, (object_class_name, object_class_obj) in enumerate(classes_and_distances.items()):
        avg_num_points = [0]
        for j, (distance_bin, num_points) in enumerate(object_class_obj.items()):
            avg_num_points.append(0 if len(num_points) == 0 else np.mean(num_points))
            average_points_per_bin[distance_bin].append(avg_num_points[-1])
        if object_class_name == "EMERGENCY_VEHICLE":
            object_class_name = "EMERGENCY_VEH"
        plt.plot(
            distances,
            avg_num_points,
            label=object_class_name,
            color=class_colors[i],
            linewidth=3,
            path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()],
            markersize=5,
            markerfacecolor=(23 / 255, 87 / 255, 217 / 255, 1.0),
            markeredgecolor=(23 / 255, 87 / 255, 217 / 255, 1.0),
            markeredgewidth=1,
        )
    # iterate over all distance bins and plot the average number of points per bin
    for distance_bin, avg_num_points in average_points_per_bin.items():
        average_points_per_bin[distance_bin] = np.mean(avg_num_points)
    plt.plot(
        distances,
        [0] + list(average_points_per_bin.values()),
        label="Average",
        color="red",
        linestyle="--",
    )

    # plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=font_size-2, frameon=True)
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)
    plt.savefig(
        os.path.join(output_folder_path_statistic_plots, "line_chart_avg_num_points_with_distance_train.pdf"))
    plt.close()
    plt.clf()

    ##################################################
    # 5. Box plot for num points per class (10 box plots)
    ##################################################
    # x-axis: 10 classes (color coded)
    # y-axis: box plots (min_number_of_points, avg_num_points, max_number_of_points). use standard deviation for start and end of box.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.4)
    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.ylabel(r"\#points", fontsize=font_size)
    plt.yticks(np.arange(0, 420, 40))
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Computer Modern Roman",
        }
    )
    data = []
    medians = []
    for object_class_obj in classes.keys():
        data.append(classes[object_class_obj])
        medians.append(np.median(classes[object_class_obj]))
    box_plot = plt.boxplot(
        data,
        patch_artist=True,  # fill with color
        vert=True,  # vertical box alignment
        medianprops=dict(color=(255 / 255, 0 / 255, 0 / 255, 1.0)),
        showfliers=False,
    )
    plt.setp(box_plot["boxes"], color="black")
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    for patch, color in zip(box_plot["boxes"], class_colors):
        patch.set_facecolor(color)
    xtickNames = plt.setp(ax, xticklabels=class_names)
    plt.setp(xtickNames, rotation=45)
    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(len(classes.keys())) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ["bold", "semibold"]
    top = 440
    for tick, label in zip(range(10), ax.get_xticklabels()):
        k = tick % 2
        ax.text(
            pos[tick],
            top - (top * 0.05),
            upperLabels[tick],
            horizontalalignment="center",
            # size="x-small",
            weight=weights[k],
            color="black",
            fontsize=font_size,
        )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "box_plot_num_points_for_each_class.pdf"))
