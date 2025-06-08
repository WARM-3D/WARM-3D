import json
from pathlib import Path

import cv2
import argparse
import sys
import os
import numpy as np

# This script visualizes 2D and/or 3D labels on top of camera images.
# Usage:
#           python visualize_image_with_labels_all_frames.py --input_folder_path_image_sequence <INPUT_FOLDER_PATH_IMAGE_SEQUENCE> --input_folder_path_labels <INPUT_FOLDER_PATH_LABELS> --output_folder_path visualization -b box2d_and_box3d_projected -c by_class
# Example:
# python visualize_image_with_labels_all_frames.py --input_folder_path_image_sequence a9_dataset/r00_s00/_images/s040_camera_basler_north_16mm --input_folder_path_labels a9_dataset/r00_s00/_labels/s040_camera_basler_north_16mm --output_folder_path a9_dataset/r00_s00/_images_visualized/s040_camera_basler_north_16mm -b box2d_and_box3d -c by_class
from src.utils.perspective import parse_perspective
from src.utils.utils import id_to_class_name_mapping
from src.utils.vis_utils import VisualizationUtils

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


def draw_line(img, start_point, end_point, color):
    cv2.line(img, start_point, end_point, color, 2)


def get_color_by_name(color_name):
    color_mapping = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (169, 49, 52),
        "Green": (56, 132, 76),
        "Blue": (7, 25, 120),
        "Yellow": (179, 179, 57),
        "Gray": (128, 128, 128),
        "Brown": (77, 53, 19),
        "Orange": (228, 85, 7),
    }
    return color_mapping[color_name]


def get_color_by_category(category):
    if category.upper() == "CAR":
        color = "#00CCF6"
    elif category.upper() == "TRUCK":
        color = "#56FFB6"
    elif category.upper() == "TRAILER":
        color = "#5AFF7E"
    elif category.upper() == "VAN":
        color = "#EBCF36"
    elif category.upper() == "MOTORCYCLE":
        color = "#B9A454"
    elif category.upper() == "BUS":
        color = "#D98A86"
    elif category.upper() == "PEDESTRIAN":
        color = "#E976F9"
    elif category.upper() == "BICYCLE":
        color = "#B18CFF"
    elif category.upper() == "SPECIAL_VEHICLE" or category.upper() == "EMERGENCY_VEHICLE":
        color = "#666bfa"
    elif category.upper() == "OTHER" or category.upper() == "OTHER_VEHICLES":
        # NOTE: r00_s00 contains the class "Other Vehicles", whereas r00_s01 - r00_s04 contain the class "OTHER"
        color = "#C7C7C7"
    elif category.upper() == "LICENSE_PLATE_LOCATION":
        color = "#000000"
    else:
        print("Unknown category: ", category.upper())
    return color


def draw_boxes_2d(img, input_folder_path_boxes, boxes_file_name, img_width, img_height, use_two_colors, input_box_type):
    with open(os.path.join(input_folder_path_boxes, boxes_file_name), "r") as file_reader:
        lines = file_reader.readlines()
    for line in lines:
        values = line.strip().split(" ")
        object_class_id = values[0]
        x_center = float(values[1]) * img_width
        y_center = float(values[2]) * img_height
        width = float(values[3]) * img_width
        height = float(values[4]) * img_height
        if use_two_colors and input_box_type == "detections":
            color_rgb = (245, 44, 71)  # red
        elif use_two_colors and input_box_type == "labels":
            color_rgb = (27, 250, 27)  # green
        else:
            color_rgb = id_to_class_name_mapping[str(object_class_id)]["color_rgb"]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        bbox = [
            int(x_center - width / 2.0),
            int(y_center - height / 2.0),
            int(x_center + width / 2.0),
            int(y_center + height / 2.0),
        ]
        utils.draw_2d_box(img=img, box_label=bbox, color=color_bgr, line_width=3)


def process_data(img, label_data, viz_color_mode, viz_box_mode, utils, camera_id, perspective):
    if "openlabel" in label_data:
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            for object_track_id, object_json in frame_obj["objects"].items():
                print("object id: ", object_track_id)
                object_data = object_json["object_data"]
                if object_data["type"] == "LICENSE_PLATE_LOCATION":
                    line_width = 1
                else:
                    line_width = 3
                color = None
                if viz_color_mode == "by_class":
                    color = get_color_by_category(object_data["type"])
                    color_rgb = utils.hex_to_rgb(color)
                elif viz_color_mode == "by_physical_color":
                    color = object_data["keypoints_2d"]["attributes"]["text"][0]["val"]
                    color_rgb = get_color_by_name(color)

                # TODO: fix bug: all labels are pedestrians in detection_processing package
                # red = detection processing
                # color_rgb = (255, 96, 96)
                # green = labels
                # color_rgb = (96, 255, 96)
                # blue = mono3d
                # color_rgb = (52, 192, 235)

                # swap channels because opencv uses bgr
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                if "box2d" in viz_box_mode:
                    bbox = object_data["bbox"][0]["val"]
                    bounding_box_2d = [
                        int(bbox[0] - bbox[2] / 2.0),
                        int(bbox[1] - bbox[3] / 2.0),
                        int(bbox[0] + bbox[2] / 2.0),
                        int(bbox[1] + bbox[3] / 2.0),
                    ]
                    utils.draw_2d_box(img, bounding_box_2d, color_bgr, line_width)

                if "mask" in viz_box_mode:
                    mask = object_data["poly2d"][0]["val"]
                    # TODO extract mask width and mask height
                    mask_height = 1920
                    mask_width = 1920
                    # mask_height = object_data["poly2d"]["mask_height"]
                    # mask_width = object_data["poly2d"]["mask_width"]
                    utils.draw_mask(img, mask, color_bgr, mask_height, mask_width)

                if "box3d" in viz_box_mode:
                    if object_data["type"] == "LICENSE_PLATE_LOCATION":
                        continue

                    box_3d = {}
                    box_3d["box3d_projected"] = {}
                    if "keypoints_2d" in object_data:
                        box_3d["box3d_projected"]["bottom_left_front"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][0]["point2d"]["val"]
                        box_3d["box3d_projected"]["bottom_left_back"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][1]["point2d"]["val"]
                        box_3d["box3d_projected"]["bottom_right_back"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][2]["point2d"]["val"]
                        box_3d["box3d_projected"]["bottom_right_front"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][3]["point2d"]["val"]

                        box_3d["box3d_projected"]["top_left_front"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][4]["point2d"]["val"]
                        box_3d["box3d_projected"]["top_left_back"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][5]["point2d"]["val"]
                        box_3d["box3d_projected"]["top_right_back"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][6]["point2d"]["val"]
                        box_3d["box3d_projected"]["top_right_front"] = object_data["keypoints_2d"]["attributes"][
                            "points2d"
                        ]["val"][7]["point2d"]["val"]
                        utils.draw_3d_box_by_keypoints(
                            img,
                            box_3d,
                            color_bgr,
                            camera_id,
                            lidar_id="",
                            use_boxes_in_s110_base=True,
                            perspective=perspective,
                        )
                    else:
                        # manually project 3d box into image
                        # np.array(object_data["cuboid"]["val"]
                        cuboid = np.array(object_data["cuboid"]["val"])
                        if np.all(cuboid == 0):
                            continue
                        # corners = get_corners(cuboid)
                        # box_3d["box3d_projected"]["bottom_left_front"] = utils.project_3d_position(corners[0])
                        # box_3d["box3d_projected"]["bottom_left_back"] = utils.project_3d_position(corners[1])
                        # box_3d["box3d_projected"]["bottom_right_back"] = utils.project_3d_position(corners[2])
                        # box_3d["box3d_projected"]["bottom_right_front"] = utils.project_3d_position(corners[3])
                        #
                        # box_3d["box3d_projected"]["top_left_front"] = utils.project_3d_position(corners[4])
                        # box_3d["box3d_projected"]["top_left_back"] = utils.project_3d_position(corners[5])
                        # box_3d["box3d_projected"]["top_right_back"] = utils.project_3d_position(corners[6])
                        # box_3d["box3d_projected"]["top_right_front"] = utils.project_3d_position(corners[7])
                        # is_within_image = utils.check_within_image(box_3d["box3d_projected"])
                        # if is_within_image:
                        # draw_3d_box(self, img, box_label, color, camera_id, lidar_id, use_boxes_in_s110_base, perspective, input_type):
                        utils.draw_3d_box_by_keypoints(
                            img,
                            object_json,
                            color_bgr,
                            camera_id=camera_id,
                            lidar_id="",
                            use_boxes_in_s110_base=True,
                            perspective=perspective,
                        )
                        # category = object_data["name"]
                        # quaternion = cuboid[3:7]
                        # roll, pitch, yaw = R.from_quat(quaternion).as_euler('xyz', degrees=True)
                        # label = category.split("_")[1] + " yaw: " + str(int(yaw))
                        # x_min = box_3d["box3d_projected"]["top_left_back"][0]
                        # y_min = box_3d["box3d_projected"]["top_left_back"][1]
                        # x_max = box_3d["box3d_projected"]["bottom_right_front"][0]
                        # y_max = box_3d["box3d_projected"]["bottom_right_front"][1]
                        # utils.plot_banner(img, x_min, y_min, x_max, y_max, color_bgr, label)
                        # else:
                        #     continue

                if "position3d" in viz_box_mode:
                    pos_x, pos_y = utils.project_3d_position(np.array(object_data["cuboid"]["val"]))
                    cv2.circle(img, (pos_x, pos_y), 5, color_bgr, thickness=-1)
                    cv2.circle(img, (pos_x, pos_y), 6, (0, 0, 0), thickness=2)

                if (
                        "box2d" not in viz_box_mode
                        and "box3d" not in viz_box_mode
                        and "position3d" not in viz_box_mode
                        and "mask" not in viz_box_mode
                ):
                    print(
                        "Error. Unknown box mode: {}. Possible visualization modes are: [box2d, box3d, position3d]. Exiting...".format(
                            viz_box_mode
                        )
                    )
                    sys.exit()
    else:
        for box_label in label_data["labels"]:
            color = None
            if viz_color_mode == "by_class":
                color = get_color_by_category(box_label["category"].upper())
                color_rgb = utils.hex_to_rgb(color)
            elif viz_color_mode == "by_physical_color":
                color = box_label["color_body"]
                color_rgb = get_color_by_name(color)
            # swap channels because opencv uses bgr
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            if "box2d" in viz_box_mode:
                utils.draw_2d_box(img=img, box_label=box_label["box2d"], color=color_bgr, line_width=3)
            if "box3d" in viz_box_mode:
                utils.draw_3d_box_by_keypoints(img, box_label, color_bgr, normalized=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--sensor_id", default="", help="Sensor ID, e.g. s040_camera_basler_north_16mm, s110_camera_basler_south1_8mm"
    )
    argparser.add_argument(
        "-i", "--input_folder_path_images", default="images", help="Input image sequence folder path. Default: images"
    )
    argparser.add_argument("-l", "--input_folder_path_labels", default="", help="Input label folder path.")
    argparser.add_argument("-d", "--input_folder_path_detections", default="", help="Input detections folder path.")
    argparser.add_argument(
        "-o", "--output_folder_path", help="Output folder path to save visualization results to disk."
    )
    argparser.add_argument(
        "-b",
        "--viz_box_mode",
        default="box3d",
        help="Visualization box mode. Available modes are: [box2d, mask, box3d, position3d]. Combinations also possible with a comma in between, e.g. box2d,box3d",
    )
    argparser.add_argument(
        "-c",
        "--viz_color_mode",
        default="by_class",
        help="Visualization color mode. Available modes are: [by_class, by_physical_color]",
    )
    argparser.add_argument(
        "--input_format", default="openlabel", help="Input format. Available format: [openlabel, mscoco]"
    )
    argparser.add_argument("--file_path_calibration_data", default="", help="File path to calibration data.")
    args = argparser.parse_args()

    sensor_id = args.sensor_id
    input_folder_path_images = args.input_folder_path_images
    input_folder_path_labels = args.input_folder_path_labels
    input_folder_path_detections = args.input_folder_path_detections
    output_folder_path = args.output_folder_path
    viz_box_mode = args.viz_box_mode
    viz_color_mode = args.viz_color_mode
    input_format = args.input_format
    file_path_calibration_data = args.file_path_calibration_data
    utils = VisualizationUtils()

    image_file_names = sorted(os.listdir(os.path.join(input_folder_path_images)))

    if input_folder_path_labels != "":
        label_file_names = sorted(os.listdir(os.path.join(input_folder_path_labels)))
    else:
        label_file_names = [""] * len(image_file_names)

    if input_folder_path_detections != "":
        detection_file_names = sorted(os.listdir(os.path.join(input_folder_path_detections)))
    else:
        detection_file_names = [""] * len(image_file_names)

    if len(image_file_names) != len(label_file_names) or len(image_file_names) != len(detection_file_names):
        print("Error: Make sure the number of image files matches the number of detection/label files.")
        sys.exit()

    camera_perspectives = {"s110_camera_basler_south1_8mm": None, "s110_camera_basler_south2_8mm": None}
    parse_camera_id = sensor_id == ""

    for image_file_name, detection_file_name, label_file_name in zip(
            image_file_names, detection_file_names, label_file_names
    ):
        print("Processing image file: " + image_file_name)

        if parse_camera_id:
            sensor_id = "_".join(os.path.splitext(image_file_name).split("_")[2:])

        if camera_perspectives[sensor_id] is None:
            camera_perspectives[sensor_id] = parse_perspective(file_path_calibration_data)
            # camera_perspectives[sensor_id].initialize_matrices()

        img = cv2.imread(os.path.join(input_folder_path_images, image_file_name), cv2.IMREAD_UNCHANGED)

        # get width and height
        img_height, img_width, channels = img.shape
        if input_format == "openlabel":
            if label_file_name:
                label_data = json.load(open(os.path.join(input_folder_path_labels, label_file_name)))
                process_data(
                    img, label_data, viz_color_mode, viz_box_mode, utils, sensor_id, camera_perspectives[sensor_id]
                )
            if detection_file_name:
                detection_data = json.load(open(os.path.join(input_folder_path_detections, detection_file_name)))
                process_data(
                    img, detection_data, viz_color_mode, viz_box_mode, utils, sensor_id, camera_perspectives[sensor_id]
                )

        elif input_format == "mscoco":
            lines = []
            use_two_colors = (label_file_name is not None) and (detection_file_name is not None)
            if label_file_name is not None:
                draw_boxes_2d(
                    img,
                    input_folder_path_labels,
                    label_file_name,
                    img_width,
                    img_height,
                    use_two_colors,
                    input_box_type="labels",
                )
            if detection_file_name is not None:
                draw_boxes_2d(
                    img,
                    input_folder_path_detections,
                    detection_file_name,
                    img_width,
                    img_height,
                    use_two_colors,
                    input_box_type="detections",
                )

        if output_folder_path:
            if not os.path.isdir(os.path.join(output_folder_path, sensor_id)):
                Path(os.path.join(output_folder_path, sensor_id)).mkdir(parents=True)
            cv2.imwrite(os.path.join(output_folder_path, sensor_id, image_file_name), img)
        else:
            cv2.imshow("image", img)
            cv2.waitKey()
