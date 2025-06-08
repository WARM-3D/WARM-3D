import json
import cv2
import argparse
import sys
import os
import numpy as np

# Example:
# python visualize_image_with_dtwin_all_frames.py -i 03_images/ -l 04_labels/ -c 05_calibration -o 06_visualization

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


def draw_line(img, start_point, end_point, color):
    cv2.line(img, start_point, end_point, color, 1)


class Utils:

    def __init__(self):
        pass

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def draw_2d_box(self, img, box_label, color):
        x_min = int(box_label[2] * IMAGE_WIDTH)
        y_min = int(box_label[0] * IMAGE_HEIGHT)
        x_max = int(box_label[3] * IMAGE_WIDTH)
        y_max = int(box_label[1] * IMAGE_HEIGHT)
        cv2.rectangle(img, (x_min, y_min),
                      (x_max, y_max), color, 1)
        # cv2.rectangle(img, (50, 100), (200, 400), color, 1)

    def draw_3d_box(self, img, corners_2d, color):
        # draw bottom 4 lines
        draw_line(img, corners_2d[0], corners_2d[1], color)
        draw_line(img, corners_2d[1], corners_2d[2], color)
        draw_line(img, corners_2d[2], corners_2d[3], color)
        draw_line(img, corners_2d[3], corners_2d[0], color)

        # draw top 4 lines
        draw_line(img, corners_2d[4], corners_2d[5], color)
        draw_line(img, corners_2d[5], corners_2d[6], color)
        draw_line(img, corners_2d[6], corners_2d[7], color)
        draw_line(img, corners_2d[7], corners_2d[4], color)

        # draw 4 vertical lines
        draw_line(img, corners_2d[0], corners_2d[4], color)
        draw_line(img, corners_2d[1], corners_2d[5], color)
        draw_line(img, corners_2d[2], corners_2d[6], color)
        draw_line(img, corners_2d[3], corners_2d[7], color)

    def get_color_by_category(self, category):
        if category == "CAR":
            color = "#00CCF6"
        elif category == "TRUCK":
            color = "#56FFB6"
        elif category == "TRAILER":
            color = "#5AFF7E"
        elif category == "VAN":
            color = "#EBCF36"
        elif category == "MOTORCYCLE":
            color = "#B9A454"
        elif category == "BUS":
            color = "#D98A86"
        elif category == "PEDESTRIAN":
            color = "#E976F9"
        elif category == "SPECIAL_VEHICLE":
            color = "#C7C7C7"
        else:
            print("found not supported category: ", str(dtwin_obj["category"]))
            sys.exit()
        return color

    def draw_3d_position_and_track_id(self, img, position_3d_center, color_bgr, sensor_calib_data, track_id,
                                      vehicle_default_sizes, category, sensor_id):
        projection_matrix = sensor_calib_data["projection_matrix"]
        projection_matrix = np.array(projection_matrix)
        # projection matrix (that was taken from pro-anno) to visualize labels (not dtwin)
        # projection_matrix = np.array([[9.40487461e+02, -2.82009326e+03, -2.01081142e+02, -1.48499626e+04],
        #                               [-3.11563016e+01, 1.47347593e+01, -2.86468986e+03, 2.42678068e+04],
        #                               [9.80955096e-01, 3.42417940e-04, -1.94234351e-01, 8.28553587e-01]], dtype=float)
        position_3d_center = np.array([position_3d_center["x"], position_3d_center["y"], position_3d_center["z"], 1])
        position_3d_corners = []
        vehicle_default_size = vehicle_default_sizes[category]
        # NOTE: if S50 cameras then do +lenght
        #       if S40 cameras then do -lenght
        if sensor_id == "s50_camera_basler_south_16mm" or sensor_id == "s50_camera_basler_south_50mm":
            length = vehicle_default_size[0]
        else:
            length = -vehicle_default_size[0]
        # bottom, front, left
        position_3d_corners.append(np.array(
            [position_3d_center[0],
             position_3d_center[1] + vehicle_default_size[1] / 2,
             position_3d_center[2]]))
        # bottom, back, left
        position_3d_corners.append(np.array(
            [position_3d_center[0] + length,
             position_3d_center[1] + vehicle_default_size[1] / 2,
             position_3d_center[2]]))
        # bottom, back, right
        position_3d_corners.append(np.array(
            [position_3d_center[0] + length,
             position_3d_center[1] - vehicle_default_size[1] / 2,
             position_3d_center[2]]))
        # bottom, front, right
        position_3d_corners.append(np.array(
            [position_3d_center[0],
             position_3d_center[1] - vehicle_default_size[1] / 2,
             position_3d_center[2]]))
        # top, front, left
        position_3d_corners.append(np.array(
            [position_3d_center[0],
             position_3d_center[1] + vehicle_default_size[1] / 2,
             position_3d_center[2] + vehicle_default_size[2]]))
        # top, back, left
        position_3d_corners.append(np.array(
            [position_3d_center[0] + length,
             position_3d_center[1] + vehicle_default_size[1] / 2,
             position_3d_center[2] + vehicle_default_size[2]]))
        # top, back, right
        position_3d_corners.append(np.array(
            [position_3d_center[0] + length,
             position_3d_center[1] - vehicle_default_size[1] / 2,
             position_3d_center[2] + vehicle_default_size[2]]))
        # top, front, right
        position_3d_corners.append(np.array(
            [position_3d_center[0],
             position_3d_center[1] - vehicle_default_size[1] / 2,
             position_3d_center[2] + vehicle_default_size[2]]))

        pos_2d = np.matmul(projection_matrix, position_3d_center)
        if pos_2d[2] > 0:
            pos_2d[0] /= pos_2d[2]
            pos_2d[1] /= pos_2d[2]
            cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), radius=2, color=color_bgr, thickness=2,
                       lineType=cv2.LINE_8)
            cv2.putText(img, str(track_id), (int(pos_2d[0]), int(pos_2d[1]) - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
        corners_2d = []
        for corner in position_3d_corners:
            # 3x4 4x1
            corner_homogeneous = np.array([corner[0], corner[1], corner[2], 1])
            pos_2d = np.matmul(projection_matrix, corner_homogeneous)
            if pos_2d[2] > 0:
                pos_2d[0] /= pos_2d[2]
                pos_2d[1] /= pos_2d[2]
                corners_2d.append([int(pos_2d[0]), int(pos_2d[1])])

        return corners_2d


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='VizLabel Argument Parser')
    argparser.add_argument('-i', '--image_folder_path', default="images", help='Image folder path. Default: images')
    argparser.add_argument('-d', '--folder_path_dtwin_positions', default="dtwin",
                           help='Folder path of dtwin 3D positions. Default: dtwin')
    argparser.add_argument('-b', '--folder_path_2d_detections', default="",
                           help='Folder path of 2D detections. Default: detections')
    argparser.add_argument('-c', '--input_folder_path_calibration', default="calibration.json",
                           help='Calibration folder path. Default: 05_calibration')
    argparser.add_argument('-o', '--output_folder_path',
                           help='Output folder path to save visualization results to disk.')
    argparser.add_argument('-v', '--input_file_path_default_vehicle_sizes', default="config/default_vehicle_sizes.json",
                           help='Default vehicle sizes file path. Default: config/default_vehicle_sizes.json')
    args = argparser.parse_args()

    image_folder_path = args.image_folder_path
    folder_path_dtwin_positions = args.folder_path_dtwin_positions
    folder_path_2d_detections = args.folder_path_2d_detections
    input_folder_path_calibration = args.input_folder_path_calibration
    output_folder_path = args.output_folder_path
    input_file_path_default_vehicle_sizes = args.input_file_path_default_vehicle_sizes

    utils = Utils()

    file_names_images = sorted(os.listdir(image_folder_path))
    file_names_dtwin_positions = sorted(os.listdir(folder_path_dtwin_positions))

    if folder_path_2d_detections != "":
        file_names_2d_detections = sorted(os.listdir(folder_path_2d_detections))
    else:
        file_names_2d_detections = [""] * len(file_names_images)

    if len(file_names_images) != len(file_names_dtwin_positions) or len(file_names_images) != len(
            file_names_2d_detections):
        print(
            "Error: Make sure the number of image files matches the number of dtwin 3D position files and 2D detection files.")
        sys.exit()

    vehicle_default_sizes_file = open(input_file_path_default_vehicle_sizes, )
    vehicle_default_sizes = json.load(vehicle_default_sizes_file)

    calib_data = json.load(open(input_folder_path_calibration, ))

    file_idx = 0
    for image_file_name, file_name_dtwin_positions, file_name_2d_detections in zip(file_names_images,
                                                                                   file_names_dtwin_positions,
                                                                                   file_names_2d_detections):
        # print("Processing image file: " + image_file_name, " . Idx=", str(file_idx))
        img = cv2.imread(os.path.join(image_folder_path, image_file_name), cv2.IMREAD_UNCHANGED)

        dtwin_positions_data = open(os.path.join(folder_path_dtwin_positions, file_name_dtwin_positions), )
        dtwin_positions = json.load(dtwin_positions_data)

        parts = image_file_name.split("/")[-1].split(".")[0].split("_")
        print("image_file_name=", image_file_name)
        print("parts=", parts)
        sensor_id = parts[2] + "_" + parts[3] + "_" + parts[4] + "_" + parts[5] + "_" + parts[6]

        if file_name_2d_detections != "":
            detections_2d_data = open(os.path.join(folder_path_2d_detections, file_name_2d_detections), )
            detections_2d = json.load(detections_2d_data)

            for detection_obj in detections_2d["labels"]:
                category = detection_obj["category"]
                color = utils.get_color_by_category(category)
                color_rgb = utils.hex_to_rgb(color)
                # swap channels because opencv uses bgr
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                # utils.draw_2d_box(img, detection_obj["box2d"], color_bgr)
                x_min = int(detection_obj["box2d"][2] * IMAGE_WIDTH)
                y_min = int(detection_obj["box2d"][0] * IMAGE_HEIGHT) - 3
                print(x_min)
                print(y_min)

                cv2.putText(img, str(category), (x_min, y_min),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3,
                            color=color_bgr, thickness=1, lineType=cv2.LINE_AA)

        for dtwin_obj in dtwin_positions["labels"]:
            color = None
            category = dtwin_obj["category"]
            color = utils.get_color_by_category(category)
            color_rgb = utils.hex_to_rgb(color)
            # swap channels because opencv uses bgr
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            corners_2d = utils.draw_3d_position_and_track_id(img, dtwin_obj["box3d"]["location"], color_bgr,
                                                             calib_data,
                                                             dtwin_obj["id"], vehicle_default_sizes, category,
                                                             sensor_id)
            utils.draw_3d_box(img, corners_2d, color_bgr)

        # print("%d objects in dtwin list. %d objects in 2D detections list" % (
        #     len(dtwin_positions["labels"]), len(detections_2d["labels"])))
        if output_folder_path:
            if not os.path.isdir(output_folder_path):
                os.mkdir(output_folder_path)
            cv2.imwrite(
                output_folder_path + "/" + image_file_name.split(".")[0] + "_with_labels." + image_file_name.split(".")[
                    1], img)
        else:
            cv2.imshow(str(dtwin_obj["id"]), img)
            cv2.waitKey()

        file_idx = file_idx + 1
