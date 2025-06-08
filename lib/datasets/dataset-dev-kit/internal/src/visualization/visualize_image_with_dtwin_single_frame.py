import json
import cv2
import argparse
import sys
import os
import numpy as np


# Example:
# python visualize_image_with_dtwin_single_frame.py -i 03_images/1611481810_938000000_s40_camera_basler_north_16mm.jpg -l 04_labels/1611481810_938000000_s40_camera_basler_north_16mm.json -c 05_calibration/ -o 06_visualization/1611481810_938000000_s40_camera_basler_north_16mm_with_dtwin.jpg -m box2d_and_box3d_projected


def draw_line(img, start_point, end_point, color):
    cv2.line(img, start_point, end_point, color, 1)


class Utils:

    def __init__(self):
        pass

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def load_calibration_data(self, input_folder_path_calibration):
        calib_data = {
            "s40_camera_basler_north_16mm": None,
            "s40_camera_basler_north_50mm": None,
            "s50_camera_basler_south_16mm": None,
            "s50_camera_basler_south_50mm": None,
        }
        for key, value in calib_data.items():
            print(key)
            json_file = open(os.path.join(input_folder_path_calibration, key + ".json"), )
            json_data = json.load(json_file)
            calib_data[key] = json_data
        return calib_data

    def draw_3d_position_and_track_id(self, img, position_3d, color_bgr, sensor_calib_data, track_id):
        projection_matrix = sensor_calib_data["projection_matrix"]
        projection_matrix = np.array(projection_matrix)
        print(projection_matrix.shape)
        position_3d = np.array([position_3d["x"], position_3d["y"], position_3d["z"], 1])
        print(position_3d.shape)
        # 3x4 4x1
        pos_2d = np.matmul(projection_matrix, position_3d)
        if pos_2d[2] > 0:
            pos_2d[0] /= pos_2d[2]
            pos_2d[1] /= pos_2d[2]
            print(pos_2d[0])
            print(pos_2d[1])
            cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), radius=2, color=(255, 255, 255), thickness=2,
                       lineType=cv2.LINE_8)
            print(track_id)
            cv2.putText(img, str(track_id), (int(pos_2d[0]), int(pos_2d[1]) - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='VizLabel Argument Parser')
    argparser.add_argument('-i', '--image_file_path', default="image.jpg", help='Image file path. Default: image.jpg')
    argparser.add_argument('-l', '--label_file_path', default="label.json", help='Label file path. Default: label.json')
    argparser.add_argument('-c', '--input_folder_path_calibration', default="calibration.json",
                           help='Calibration folder path. Default: 05_calibration')
    argparser.add_argument('-o', '--output_file_path', help='Output file path to save visualization result to disk.')
    args = argparser.parse_args()

    image_file_path = args.image_file_path
    label_file_path = args.label_file_path
    input_folder_path_calibration = args.input_folder_path_calibration
    output_file_path = args.output_file_path

    parts = image_file_path.split("/")[-1].split(".")[0].split("_")
    print(parts)
    sensor_id = parts[2] + "_" + parts[3] + "_" + parts[4] + "_" + parts[5] + "_" + parts[6]

    utils = Utils()
    calib_data = utils.load_calibration_data(input_folder_path_calibration)
    img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)

    data = open(label_file_path, )
    labels = json.load(data)

    for box_label in labels["labels"]:
        color = None
        if box_label["category"] == "CAR":
            color = "#00CCF6"
        elif box_label["category"] == "TRUCK":
            color = "#56FFB6"
        elif box_label["category"] == "TRAILER":
            color = "#5AFF7E"
        elif box_label["category"] == "VAN":
            color = "#EBCF36"
        elif box_label["category"] == "MOTORCYCLE":
            color = "#B9A454"
        elif box_label["category"] == "BUS":
            color = "#D98A86"
        elif box_label["category"] == "PEDESTRIAN":
            color = "#E976F9"
        elif box_label["category"] == "SPECIAL_VEHICLE":
            color = "#C7C7C7"
        else:
            print("found not supported category: ", str(box_label["category"]))
            sys.exit()
        color_rgb = utils.hex_to_rgb(color)
        # swap channels because opencv uses bgr
        color_bgr = (color_rgb[1], color_rgb[1], color_rgb[0])
        utils.draw_3d_position_and_track_id(img, box_label["box3d"]["location"], color_bgr, calib_data[sensor_id],
                                            box_label["id"])

    if output_file_path:
        cv2.imwrite(output_file_path, img)
    else:
        cv2.imshow("image", img)
        cv2.waitKey()
