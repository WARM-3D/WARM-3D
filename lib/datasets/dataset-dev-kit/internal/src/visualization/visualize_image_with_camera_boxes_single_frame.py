import json
import cv2
import argparse
import sys


# Example:
# python visualize_image_with_labels_single_frame.py -i 03_images/1611481810_938000000_s40_camera_basler_north_16mm.jpg -l 04_labels/1611481810_938000000_s40_camera_basler_north_16mm.json -o 97_visualization_box2d_and_box3d_projected/1611481810_938000000_s40_camera_basler_north_16mm.jpg -m box2d_and_box3d_projected


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
        cv2.rectangle(img, (box_label[0], box_label[1]), (box_label[2], box_label[3]), color, 1)

    def draw_3d_box(self, img, box_label, color):
        # draw bottom 4 lines
        start_point = (int(box_label["box3d_projected"]["bottom_left_front"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_left_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["bottom_left_back"][0] * 1920),
                     int(box_label["box3d_projected"]["bottom_left_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_left_back"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_left_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["bottom_right_back"][0] * 1920),
                     int(box_label["box3d_projected"]["bottom_right_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_right_back"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_right_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["bottom_right_front"][0] * 1920),
                     int(box_label["box3d_projected"]["bottom_right_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_right_front"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_right_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["bottom_left_front"][0] * 1920),
                     int(box_label["box3d_projected"]["bottom_left_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        # draw top 4 lines
        start_point = (int(box_label["box3d_projected"]["top_left_front"][0] * 1920),
                       int(box_label["box3d_projected"]["top_left_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_left_back"][0] * 1920),
                     int(box_label["box3d_projected"]["top_left_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["top_left_back"][0] * 1920),
                       int(box_label["box3d_projected"]["top_left_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_right_back"][0] * 1920),
                     int(box_label["box3d_projected"]["top_right_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["top_right_back"][0] * 1920),
                       int(box_label["box3d_projected"]["top_right_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_right_front"][0] * 1920),
                     int(box_label["box3d_projected"]["top_right_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["top_right_front"][0] * 1920),
                       int(box_label["box3d_projected"]["top_right_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_left_front"][0] * 1920),
                     int(box_label["box3d_projected"]["top_left_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        # draw 4 vertical lines
        start_point = (int(box_label["box3d_projected"]["bottom_left_front"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_left_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_left_front"][0] * 1920),
                     int(box_label["box3d_projected"]["top_left_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_left_back"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_left_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_left_back"][0] * 1920),
                     int(box_label["box3d_projected"]["top_left_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_right_back"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_right_back"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_right_back"][0] * 1920),
                     int(box_label["box3d_projected"]["top_right_back"][1] * 1200))
        draw_line(img, start_point, end_point, color)

        start_point = (int(box_label["box3d_projected"]["bottom_right_front"][0] * 1920),
                       int(box_label["box3d_projected"]["bottom_right_front"][1] * 1200))
        end_point = (int(box_label["box3d_projected"]["top_right_front"][0] * 1920),
                     int(box_label["box3d_projected"]["top_right_front"][1] * 1200))
        draw_line(img, start_point, end_point, color)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='VizLabel Argument Parser')
    argparser.add_argument('-i', '--image_file_path', default="image.jpg", help='Image file path. Default: image.jpg')
    argparser.add_argument('-l', '--label_file_path', default="label.json", help='Label file path. Default: label.json')
    argparser.add_argument('-o', '--output_file_path', help='Output file path to save visualization result to disk.')
    argparser.add_argument('-m', '--viz_mode', default="box3d_projected",
                           help='Visualization mode. Available modes are: [box2d, box3d_projected, box2d_and_box3d_projected]')
    args = argparser.parse_args()

    image_file_path = args.image_file_path
    label_file_path = args.label_file_path
    output_file_path = args.output_file_path
    viz_mode = args.viz_mode

    utils = Utils()
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
        if "box2d" in viz_mode:
            utils.draw_2d_box(img, box_label["box2d"], color_bgr)
        if "box3d_projected" in viz_mode:
            utils.draw_3d_box(img, box_label, color_bgr)

    if output_file_path:
        cv2.imwrite(output_file_path, img)
    else:
        cv2.imshow("image", img)
        cv2.waitKey()
