import json
import os
import cv2
import argparse
from math import sqrt
import matplotlib.cm as cm
import matplotlib as mpl
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import pathlib
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import distance

from visualize_point_cloud_with_lidar_boxes import box_center_to_corner


# Example:
# python visualize_image_with_labels_single_frame.py -i 03_images/1611481810_938000000_s40_camera_basler_north_16mm.jpg -l 04_labels/1611481810_938000000_s40_camera_basler_north_16mm.json -o 97_visualization_box2d_and_box3d_projected/1611481810_938000000_s40_camera_basler_north_16mm.jpg -m box2d_and_box3d_projected


def draw_line(img, start_point, end_point, color):
    cv2.line(img, start_point, end_point, color, 3)


def check_intersection(polygon1, polygon2):
    intersection = False
    count = 0
    for point in polygon2:
        result = cv2.pointPolygonTest(
            polygon1, (int(point[0]), int(point[1])), measureDist=False
        )
        # if point inside return 1
        # if point outside return -1
        # if point on the contour return 0

        if result == 1:
            count += 1
            if count > 3:
                intersection = True

    return intersection


class Utils:
    # calibration data for r01_s04 from bohan (manual lidar camera calibration using HD Maps, September 2022)
    # projection_matrix = np.array([[1.20587571e+03, -3.59496726e+02, -6.63647142e+02, -1.28523658e+03],
    #                               [4.66880759e+01, 9.72902003e+00, -1.59129168e+03, -3.90047943e+02],
    #                               [5.68153051e-01, 4.48182446e-01, -6.90169983e-01, -2.81123853e+00]], dtype=float)

    # calibration data for r01_s04 from walter (manual lidar camera calibration, April 2022)

    # projection_matrix_2 = np.array(
    #     [
    #         [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
    #         [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
    #         [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
    #     ],
    #     dtype=float,
    # )

    # if camera =
    # projection_matrix_1 = np.array(
    #     [
    #         [1.52294360e03, -4.22479193e02, -3.15118817e02, 1.14524891e03],
    #         [9.04115233e01, 4.75516010e01, -1.46460484e03, 7.17379127e02],
    #         [7.22092000e-01, 5.97278000e-01, -3.49058000e-01, -1.50021000e00],
    #     ],
    #     dtype=float,
    # )

    def __init__(self, camera, lidar):
        self.projection_matrix_lidar = None
        self.projection_matrix_camera = None
        # self.projection_matrix = np.array(
        #     [
        #         [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
        #         [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
        #         [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
        #     ],
        #     dtype=float,
        # )

        # intrinsic_optimized = np.array(
        #     [[1253.45, 0, 957.907], [0, 1264.81, 628.129], [0, 0, 1]], dtype=float
        # )

        # extrinsic_s110_lidar_ouster_north = np.array(
        #     [
        #         [-0.374855, -0.926815, 0.0222604, -0.284537],
        #         [-0.465575, 0.167432, -0.869026, 0.683219],
        #         [0.8017, -0.336123, -0.494264, -0.837352],
        #     ],
        #     dtype=float,
        # )

        # if camera == "south2":
        #     self.projection_matrix = np.array(
        #         [
        #             [1.52294360e03, -4.22479193e02, -3.15118817e02, 1.14524891e03],
        #             [9.04115233e01, 4.75516010e01, -1.46460484e03, 7.17379127e02],
        #             [7.22092000e-01, 5.97278000e-01, -3.49058000e-01, -1.50021000e00],
        #         ],
        #         dtype=float,
        #     )

        #     intrinsic_optimized = np.array(
        #         [[1282.35, 0, 957.578], [0, 1334.48, 563.157], [0, 0, 1]], dtype=float
        #     )

        #     extrinsic_s110_lidar_ouster_north = np.array(
        #         [
        #             [0.37383, -0.927155, 0.0251845, 14.2181],
        #             [-0.302544, -0.147564, -0.941643, 3.50648],
        #             [0.876766, 0.344395, -0.335669, -7.26891],
        #         ],
        #         dtype=float,
        #     )

        # if lidar == "north":
        #     self.projection_matrix = np.dot(
        #         intrinsic_optimized,
        #         extrinsic_s110_lidar_ouster_north,
        #     )

        # self.transformation_matrix_s110_lidar_ouster_north_to_south = np.array(
        #     [
        #         [9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e00],
        #         [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e01],
        #         [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
        #         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        #     ],
        #     dtype=float,
        # )

    def hex_to_rgb(self, value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def draw_2d_box(self, img, box_label, color):
        cv2.rectangle(
            img, (box_label[0], box_label[1]), (box_label[2], box_label[3]), color, 1
        )

    # def project_lidar_north_to_south(self, point_cloud):
    #     points_3d = np.asarray(point_cloud.points)

    def project_lidar_to_image(self, image, point_cloud_south, point_cloud_north):
        # points_3d = np.asarray(point_cloud.points)
        points_3d_south = np.asarray(point_cloud_south.points)
        points_3d_north = np.asarray(point_cloud_north.points)
        # points_3d = np.concatenate(points_3d_south, points_3d_north)

        # remove rows having all zeros (131k points -> 59973 points)
        points_3d_south = points_3d_south[~np.all(points_3d_south == 0, axis=1)]
        points_3d_south = np.transpose(points_3d_south)
        points_3d_south = np.append(
            points_3d_south, np.ones((1, points_3d_south.shape[1])), axis=0
        )

        points_3d_north = points_3d_north[~np.all(points_3d_north == 0, axis=1)]
        points_3d_north = np.transpose(points_3d_north)
        points_3d_north = np.append(
            points_3d_north, np.ones((1, points_3d_north.shape[1])), axis=0
        )

        points_3d_north = np.matmul(
            self.transformation_matrix_s110_lidar_ouster_north_to_south, points_3d_north
        )
        points_3d = points_3d_north  # np.concatenate((points_3d_south, points_3d_north), axis=1)

        distances = []
        indices_to_keep = []
        for i in range(len(points_3d[0, :])):
            point = points_3d[:, i]
            distance = sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
            if distance > 2:
                distances.append(distance)
                indices_to_keep.append(i)

        points_3d = points_3d[:, indices_to_keep]

        # project points to 2D
        points = np.matmul(self.projection_matrix, points_3d[:4, :])
        distances_numpy = np.asarray(distances)
        max_distance = max(distances_numpy)
        norm = mpl.colors.Normalize(vmin=200, vmax=250)
        cmap = cm.gnuplot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for i in range(len(points[0, :])):
            if points[2, i] > 0:
                pos_x = int(points[0, i] / points[2, i])
                pos_y = int(points[1, i] / points[2, i])
                if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
                    distance_idx = 255 - (int(distances_numpy[i] / max_distance * 255))
                    color_rgba = m.to_rgba(distance_idx)
                    color_rgb = (
                        color_rgba[0] * 255,
                        color_rgba[1] * 255,
                        color_rgba[2] * 255,
                    )
                    cv2.circle(image, (pos_x, pos_y), 4, color_rgb, thickness=-1)
                    # print("pos_x: %f, pos_y: %f" % (pos_x, pos_y))
        return image

    def project_3d_to_2d(self, points_3d):
        points_3d = np.transpose(points_3d)
        points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)

        # project points to 2D
        points = np.matmul(self.projection_matrix, points_3d[:4, :])
        edge_points = []
        for i in range(len(points[0, :])):
            if points[2, i] > 0:
                pos_x = int((points[0, i] / points[2, i]))
                pos_y = int((points[1, i] / points[2, i]))
                # if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
                if pos_x < 1920 and pos_y < 1200:
                    edge_points.append((pos_x, pos_y))

        return edge_points

    def draw_3d_box(self, img, box_label, color):
        l = float(box_label["object_data"]["cuboid"]["val"][7])
        w = float(box_label["object_data"]["cuboid"]["val"][8])
        h = float(box_label["object_data"]["cuboid"]["val"][9])

        quat_x = float(box_label["object_data"]["cuboid"]["val"][3])
        quat_y = float(box_label["object_data"]["cuboid"]["val"][4])
        quat_z = float(box_label["object_data"]["cuboid"]["val"][5])
        quat_w = float(box_label["object_data"]["cuboid"]["val"][6])
        _, _, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler(
            "xyz", degrees=True
        )
        # rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('zyx', degrees=False)[0]

        center = [
            float(box_label["object_data"]["cuboid"]["val"][0]),
            float(box_label["object_data"]["cuboid"]["val"][1]),
            float(box_label["object_data"]["cuboid"]["val"][2] - h / 2),
        ]

        # points_3d = box_center_to_corner(
        #     center[0], center[1], center[2], l, w, h, np.radians(yaw)
        # )

        # points_2d = self.project_3d_to_2d(points_3d)

        # if len(points_2d) == 8:
        #     draw_line(img, points_2d[0], points_2d[1], color)
        #     draw_line(img, points_2d[1], points_2d[2], color)
        #     draw_line(img, points_2d[2], points_2d[3], color)
        #     draw_line(img, points_2d[3], points_2d[0], color)
        #     draw_line(img, points_2d[4], points_2d[5], color)
        #     draw_line(img, points_2d[5], points_2d[6], color)
        #     draw_line(img, points_2d[6], points_2d[7], color)
        #     draw_line(img, points_2d[7], points_2d[4], color)
        #     draw_line(img, points_2d[0], points_2d[4], color)
        #     draw_line(img, points_2d[1], points_2d[5], color)
        #     draw_line(img, points_2d[2], points_2d[6], color)
        #     draw_line(img, points_2d[3], points_2d[7], color)

        return (center, yaw, l, w, h)  # points_2d

    def draw_3d_box_camera_labels(self, img, box_label, color):
        l = float(box_label["object_data"]["cuboid"]["val"][7])
        w = float(box_label["object_data"]["cuboid"]["val"][8])
        h = float(box_label["object_data"]["cuboid"]["val"][9])

        quat_x = float(box_label["object_data"]["cuboid"]["val"][3])
        quat_y = float(box_label["object_data"]["cuboid"]["val"][4])
        quat_z = float(box_label["object_data"]["cuboid"]["val"][5])
        quat_w = float(box_label["object_data"]["cuboid"]["val"][6])
        _, _, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler(
            "xyz", degrees=True
        )
        # rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('zyx', degrees=False)[0]

        center = [
            float(box_label["object_data"]["cuboid"]["val"][0]),
            float(box_label["object_data"]["cuboid"]["val"][1]),
            float(box_label["object_data"]["cuboid"]["val"][2]),  # - h / 2),
        ]

        # points_3d = box_center_to_corner(
        #     center[0], center[1], center[2], l, w, h, np.radians(yaw)
        # )

        # points_2d = self.project_3d_to_2d(points_3d)

        # if len(points_2d) == 8:
        #     draw_line(img, points_2d[0], points_2d[1], color)
        #     draw_line(img, points_2d[1], points_2d[2], color)
        #     draw_line(img, points_2d[2], points_2d[3], color)
        #     draw_line(img, points_2d[3], points_2d[0], color)
        #     draw_line(img, points_2d[4], points_2d[5], color)
        #     draw_line(img, points_2d[5], points_2d[6], color)
        #     draw_line(img, points_2d[6], points_2d[7], color)
        #     draw_line(img, points_2d[7], points_2d[4], color)
        #     draw_line(img, points_2d[0], points_2d[4], color)
        #     draw_line(img, points_2d[1], points_2d[5], color)
        #     draw_line(img, points_2d[2], points_2d[6], color)
        #     draw_line(img, points_2d[3], points_2d[7], color)

        return (center, yaw, l, w, h)  # points_2d

    # def draw_3d_box_camera_labels(self, img, box_label, color):
    #     points2d_val = box_label["object_data"]["keypoints_2d"]["attributes"][
    #         "points2d"
    #     ]["val"]
    #     # point_bottom_left_front = (float(points2d_val[0]["val"][0]), float(points2d_val[0]["val"][1]))
    #     # point_bottom_left_back = (float(points2d_val[1]["val"][0]), float(points2d_val[1]["val"][1]))
    #     # point_bottom_right_back = (float(points2d_val[2]["val"][0]), float(points2d_val[2]["val"][1]))
    #     # point_bottom_right_front = (float(points2d_val[3]["val"][0]), float(points2d_val[3]["val"][1]))
    #     # point_top_left_front = (float(points2d_val[4]["val"][0]), float(points2d_val[4]["val"][1]))
    #     # point_top_left_back = (float(points2d_val[5]["val"][0]), float(points2d_val[5]["val"][1]))
    #     # point_top_right_back = (float(points2d_val[6]["val"][0]), float(points2d_val[6]["val"][1]))
    #     # point_top_right_front = (float(points2d_val[7]["val"][0]), float(points2d_val[7]["val"][1]))

    #     points_2d = []
    #     for point2d in points2d_val:
    #         points_2d.append(
    #             (int(point2d["point2d"]["val"][0]), int(point2d["point2d"]["val"][1]))
    #         )

    #     # if len(points_2d) == 8:
    #     #     draw_line(img, points_2d[0], points_2d[1], color)
    #     #     draw_line(img, points_2d[1], points_2d[2], color)
    #     #     draw_line(img, points_2d[2], points_2d[3], color)
    #     #     draw_line(img, points_2d[3], points_2d[0], color)
    #     #     draw_line(img, points_2d[4], points_2d[5], color)
    #     #     draw_line(img, points_2d[5], points_2d[6], color)
    #     #     draw_line(img, points_2d[6], points_2d[7], color)
    #     #     draw_line(img, points_2d[7], points_2d[4], color)
    #     #     draw_line(img, points_2d[0], points_2d[4], color)
    #     #     draw_line(img, points_2d[1], points_2d[5], color)
    #     #     draw_line(img, points_2d[2], points_2d[6], color)
    #     #     draw_line(img, points_2d[3], points_2d[7], color)

    #     return points_2d

    def bb_intersection_over_union(self, img, lidar_bbox, camera_bbox, color):
        # determine the (x, y)-coordinates of the intersection rectangle
        # xA = max(lidar_bbox[0], camera_bbox[0])
        # yA = max(lidar_bbox[1], camera_bbox[1])
        # xB = min(lidar_bbox[2], camera_bbox[2])
        # yB = min(lidar_bbox[3], camera_bbox[3])
        if len(lidar_bbox) < 8 or len(camera_bbox) < 8:
            return

        # if not check_intersection(
        #     np.array(lidar_bbox[:4]), np.array(camera_bbox[:4])
        # ) and not check_intersection(
        #     np.array(lidar_bbox[4:]), np.array(camera_bbox[4:])
        # ):
        #     return

        if not check_intersection(np.array(lidar_bbox), np.array(camera_bbox)):
            return

        # point_0 = (
        #     int(np.mean([lidar_bbox[0][0], camera_bbox[4][0]])),
        #     int(np.mean([lidar_bbox[0][1], camera_bbox[4][1]])),
        # )  # (max(lidar_bbox[0][0], camera_bbox[0][0]), max(lidar_bbox[0][1], camera_bbox[0][1]))
        # point_1 = (
        #     int(np.mean([lidar_bbox[1][0], camera_bbox[7][0]])),
        #     int(np.mean([lidar_bbox[1][1], camera_bbox[7][1]])),
        # )  # (max(lidar_bbox[1][0], camera_bbox[1][0]), max(lidar_bbox[1][1], camera_bbox[1][1]))
        # point_2 = (
        #     int(np.mean([lidar_bbox[2][0], camera_bbox[3][0]])),
        #     int(np.mean([lidar_bbox[2][1], camera_bbox[3][1]])),
        # )  # (min(lidar_bbox[2][0], camera_bbox[2][0]), min(lidar_bbox[2][1], camera_bbox[2][1]))
        # point_3 = (
        #     int(np.mean([lidar_bbox[3][0], camera_bbox[0][0]])),
        #     int(np.mean([lidar_bbox[3][1], camera_bbox[0][1]])),
        # )  # (min(lidar_bbox[3][0], camera_bbox[3][0]), min(lidar_bbox[3][1], camera_bbox[3][1]))
        # point_4 = (
        #     int(np.mean([lidar_bbox[4][0], camera_bbox[5][0]])),
        #     int(np.mean([lidar_bbox[4][1], camera_bbox[5][1]])),
        # )  # (max(lidar_bbox[4][0], camera_bbox[4][0]), max(lidar_bbox[4][1], camera_bbox[4][1]))
        # point_5 = (
        #     int(np.mean([lidar_bbox[5][0], camera_bbox[6][0]])),
        #     int(np.mean([lidar_bbox[5][1], camera_bbox[6][1]])),
        # )  # (max(lidar_bbox[5][0], camera_bbox[5][0]), max(lidar_bbox[5][1], camera_bbox[5][1]))
        # point_6 = (
        #     int(np.mean([lidar_bbox[6][0], camera_bbox[2][0]])),
        #     int(np.mean([lidar_bbox[6][1], camera_bbox[2][1]])),
        # )  # (min(lidar_bbox[6][0], camera_bbox[6][0]), min(lidar_bbox[6][1], camera_bbox[6][1]))
        # point_7 = (
        #     int(np.mean([lidar_bbox[7][0], camera_bbox[1][0]])),
        #     int(np.mean([lidar_bbox[7][1], camera_bbox[1][1]])),
        # )  # (min(lidar_bbox[7][0], camera_bbox[7][0]), min(lidar_bbox[7][1], camera_bbox[7][1]))

        point_0 = (
            int(np.mean([lidar_bbox[0][0], camera_bbox[2][0]])),
            int(np.mean([lidar_bbox[0][1], camera_bbox[2][1]])),
        )
        point_1 = (
            int(np.mean([lidar_bbox[1][0], camera_bbox[6][0]])),
            int(np.mean([lidar_bbox[1][1], camera_bbox[6][1]])),
        )
        point_2 = (
            int(np.mean([lidar_bbox[2][0], camera_bbox[7][0]])),
            int(np.mean([lidar_bbox[2][1], camera_bbox[7][1]])),
        )
        point_3 = (
            int(np.mean([lidar_bbox[3][0], camera_bbox[3][0]])),
            int(np.mean([lidar_bbox[3][1], camera_bbox[3][1]])),
        )
        point_4 = (
            int(np.mean([lidar_bbox[4][0], camera_bbox[1][0]])),
            int(np.mean([lidar_bbox[4][1], camera_bbox[1][1]])),
        )
        point_5 = (
            int(np.mean([lidar_bbox[5][0], camera_bbox[5][0]])),
            int(np.mean([lidar_bbox[5][1], camera_bbox[5][1]])),
        )
        point_6 = (
            int(np.mean([lidar_bbox[6][0], camera_bbox[4][0]])),
            int(np.mean([lidar_bbox[6][1], camera_bbox[4][1]])),
        )
        point_7 = (
            int(np.mean([lidar_bbox[7][0], camera_bbox[0][0]])),
            int(np.mean([lidar_bbox[7][1], camera_bbox[0][1]])),
        )

        points_2d = [
            point_0,
            point_1,
            point_2,
            point_3,
            point_4,
            point_5,
            point_6,
            point_7,
        ]

        # compute the area of intersection rectangle
        # interArea = abs(max((xB - xA), 0) * max((yB - yA), 0))
        # interArea = abs(
        #     max((point_2[0] - point_0[0]) * (point_2[1] - point_0[1]), 0)
        #     * max((point_3[0] - point_1[0]) * (point_3[1] - point_1[1]), 0)
        #     # * np.multiply(np.subtract(point_4, point_0))
        # )
        # if interArea == 0:
        #     return 0
        # # compute the area of both the prediction and ground-truth
        # # rectangles
        # lidar_bboxArea = abs((lidar_bbox[2] - lidar_bbox[0]) * (lidar_bbox[3] - lidar_bbox[1]))
        # camera_bboxArea = abs((camera_bbox[2] - camera_bbox[0]) * (camera_bbox[3] - camera_bbox[1]))

        # # compute the intersection over union by taking the intersection
        # # area and dividing it by the sum of prediction + ground-truth
        # # areas - the interesection area
        # iou = interArea / float(lidar_bboxArea + camera_bboxArea - interArea)

        if len(points_2d) == 8:
            draw_line(img, points_2d[0], points_2d[1], color)
            draw_line(img, points_2d[1], points_2d[2], color)
            draw_line(img, points_2d[2], points_2d[3], color)
            draw_line(img, points_2d[3], points_2d[0], color)
            draw_line(img, points_2d[4], points_2d[5], color)
            draw_line(img, points_2d[5], points_2d[6], color)
            draw_line(img, points_2d[6], points_2d[7], color)
            draw_line(img, points_2d[7], points_2d[4], color)
            draw_line(img, points_2d[0], points_2d[4], color)
            draw_line(img, points_2d[1], points_2d[5], color)
            draw_line(img, points_2d[2], points_2d[6], color)
            draw_line(img, points_2d[3], points_2d[7], color)

        # return the intersection over union value
        # return iou

    def draw_multi_modal_centers(self, img, lidar_bbox, camera_bbox):
        lidar_center = 0
        camera_center = 0
        if lidar_bbox is not None:
            lidar_center = (
                int(sum(corner[0] for corner in lidar_bbox) / 8),
                int(sum(corner[1] for corner in lidar_bbox) / 8),
            )
        if camera_bbox is not None:
            camera_center = (
                int(sum(corner[0] for corner in camera_bbox) / 8),
                int(sum(corner[1] for corner in camera_bbox) / 8),
            )

        # cv2.circle(img, lidar_center, radius=1, color=(0, 0, 255), thickness=2)
        # cv2.circle(img, camera_center, radius=1, color=(255, 0, 0), thickness=2)

        return lidar_center, camera_center

    def assign_detections_to_trackers(
            self, img, lidar_det, camera_det, iou_thrd=3, option="mixed"
    ):  # 0.3):
        """
        From current list of lidar_det and new camera_det, output matched camera_det,
        unmatchted lidar_det, unmatched camera_det.
        """

        lidar_detection_centers = []
        for det in lidar_det:
            lidar_detection_centers.append(det[0])

        monocular_detection_centers = []
        for det in camera_det:
            monocular_detection_centers.append(det[0])

        IOU_mat = np.zeros((len(lidar_detection_centers), len(monocular_detection_centers)), dtype=np.float32)
        for t, trk in enumerate(lidar_detection_centers):
            for d, det in enumerate(monocular_detection_centers):
                dist = self.center_dist(trk, det)
                if dist > iou_thrd:
                    dist = 99999
                IOU_mat[t, d] = dist

        # Produces matches
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)

        matched_idx_0, matched_idx_1 = linear_assignment(IOU_mat)  # (-IOU_mat)
        matched_idx = np.column_stack((matched_idx_0, matched_idx_1))

        unmatched_lidar, unmatched_monocular = [], []
        for t, trk in enumerate(lidar_det):
            if t not in matched_idx[:, 0]:
                unmatched_lidar.append(t)

        for d, det in enumerate(camera_det):
            if d not in matched_idx[:, 1]:
                unmatched_monocular.append(d)

        matches = []
        for m in matched_idx:
            if IOU_mat[m[0], m[1]] > iou_thrd:
                unmatched_lidar.append(m[0])
                unmatched_monocular.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        # for idx_0 in unmatched_lidar:
        #     center, yaw, l, w, h = lidar_det[idx_0]

        #     points_3d = box_center_to_corner(
        #         center[0], center[1], center[2], l, w, h, np.radians(yaw)
        #     )

        #     points_2d = self.project_3d_to_2d(points_3d)

        #     color_bgr = (235, 192, 52)  # blue
        #     self.draw_box(img, points_2d, color_bgr)

        for idx_1 in unmatched_monocular:
            center, yaw, l, w, h = camera_det[idx_1]

            points_3d = box_center_to_corner(
                center[0], center[1], center[2], l, w, h, np.radians(yaw)
            )

            points_2d = self.project_3d_to_2d(points_3d)

            color_bgr = (96, 255, 96)  # green
            self.draw_box(img, points_2d, color_bgr)

        if option == "mixed":
            for idx_0, idx_1 in matches:
                center = camera_det[idx_1][0]
                yaw = camera_det[idx_1][1]
                l = lidar_det[idx_0][2]
                w = lidar_det[idx_0][3]
                h = lidar_det[idx_0][4]

                points_3d = box_center_to_corner(
                    center[0], center[1], center[2], l, w, h, np.radians(yaw)
                )

                points_2d = self.project_3d_to_2d(points_3d)

                color_bgr = (96, 96, 255)  # red
                self.draw_box(img, points_2d, color_bgr)

            for idx_0, _ in matches:
                center, yaw, l, w, h = lidar_det[idx_0]

                points_3d = box_center_to_corner(
                    center[0], center[1], center[2], l, w, h, np.radians(yaw)
                )

                points_2d = self.project_3d_to_2d(points_3d)

                color_bgr = (235, 192, 52)  # blue
                self.draw_box(img, points_2d, color_bgr)

            for _, idx_1 in matches:
                center, yaw, l, w, h = camera_det[idx_1]

                points_3d = box_center_to_corner(
                    center[0], center[1], center[2], l, w, h, np.radians(yaw)
                )

                points_2d = self.project_3d_to_2d(points_3d)

                color_bgr = (96, 255, 96)  # green
                self.draw_box(img, points_2d, color_bgr)

        if option == "double_hungarian":
            for idx_0, idx_1 in matches:
                lidar_center, camera_center = self.draw_multi_modal_centers(
                    img, lidar_det[idx_0], camera_det[idx_1]
                )
                mean_center = (
                    int(np.mean([lidar_center[0], camera_center[0]])),
                    int(np.mean([lidar_center[1], camera_center[1]])),
                )

                cv2.circle(
                    img,
                    mean_center,
                    radius=1,
                    color=(0, 0, 255),
                    thickness=2,
                )

                lidar_change = np.subtract(lidar_center, camera_center)
                for i, corner in enumerate(lidar_det[idx_0]):
                    lidar_det[idx_0][i] = np.subtract(corner, lidar_change)
                camera_change = np.subtract(camera_center, mean_center)
                for i, corner in enumerate(camera_det[idx_1]):
                    camera_det[idx_1][i] = corner - camera_change
                # corner_points = box_center_to_corner(
                #     mean_center[0],
                #     mean_center[1],
                #     center_z[idx_0],
                #     l[idx_0],
                #     w[idx_0],
                #     h[idx_0],
                #     np.radians(yaw[idx_0]),
                # )

                # points_2d = self.project_3d_to_2d(corner_points)
                # # print(points_2d)

                # if len(points_2d) == 8:

                # for t, trk in enumerate(lidar_det[idx_0]):
                #     for d, det in enumerate(camera_det[idx_1]):
                if lidar_det[idx_0]:
                    IOU_mat = np.zeros(
                        (len(lidar_det[idx_0]), len(camera_det[idx_1])), dtype=np.float32
                    )
                    IOU_mat = distance.cdist(lidar_det[idx_0], camera_det[idx_1])

                    # Produces matches
                    # Solve the maximizing the sum of IOU assignment problem using the
                    # Hungarian algorithm (also known as Munkres algorithm)

                    matched_idx_0, matched_idx_1 = linear_assignment(IOU_mat)  # (-IOU_mat)
                    matched_idx = np.column_stack((matched_idx_0, matched_idx_1))

                    corner_matches = []

                    # For creating lidar_det we consider any detection with an
                    # overlap less than iou_thrd to signifiy the existence of
                    # an untracked object

                    for m in matched_idx:
                        corner_matches.append(m.reshape(1, 2))

                    if len(corner_matches) == 0:
                        corner_matches = np.empty((0, 2), dtype=int)
                    else:
                        corner_matches = np.concatenate(corner_matches, axis=0)

                    points_2d = []
                    for corner_idx_0, corner_idx_1 in corner_matches:
                        corner = (
                            int(
                                np.mean(
                                    [
                                        lidar_det[idx_0][corner_idx_0][0],
                                        camera_det[idx_1][corner_idx_1][0],
                                    ]
                                )
                            ),
                            int(
                                np.mean(
                                    [
                                        lidar_det[idx_0][corner_idx_0][1],
                                        camera_det[idx_1][corner_idx_1][1],
                                    ]
                                )
                            ),
                        )
                        points_2d.append(corner)

                    color = (0, 0, 255)
                    # points_2d = lidar_det[idx_0]
                    if len(points_2d) == 8:
                        draw_line(img, points_2d[0], points_2d[1], color)
                        draw_line(img, points_2d[1], points_2d[2], color)
                        draw_line(img, points_2d[2], points_2d[3], color)
                        draw_line(img, points_2d[3], points_2d[0], color)
                        draw_line(img, points_2d[4], points_2d[5], color)
                        draw_line(img, points_2d[5], points_2d[6], color)
                        draw_line(img, points_2d[6], points_2d[7], color)
                        draw_line(img, points_2d[7], points_2d[4], color)
                        draw_line(img, points_2d[0], points_2d[4], color)
                        draw_line(img, points_2d[1], points_2d[5], color)
                        draw_line(img, points_2d[2], points_2d[6], color)
                        draw_line(img, points_2d[3], points_2d[7], color)

                # color = (0, 255, 0)
                # points_2d = camera_det[idx_1]
                # if len(points_2d) == 8:
                #     draw_line(img, points_2d[0], points_2d[1], color)
                #     draw_line(img, points_2d[1], points_2d[2], color)
                #     draw_line(img, points_2d[2], points_2d[3], color)
                #     draw_line(img, points_2d[3], points_2d[0], color)
                #     draw_line(img, points_2d[4], points_2d[5], color)
                #     draw_line(img, points_2d[5], points_2d[6], color)
                #     draw_line(img, points_2d[6], points_2d[7], color)
                #     draw_line(img, points_2d[7], points_2d[4], color)
                #     draw_line(img, points_2d[0], points_2d[4], color)
                #     draw_line(img, points_2d[1], points_2d[5], color)
                #     draw_line(img, points_2d[2], points_2d[6], color)
                #     draw_line(img, points_2d[3], points_2d[7], color)

            for idx_0 in unmatched_lidar:
                lidar_center, _ = self.draw_multi_modal_centers(
                    img, lidar_det[idx_0], camera_det[0]
                )
                cv2.circle(
                    img,
                    (lidar_center[0], lidar_center[1]),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=2,
                )

            for idx_1 in unmatched_monocular:
                _, camera_center = self.draw_multi_modal_centers(
                    img, lidar_det[0], camera_det[idx_1]
                )
                cv2.circle(
                    img,
                    (camera_center[0], camera_center[1]),
                    radius=1,
                    color=(255, 0, 0),
                    thickness=2,
                )

            return matches  # , np.array(unmatched_monocular), np.array(unmatched_lidar)

    def draw_box(self, img, points_2d, color):
        if len(points_2d) == 8:
            draw_line(img, points_2d[0], points_2d[1], color)
            draw_line(img, points_2d[1], points_2d[2], color)
            draw_line(img, points_2d[2], points_2d[3], color)
            draw_line(img, points_2d[3], points_2d[0], color)
            draw_line(img, points_2d[4], points_2d[5], color)
            draw_line(img, points_2d[5], points_2d[6], color)
            draw_line(img, points_2d[6], points_2d[7], color)
            draw_line(img, points_2d[7], points_2d[4], color)
            draw_line(img, points_2d[0], points_2d[4], color)
            draw_line(img, points_2d[1], points_2d[5], color)
            draw_line(img, points_2d[2], points_2d[6], color)
            draw_line(img, points_2d[3], points_2d[7], color)

    # def assign_detections_to_trackers(
    #     self, img, lidar_track, camera_det, iou_thrd=300
    # ):  # 0.3):
    #     """
    #     From current list of lidar_det and new camera_det, output matched camera_det,
    #     unmatchted lidar_det, unmatched camera_det.
    #     """
    #     lidar_det = []
    #     l = []
    #     w = []
    #     h = []
    #     yaw = []
    #     center_z = []
    #     for lidar in lidar_track:
    #         lidar_center, _ = self.draw_multi_modal_centers(
    #             img=None, lidar_bbox=lidar[0], camera_bbox=None
    #         )
    #         lidar_det.append(np.array(lidar_center))
    #         l.append(lidar[1])
    #         w.append(lidar[2])
    #         h.append(lidar[3])
    #         yaw.append(lidar[4])
    #         center_z.append(lidar[5])

    #     kdt = KDTree(lidar_det)
    #     detection_centers = []
    #     for det in camera_det:
    #         _, camera_center = self.draw_multi_modal_centers(
    #             img=None, lidar_bbox=None, camera_bbox=det
    #         )
    #         detection_centers.append(np.array(camera_center))
    #     dist, indices = kdt.query(detection_centers, k=1, distance_upper_bound=500)

    #     # Produces matches
    #     # Solve the maximizing the sum of IOU assignment problem using the
    #     # Hungarian algorithm (also known as Munkres algorithm)

    #     matched_idx_0, matched_idx_1 = indices  # (-IOU_mat)

    #     matches = []

    #     for idx_0, idx_1 in matches:
    #         lidar_center, camera_center = self.draw_multi_modal_centers(
    #             img, lidar_det[idx_0], camera_det[idx_1]
    #         )
    #         mean_center = (
    #             int(np.mean([lidar_center[0], camera_center[0]])),
    #             int(np.mean([lidar_center[1], camera_center[1]])),
    #         )

    #         cv2.circle(
    #             img,
    #             mean_center,
    #             radius=1,
    #             color=(0, 0, 255),
    #             thickness=2,
    #         )

    #     return matches  # , np.array(unmatched_monocular), np.array(unmatched_lidar)

    def center_dist(self, lidar_center, camera_center):
        dist = sqrt(
            (lidar_center[0] - camera_center[0]) ** 2
            + (lidar_center[1] - camera_center[1]) ** 2
        )
        return dist

    # def center_dist(self, lidar_bbox, camera_bbox):
    #     lidar_center, camera_center = self.draw_multi_modal_centers(
    #         img=None, lidar_bbox=lidar_bbox, camera_bbox=camera_bbox
    #     )
    #     dist = sqrt(
    #         (lidar_center[0] - camera_center[0]) ** 2
    #         + (lidar_center[1] - camera_center[1]) ** 2
    #     )
    #     return dist


def get_color_by_category(category):
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
    elif category == "BICYCLE":
        color = "#B18CFF"
    elif category == "SPECIAL_VEHICLE" or category == "EMERGENCY VEHICLE":
        color = "#666bfa"
    elif category == "OTHER" or category == "OTHER_VEHICLES":
        # NOTE: r00_s00 contains the class "Other Vehicles", whereas r00_s01 - r00_s04 contain the class "OTHER"
        color = "#C7C7C7"
    elif category == "LICENSE_PLATE_LOCATION":
        color = "#000000"
    else:
        print("Unknown category: ", category)
    return color


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    # argparser.add_argument(
    #     "-i",
    #     "--image_file_path",
    #     default="image.jpg",
    #     help="Image file path. Default: image.jpg",
    # )
    # argparser.add_argument(
    #     "-l",
    #     "--label_file_path",
    #     default="label.json",
    #     help="Label file path. Default: label.json",
    # )
    # argparser.add_argument(
    #     "-o",
    #     "--output_file_path",
    #     help="Output file path to save visualization result to disk.",
    # )
    # argparser.add_argument(
    #     "-m",
    #     "--viz_mode",
    #     default="box3d_projected",
    #     help="Visualization mode. Available modes are: [box2d, box3d_projected, box2d_and_box3d_projected]",
    # )
    # args = argparser.parse_args()

    # root_path = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/"
    # image_file_path = root_path + "04_R1_S4/03_b_images_anonymized_undistorted/s110_camera_basler_south2_8mm/1646667310_055996268_s110_camera_basler_south2_8mm.png"  # args.image_file_path
    # label_file_path = root_path + "04_R1_S4/05_labels_openlabel/s110_lidar_ouster_south/1646667310_053239541_s110_lidar_ouster_south.json"  # args.label_file_path
    # output_file_path = "/home/walter/Pictures/result.png"

    scenario = "s04"
    camera = "south1"
    lidar = "north"
    if camera == "south2":
        lidar = "south"

    root_path = "/home/stefan/Documents/master/"
    image_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/images/s110_camera_basler_{camera}_8mm/"
    )  # 1646667310_055996268_s110_camera_basler_south2_8mm.png"  # args.image_file_path
    label_lidar_south_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/labels/s110_lidar_ouster_south/"
    )  # 1646667310_053239541_s110_lidar_ouster_south.json"  # args.label_file_path
    point_cloud_south_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/point_clouds/s110_lidar_ouster_south/"
    )
    label_lidar_north_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/labels/s110_lidar_ouster_north/"
    )  # 1646667310_053239541_s110_lidar_ouster_south.json"  # args.label_file_path
    point_cloud_north_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/point_clouds/s110_lidar_ouster_north/"
    )
    lidar_label_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/labels/s110_lidar_ouster_{lidar}/"
    )

    camera_label_dir_path = (
            root_path
            + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/labels/s110_camera_basler_{camera}_8mm/"
    )
    output_dir_path = root_path + f"pictures/video_frames/r01_{scenario}_{camera}_what_123/"
    pathlib.Path(f"{output_dir_path}").mkdir(parents=True, exist_ok=True)

    lidar_north_detections_dir = root_path + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/lidar3d_det/openlabel/"
    lidar_south_detections_dir = root_path + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/lidar3d_det/openlabel/"
    camera_detections_dir = root_path + f"00_a9_dataset/a9_r1_dataset/r01_{scenario}/mono3d_det/"
    # viz_mode = args.viz_mode

    for frame, image_file_path in enumerate(sorted(os.listdir(image_dir_path))):
        if True:
            img = cv2.imread(image_dir_path + image_file_path, cv2.IMREAD_UNCHANGED)
            utils = Utils(camera, lidar)

            # point_cloud_south_path = sorted(os.listdir(point_cloud_south_dir_path))[
            #     frame
            # ]
            # point_cloud_south = o3d.io.read_point_cloud(
            #     point_cloud_south_dir_path + point_cloud_south_path
            # )
            # point_cloud_north_path = sorted(os.listdir(point_cloud_north_dir_path))[
            #     frame
            # ]
            # point_cloud_north = o3d.io.read_point_cloud(
            #     point_cloud_north_dir_path + point_cloud_north_path
            # )
            # img = utils.project_lidar_to_image(img, point_cloud_south, point_cloud_north)

            # lidar_label_file_path = sorted(os.listdir(lidar_label_dir_path))[frame]
            # camera_label_file_path = sorted(os.listdir(camera_label_dir_path))[frame]

            # lidar_data = open(
            #     lidar_label_dir_path + lidar_label_file_path,
            # )
            # camera_data = open(
            #     camera_label_dir_path + camera_label_file_path,
            # )

            # lidar_labels = json.load(lidar_data)
            # camera_labels = json.load(camera_data)

            lidar_det_file_path = sorted(os.listdir(lidar_north_detections_dir))[frame]
            camera_det_file_path = sorted(os.listdir(camera_detections_dir))[frame]

            lidar_data = open(
                lidar_north_detections_dir + lidar_det_file_path,
            )
            camera_data = open(
                camera_detections_dir + camera_det_file_path,
            )

            lidar_dets = json.load(lidar_data)
            camera_dets = json.load(camera_data)

            # if "openlabel" in lidar_labels:
            # for lidar_label in lidar_labels["frames"]["objects"].values():
            #     category = lidar_label["object_data"]["type"]
            #     color = get_color_by_category(category)

            #     color_rgb = utils.hex_to_rgb(color)
            #     # swap channels because opencv uses bgr
            #     color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            #     lidar_bbox = utils.draw_3d_box(img, lidar_label, color_bgr)

            # for camera_label in camera_labels["frames"]["objects"].values():
            #     category = camera_label["object_data"]["type"]
            #     color = get_color_by_category(category)

            #     color_rgb = utils.hex_to_rgb(color)
            #     # swap channels because opencv uses bgr
            #     color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            #     camera_bbox = utils.draw_3d_box_camera_labels(
            #         img, camera_label, color_bgr
            #     )

            # for lidar_label in lidar_labels["frames"]["objects"].values():
            #     category = lidar_label["object_data"]["type"]
            #     color = get_color_by_category(category)

            #     color_rgb = utils.hex_to_rgb(color)
            #     # swap channels because opencv uses bgr
            #     color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            #     color_bgr = (0, 0, 255)
            #     lidar_bbox = utils.draw_3d_box(img, lidar_label, color_bgr)
            #     for camera_frame in camera_labels["openlabel"]["frames"].values():
            #         for camera_label in camera_frame["objects"].values():
            #             category = camera_label["object_data"]["type"]
            #             color = get_color_by_category(category)

            #             color_rgb = utils.hex_to_rgb(color)
            #             # swap channels because opencv uses bgr
            #             color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            #             color_bgr = (0, 255, 255)
            #             camera_bbox = utils.draw_3d_box_camera_labels(
            #                 img, camera_label, color_bgr
            #             )
            #             # color_bgr = (0, 255, 0)
            #             # utils.bb_intersection_over_union(
            #             #     img, lidar_bbox, camera_bbox, color_bgr
            #             # )
            #             utils.draw_multi_modal_centers(img, lidar_bbox, camera_bbox)

            utils.projection_matrix_lidar = np.array([
                [
                    1599.6787257016188,
                    391.55387236603775,
                    -430.34650625835917,
                    6400.522155319611
                ],
                [
                    -21.862527625533737,
                    -135.38146150648188,
                    -1512.651893582593,
                    13030.4682633739
                ],
                [
                    0.27397972486181504,
                    0.842440925400074,
                    -0.4639271468406554,
                    4.047780978836272
                ]
            ], dtype=np.float32)

            utils.projection_matrix_camera = np.array([
                [
                    7.04216073e02,
                    -1.37317442e03,
                    -4.32235765e02,
                    -2.03369364e04
                ],
                [
                    -9.28351327e01,
                    -1.77543929e01,
                    -1.45629177e03,
                    9.80290034e02
                ],
                [
                    8.71736000e-01,
                    -9.03453000e-02,
                    -4.81574000e-01,
                    -2.58546000e00
                ]
            ], dtype=np.float32)

            lidar_bbox = []
            for id, lidar_label in lidar_dets["openlabel"]["frames"]["0"]["objects"].items():
                category = lidar_label["object_data"]["type"]
                color = get_color_by_category(category)

                color_rgb = utils.hex_to_rgb(color)
                # swap channels because opencv uses bgr
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                color_bgr = (0, 0, 255)
                lidar_bbox.append(utils.draw_3d_box(img, lidar_label, color_bgr))

            # lidar_bbox = []
            # for id, lidar_label in lidar_labels["openlabel"]["frames"]["objects"].items():
            #     category = lidar_label["object_data"]["type"]
            #     color = get_color_by_category(category)

            #     color_rgb = utils.hex_to_rgb(color)
            #     # swap channels because opencv uses bgr
            #     color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            #     color_bgr = (0, 0, 255)
            #     lidar_bbox.append(utils.draw_3d_box(img, lidar_label, color_bgr))

            camera_bbox = []
            for id, camera_label in camera_dets["openlabel"]["frames"]["0"]["objects"].items():
                category = camera_label["object_data"]["type"]
                color = get_color_by_category(category)

                color_rgb = utils.hex_to_rgb(color)
                # swap channels because opencv uses bgr
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                color_bgr = (0, 255, 255)
                camera_bbox.append(utils.draw_3d_box_camera_labels(img, camera_label, color_bgr))

                # color_bgr = (0, 255, 0)
                # utils.bb_intersection_over_union(
                #     img, lidar_bbox, camera_bbox, color_bgr
                # )
            # utils.draw_multi_modal_centers(img, lidar_bbox, camera_bbox)
            utils.assign_detections_to_trackers(img, lidar_bbox, camera_bbox)

        if output_dir_path:
            cv2.imwrite(output_dir_path + f"frame_{frame}.png", img)
        else:
            cv2.imshow("image", img)
            cv2.waitKey()
