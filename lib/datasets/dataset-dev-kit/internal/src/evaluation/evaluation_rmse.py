import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path

from scipy.optimize import linear_sum_assignment

# TODO: use argparse to parse arguments
# TODO: remove hard coded paths
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--input_folder_path_images",
        default="images",
        help="Input folder path of images. Default: images",
    )
    argparser.add_argument(
        "--input_folder_path_labels",
        help="Input folder path to labels. Default: labels",
    )
    argparser.add_argument(
        "--input_folder_path_detection_processing",
        help="Input folder path to labels. Default: labels",
    )
    argparser.add_argument(
        "--input_folder_path_mono3d",
        help="Input folder path mono3d detections. Default: mono3d",
    )
    argparser.add_argument(
        "--output_folder_path_images",
        help="Output folder path of visualization. Default: output",
    )

    args = argparser.parse_args()
    input_folder_path_images = args.input_folder_path_images
    input_folder_path_labels = args.input_folder_path_labels
    input_folder_path_detection_processing = args.input_folder_path_detection_processing
    input_folder_path_mono3d = args.input_folder_path_mono3d
    output_folder_path_images = args.output_folder_path_images

    if not os.path.exists(output_folder_path_images):
        Path(output_folder_path_images).mkdir(parents=True, exist_ok=True)

    input_files_detection_processing = sorted(os.listdir(input_folder_path_detection_processing))
    input_files_mono3d = sorted(os.listdir(input_folder_path_mono3d))
    input_image_file_names = sorted(os.listdir(input_folder_path_images))
    input_label_file_names = sorted(os.listdir(input_folder_path_labels))

    error_total = 0

    # TODO: load projection matrix from file (a9 dataset)
    # projection matrix s110_lidar_ouster_south to s110_camera_basler_south1_8mm
    # projection_matrix = np.array([
    #     [
    #         1599.6787257016188,
    #         391.55387236603775,
    #         -430.34650625835917,
    #         6400.522155319611
    #     ],
    #     [
    #         -21.862527625533737,
    #         -135.38146150648188,
    #         -1512.651893582593,
    #         13030.4682633739
    #     ],
    #     [
    #         0.27397972486181504,
    #         0.842440925400074,
    #         -0.4639271468406554,
    #         4.047780978836272
    #     ]
    # ], dtype=float)
    # projection matrix s110_base to s110_camera_basler_south1_8mm
    projection_matrix = np.array(
        [
            [1599.6787257016188, 391.55387236603775, -430.34650625835917, 6400.522155319611],
            [-21.862527625533737, -135.38146150648188, -1512.651893582593, 13030.4682633739],
            [0.27397972486181504, 0.842440925400074, -0.4639271468406554, 4.047780978836272],
        ],
        dtype=float,
    )


    def get_3d_positions(json_obj):
        positions_3d_detection_processing = []
        for frame_id, frame_obj in json_obj["openlabel"]["frames"].items():
            for object_track_id, object_json in frame_obj["objects"].items():
                object_data = object_json["object_data"]
                if np.all(np.array(object_data["cuboid"]["val"]) == 0):
                    continue
                else:
                    cuboid = np.array(object_data["cuboid"]["val"])
                    positions_3d_detection_processing.append(cuboid[:3])
        return positions_3d_detection_processing


    def get_closest_3d_position_by_distance(positions_3d_detection_processing, position_3d_mono3d):
        distances = []
        for position_3d_detection_processing in positions_3d_detection_processing:
            distance = np.sqrt(
                (position_3d_mono3d[0] - position_3d_detection_processing[0]) ** 2
                + (position_3d_mono3d[1] - position_3d_detection_processing[1]) ** 2
            )
            distances.append(distance)
        if len(distances) > 0:
            idx_closest = np.argmin(np.array(distances))
            return positions_3d_detection_processing[idx_closest], distances[idx_closest]
        else:
            return None, -1


    def project(position_3d, projection_matrix):
        position_3d_homogeneous = np.array([position_3d[0], position_3d[1], position_3d[2], 1])
        points = np.matmul(projection_matrix, position_3d_homogeneous)
        if points[2] > 0:
            pos_x = int(points[0] / points[2])
            pos_y = int(points[1] / points[2])
            # if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
            return [pos_x, pos_y]
        return None


    def center_to_center_dist(position_3d_detection_processing, position_3d_mono3d):
        distance = np.sqrt(
            (position_3d_mono3d[0] - position_3d_detection_processing[0]) ** 2
            + (position_3d_mono3d[1] - position_3d_detection_processing[1]) ** 2
        )
        return distance


    def assignment(positions_3d_detection_processing, positions_3d_mono3d, distance_threshold):
        iou_dst = np.zeros((len(positions_3d_detection_processing), len(positions_3d_mono3d)))
        for id_mono3d, position_3d_mono3d in enumerate(positions_3d_mono3d):
            for id_detection_processing, position_3d_detection_processing in enumerate(
                    positions_3d_detection_processing
            ):
                distance = center_to_center_dist(position_3d_detection_processing, position_3d_mono3d)
                if distance > distance_threshold:
                    distance = 999999
                iou_dst[id_detection_processing, id_mono3d] = distance

        # matched_idx = linear_sum_assignment(iou_dst)
        matched_idx_detection_processing, matched_index_mono3d = linear_sum_assignment(iou_dst)
        matched_idx = np.column_stack((matched_idx_detection_processing, matched_index_mono3d))

        unmatched_detection_processing, unmatched_mono3d = [], []
        for id_detection_processing, position_3d_detection_processing in enumerate(positions_3d_detection_processing):
            if id_detection_processing not in matched_idx[:, 0]:
                unmatched_detection_processing.append(id_detection_processing)

        for id_mono3d, position_3d_mono3d in enumerate(positions_3d_mono3d):
            if id_mono3d not in matched_idx[:, 1]:
                unmatched_mono3d.append(id_mono3d)

        matches = []
        for idx in matched_idx:
            if iou_dst[idx[0], idx[1]] > distance_threshold:
                unmatched_detection_processing.append(idx[0])
                unmatched_mono3d.append(idx[1])
            else:
                matches.append(idx.reshape(1, 2))
        return unmatched_detection_processing, unmatched_mono3d, matches


    def match_positions(
            positions_3d_predictions, positions_3d_label, stats: np.ndarray, color, distance_threshold, visualization
    ):
        # assignment: match 3d positions from detection_processing/mono3d with labels
        unmatched_prediction_ids, unmatched_label_ids, matches_ids = assignment(
            positions_3d_predictions, positions_3d_label, distance_threshold
        )

        for matches in matches_ids:
            match_id_predictions = matches[0, 0]
            match_id_label = matches[0, 1]

            position_3d_prediction = positions_3d_predictions[match_id_predictions]
            position_3d_label = positions_3d_label[match_id_label]
            distance = center_to_center_dist(position_3d_prediction, position_3d_label)
            stats[0] += distance
            stats[1] += 1

            if visualization:
                # draw residual (green line)
                position_2d_prediction = project(position_3d_prediction, projection_matrix)
                position_2d_label = project(position_3d_label, projection_matrix)
                cv2.line(img, position_2d_label, position_2d_prediction, (96, 255, 96), 4)

                # predictions
                cv2.circle(img, position_2d_prediction, 5, color, thickness=-1)
                cv2.circle(img, position_2d_prediction, 6, (0, 0, 0), thickness=2)
                # labels
                cv2.circle(img, position_2d_label, 5, (96, 255, 96), thickness=-1)
                cv2.circle(img, position_2d_label, 6, (0, 0, 0), thickness=2)


    if __name__ == "__main__":
        frame_id = 0
        detection_processing_stats = np.zeros(2, dtype=float)
        mono3d_stats = np.zeros(2, dtype=float)
        distance_threshold = 4
        visualization = True

        for (
                detection_file_name_detection_processing,
                label_file_name_mono3d,
                input_image_file_name,
                input_label_file_name,
        ) in zip(input_files_detection_processing, input_files_mono3d, input_image_file_names, input_label_file_names):
            json_detection_processing = json.load(
                open(
                    os.path.join(input_folder_path_detection_processing, detection_file_name_detection_processing),
                )
            )
            json_mono3d = json.load(
                open(
                    os.path.join(input_folder_path_mono3d, label_file_name_mono3d),
                )
            )
            json_labels = json.load(
                open(
                    os.path.join(input_folder_path_labels, input_label_file_name),
                )
            )

            if visualization:
                img = cv2.imread(os.path.join(input_folder_path_images, input_image_file_name))
            # get 3d positions from detection processing
            positions_3d_detection_processing = get_3d_positions(json_detection_processing)
            # get 3d position from mono3d
            positions_3d_mono3d = get_3d_positions(json_mono3d)
            # get 3d positions from labels
            positions_3d_label = get_3d_positions(json_labels)

            match_positions(
                positions_3d_detection_processing,
                positions_3d_label,
                detection_processing_stats,
                (96, 96, 255),
                distance_threshold,
                visualization,
            )  # red
            match_positions(
                positions_3d_mono3d, positions_3d_label, mono3d_stats, (235, 192, 52), distance_threshold, visualization
            )  # blue

            if visualization:
                # write image
                cv2.imwrite(os.path.join(output_folder_path_images, input_image_file_name), img)
            frame_id += 1

        print(f"RMSE detection_processing: {detection_processing_stats[0] / detection_processing_stats[1]:.3f}")
        print(f"RMSE mono3d: {mono3d_stats[0] / mono3d_stats[1]:.3f}")
