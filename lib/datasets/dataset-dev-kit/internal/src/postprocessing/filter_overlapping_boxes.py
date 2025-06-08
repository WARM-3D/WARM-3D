import argparse
import json
import os
from pathlib import Path

import open3d
import torch
from pytorch3d.ops import box3d_overlap
from scipy.spatial.transform.rotation import Rotation as R

import numpy as np

from src.utils.detection import save_to_openlabel, Detection
from src.utils.vis_utils import VisualizationUtils


def get_corner_points(box: open3d.geometry.OrientedBoundingBox):
    center = np.array(box.center)
    extent = np.array(box.extent) / 2  # Half-lengths
    R = np.array(box.R)
    corners = np.empty((8, 3))
    for i in range(8):
        sign = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32)
        corner = center + R @ (sign[i] * extent)
        corners[i] = corner
    return corners


def filter_by_overlap(detections, boxes):
    overlapping_detections = []
    valid_detections = []
    valid_boxes = []
    for i, box in enumerate(boxes):
        if i in overlapping_detections:
            continue
        # source_corner_points = np.asarray(box.get_box_points())
        # convert to tensor
        # source_corner_points = torch.from_numpy(source_corner_points).float()

        # get source corner points from center, extent and rotation using get_3d_box
        # get yaw from rotation matrix
        # heading_angle = np.arctan2(box.R[2, 0], box.R[0, 0])
        heading_angle = np.arctan2(box.R[1, 0], box.R[0, 0])
        source_corner_points = get_corner_points(box)

        # convert array to torch tensor
        source_corner_points = torch.from_numpy(source_corner_points).float()

        # add batch dimension
        source_corner_points = source_corner_points.unsqueeze(0)

        current_max_score_idx = i
        current_to_be_filtered = []
        for j in range(i + 1, len(boxes)):
            if j in overlapping_detections:
                continue

            # target_corner_points = np.asarray(boxes[j].get_box_points())
            # convert to tensor
            # target_corner_points = torch.from_numpy(target_corner_points).float()

            # get target corner points from center, extent and rotation using get_3d_box
            # get yaw from rotation matrix
            heading_angle = np.arctan2(boxes[j].R[1, 0], boxes[j].R[0, 0])
            target_corner_points = get_corner_points(boxes[j])
            # convert array to torch tensor
            target_corner_points = torch.from_numpy(target_corner_points).float()

            # add batch dimension
            target_corner_points = target_corner_points.unsqueeze(0)

            intersection_vol, iou_3d = box3d_overlap(source_corner_points, target_corner_points)
            # define epsilon
            epsilon = 1e-4
            # --- Add rules here ---
            if iou_3d.cpu().numpy()[0, 0] < epsilon:
                # no overlap found
                continue
            if detections[current_max_score_idx].category == "TRUCK" and detections[j].category == "TRAILER" or \
                    detections[current_max_score_idx].category == "TRAILER" and detections[j].category == "TRUCK":
                continue

                # TODO: temp set overlap to True for testing:
            detections[current_max_score_idx].overlap = True
            detections[j].overlap = True
            # print overlap indices
            print("Overlap detected between detections: trackID1", detections[current_max_score_idx].uuid,
                  "category",
                  detections[current_max_score_idx].category,
                  "and trackID2:", detections[j].uuid, "category", detections[j].category, "IoU: ",
                  iou_3d.cpu().numpy()[0, 0])

            if detections[current_max_score_idx].score < detections[j].score:
                current_to_be_filtered.append(current_max_score_idx)
                current_max_score_idx = j
            else:
                current_to_be_filtered.append(j)
        overlapping_detections.extend(current_to_be_filtered)
        valid_detections.append(detections[current_max_score_idx])
        valid_boxes.append(boxes[current_max_score_idx])
    # TODO: temp return all detections for testing
    print("Detections before filtering:", len(detections), "Detections after filtering", len(valid_detections))
    return valid_detections, valid_boxes
    # return detections, boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Input directory path",
        default="",
    )
    parser.add_argument(
        "--output_folder_path_boxes_filtered",
        type=str,
        help="Output directory path",
        default="",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_folder_path_boxes_filtered):
        os.makedirs(args.output_folder_path_boxes_filtered)

    for file_name in sorted(os.listdir(args.input_folder_path_boxes)):
        label_data = json.load(open(os.path.join(args.input_folder_path_boxes, file_name)))
        detections = []
        frame_id_str = None
        frame_properties = None
        coordinate_systems = label_data["openlabel"]["coordinate_systems"]
        if "streams" in label_data["openlabel"]:
            streams = label_data["openlabel"]["streams"]
        else:
            streams = None
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            frame_id_str = str(frame_id)
            if "frame_properties" in frame_obj:
                frame_properties = frame_obj["frame_properties"]
            else:
                frame_properties = None
            for object_id, label in frame_obj["objects"].items():
                cuboid = np.array(label["object_data"]["cuboid"]["val"])
                if "attributes" in label["object_data"]["cuboid"]:
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "sensor_id"
                    )
                    if attribute is not None:
                        sensor_id = attribute["val"]
                    else:
                        sensor_id = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "body_color"
                    )
                    if attribute is not None:
                        color = attribute["val"]
                    else:
                        color = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                    )
                    if attribute is not None:
                        num_points = int(float(attribute["val"]))
                    else:
                        num_points = -1
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "score"
                    )
                    if attribute is not None:
                        score = round(float(attribute["val"]), 2)
                    else:
                        score = -1
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "occlusion_level"
                    )
                    if attribute is not None:
                        occlusion_level = attribute["val"]
                    else:
                        occlusion_level = "NOT_OCCLUDED"
                else:
                    sensor_id = ""
                    color = ""
                    num_points = -1
                    score = -1
                detections.append(
                    Detection(
                        uuid=object_id,
                        category=label["object_data"]["type"],
                        location=np.array([[cuboid[0]], [cuboid[1]], [cuboid[2]]]),
                        dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                        yaw=R.from_quat(np.array([cuboid[3], cuboid[4], cuboid[5], cuboid[6]])).as_euler(
                            "xyz", degrees=False
                        )[2],
                        score=score,
                        num_lidar_points=num_points,
                        color=color,
                        occlusion_level=occlusion_level,
                        sensor_id=sensor_id,
                    )
                )

        boxes = []
        for detection in detections:
            bbox = open3d.geometry.OrientedBoundingBox()
            bbox.center = detection.location
            bbox.R = open3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, detection.yaw]))
            bbox.extent = detection.dimensions
            bbox.color = np.array([1, 0, 0])
            boxes.append(bbox)

        # filter by overlap
        detections, boxes = filter_by_overlap(detections, boxes)

        save_to_openlabel(
            detections,
            file_name,
            Path(args.output_folder_path_boxes_filtered),
            coordinate_systems,
            frame_properties,
            frame_id_str,
            streams,
        )
