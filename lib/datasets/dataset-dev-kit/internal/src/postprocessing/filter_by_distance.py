import argparse
import json
import os
from pathlib import Path
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
from src.utils.detection import detections_to_openlabel, Detection
from src.utils.vis_utils import VisualizationUtils


def filter_by_distance(detections):
    # filter/remove all detections that are too far away from the ego vehicle
    detections_valid = []
    for detection in detections:
        # get center of detection
        position_3d = detection.location
        # filter x and y position
        if position_3d[0] >= -75 and position_3d[0] <= 75 and position_3d[1] >= -75 and position_3d[1] <= 75:
            detections_valid.append(detection)

    return detections_valid


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

        print("num. boxes before filtering: ", len(detections))
        # filter by overlap
        detections = filter_by_distance(detections)
        print("num. boxes after filtering: ", len(detections))

        detections_to_openlabel(
            detections,
            file_name,
            Path(args.output_folder_path_boxes_filtered),
            coordinate_systems,
            frame_properties,
            frame_id_str,
            streams,
        )
