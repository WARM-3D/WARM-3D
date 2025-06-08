import argparse
import glob
import os
import json
import sys
from pathlib import Path

import numpy as np

import uuid
import hashlib
from internal.src.data_generation.set_default_dimensions import default_dimensions
from src.utils.detection import Detection, save_to_openlabel
import decimal


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_folder_path_detections', type=str, help='Path to detections input folder',
                           default='')
    argparser.add_argument('--input_file_path_calibration_data', type=str, help='Path to calibration data', default='')
    argparser.add_argument('--output_folder_path_detections', type=str, help='Output folder path detections',
                           default='')
    argparser.add_argument('--image_width', type=int, help='Image width', default=1920)
    argparser.add_argument('--image_height', type=int, help='Image height', default=1200)
    argparser.add_argument('--weather_type', type=str, help='Weather type', default='')
    argparser.add_argument('--time_of_day', type=str, help='Time of day', default='')
    argparser.add_argument('--sensor_id', type=str, help='Sensor ID', default='')

    args = argparser.parse_args()
    input_folder_path_detections = args.input_folder_path_detections
    input_file_path_calibration_data = args.input_file_path_calibration_data
    output_folder_path_detections = args.output_folder_path_detections
    sensor_id = args.sensor_id
    sensor_station_id = args.sensor_id.split("_")[0]
    image_width = args.image_width
    image_height = args.image_height

    # load calibration data
    calib_data_json = json.load(open(input_file_path_calibration_data))

    # create output folder
    if not os.path.exists(output_folder_path_detections):
        os.makedirs(output_folder_path_detections)

    # iterate over all files in input folder
    for file_path_label in sorted(glob.glob(input_folder_path_detections + '/*.json')):
        # load json file
        data_json = json.load(open(file_path_label))

        coordinate_systems = {}
        coordinate_systems["hd_map_origin"] = {
            "type": "scene_cs",
            "parent": "",
            "children": [
                sensor_station_id + "_base"
            ]
        }

        coordinate_systems[sensor_station_id + "_base"] = {
            "type": "scene_cs",
            "parent": "hd_map_origin",
            "children": [
                "common_road"
            ],
            "pose_wrt_parent": {
                "matrix4x4": np.asarray(calib_data_json["transformation_hd_map_origin_to_sensor_station_base"],
                                        dtype=float).ravel().tolist()
            }
        }

        coordinate_systems[sensor_id + "_south_driving_direction"] = {
            "type": "sensor_cs",
            "parent": "common_road",
            "children": [],
            "pose_wrt_parent": {
                "matrix4x4": np.asarray(
                    calib_data_json["transformation_common_road_to_sensor_south_driving_direction"],
                    dtype=float).ravel().tolist()
            }
        }

        coordinate_systems[sensor_id + "_north_driving_direction"] = {
            "type": "sensor_cs",
            "parent": "common_road",
            "children": [],
            "pose_wrt_parent": {
                "matrix4x4": np.asarray(
                    calib_data_json["transformation_common_road_to_sensor_north_driving_direction"],
                    dtype=float).ravel().tolist()
            }
        }

        coordinate_systems["common_road"] = {
            "type": "scene_cs",
            "parent": sensor_station_id + "_base",
            "children": [
                sensor_id + "_south_driving_direction",
                sensor_id + "_north_driving_direction"
            ],
            "pose_wrt_parent": {
                "matrix4x4": np.asarray(calib_data_json["transformation_sensor_station_base_to_common_road"],
                                        dtype=float).ravel().tolist()
            }
        }

        file_name_label = os.path.basename(file_path_label)
        print("Processing file: " + file_name_label)

        # TODO: move to separate script
        # sensor_id = "dtwin_s40_s50"
        detections = []
        frame_properties = {}
        # timestamp = decimal.Decimal(float(data_json['timestamp_secs']) * 1000000000 + int(
        #     data_json['timestamp_nsecs'])) / 1000000000.0
        # convert to float with 9 decimal places

        seconds = decimal.Decimal(float(data_json['timestamp_secs']))
        nano_seconds = decimal.Decimal(float(data_json['timestamp_nsecs']) / 1000000000.0)
        timestamp = round(seconds + nano_seconds, 9)

        # timestamp_float = timestamp.quantize(decimal.Decimal('.000000001'), rounding=decimal.ROUND_DOWN)
        frame_properties["timestamp"] = timestamp

        if "image_file_name" in data_json:
            frame_properties["image_file_name"] = data_json['image_file_name']
        if "point_cloud_file_name" in data_json:
            frame_properties["point_cloud_file_name"] = data_json['point_cloud_file_name']
        if "weather_type" in data_json:
            frame_properties["weather_type"] = data_json['weather_type']
        elif args.weather_type != '':
            frame_properties["weather_type"] = args.weather_type
        if args.time_of_day != '':
            frame_properties["time_of_day"] = args.time_of_day

        frame_properties["transforms"] = {
            sensor_station_id + "_base" + "_to_" + "hd_map_origin": {
                "src": sensor_station_id + "_base",
                "dst": "hd_map_origin",
                "transform_src_to_dst": {
                    "matrix4x4": np.asarray(
                        calib_data_json["transformation_sensor_station_base_to_hd_map_origin"],
                        dtype=float).ravel().tolist()
                }
            }
        }
        coordinate_system_common_road = "common_road"
        coordinate_system_hd_map_origin = "hd_map_origin"
        coordinate_system_sensor_station_base = sensor_station_id + "_base"

        frame_properties["transforms"][
            coordinate_system_common_road + "_to_" + coordinate_system_sensor_station_base] = {
            "src": coordinate_system_common_road,
            "dst": coordinate_system_sensor_station_base,
            "transform_src_to_dst": {
                "matrix4x4": np.asarray(
                    calib_data_json["transformation_common_road_to_sensor_station_base"],
                    dtype=float).ravel().tolist()
            }
        }
        frame_properties["transforms"][
            sensor_id + "_south_driving_direction_to_" + coordinate_system_common_road] = {
            "src": sensor_id + "_south_driving_direction",
            "dst": coordinate_system_common_road,
            "transform_src_to_dst": {
                "matrix4x4": np.asarray(
                    calib_data_json["transformation_sensor_south_driving_direction_to_common_road"],
                    dtype=float).ravel().tolist()
            }
        }
        frame_properties["transforms"][
            sensor_id + "_north_driving_direction_to_" + coordinate_system_common_road] = {
            "src": sensor_id + "_north_driving_direction",
            "dst": coordinate_system_common_road,
            "transform_src_to_dst": {
                "matrix4x4": np.asarray(
                    calib_data_json["transformation_sensor_north_driving_direction_to_common_road"],
                    dtype=float).ravel().tolist()
            }
        }

        # iterate over all frames
        for label_obj in data_json['labels']:

            # if type(label_obj['id']) == str:
            #     category_id = label_obj['id']
            #     id = int(category_id.split('_')[1])
            # else:
            #     id = int(label_obj['id'])
            category = label_obj['category']
            detection = Detection(location=None, dimensions=None, yaw=None, category=category)

            # NOTE: R0_S0 id is already a uuid
            if len(str(label_obj['id'])) == 36:
                detection.uuid = str(label_obj['id'])
            else:
                # R0_S1 #
                detection.uuid = uuid.uuid5(uuid.NAMESPACE_OID, str(label_obj['id']))

            if "box3d" in label_obj:
                location = np.array([label_obj['box3d']['location']['x'], label_obj['box3d']['location']['y'],
                                     label_obj['box3d']['location']['z']], dtype=np.float64)
                # make location [[x], [y], [z]]
                location = np.expand_dims(location, axis=1)
                dimensions = np.array([label_obj['box3d']['dimension']['length'],
                                       label_obj['box3d']['dimension']['width'],
                                       label_obj['box3d']['dimension']['height']], dtype=np.float64)
                yaw = float(label_obj['box3d']['orientation']['rotationYaw'])
                # set default dimensions if all dimensions are zero
                if np.all(dimensions == 0) and category in default_dimensions:
                    dimensions = default_dimensions[category]

                detection.location = location
                detection.dimensions = dimensions
                detection.yaw = yaw
                detection.category = category

                detection.sensor_id = sensor_id
            if "box3d_projected" in label_obj and len(label_obj['box3d_projected'].items()) > 0:
                detection.box3d_projected = label_obj['box3d_projected']
                box3d_projected = label_obj['box3d_projected']
                # store 8 corner points of box3d_projected
                box3d_projected_corners = []
                x_min = sys.maxsize
                y_min = sys.maxsize
                x_max = -sys.maxsize
                y_max = -sys.maxsize
                for attribute, value in box3d_projected.items():
                    box3d_projected_corners.append(value)
                    if value[0] < x_min:
                        x_min = value[0]
                    if value[1] < y_min:
                        y_min = value[1]
                    if value[0] > x_max:
                        x_max = value[0]
                    if value[1] > y_max:
                        y_max = value[1]
                    # NOTE: save_to_openlabel function converts from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height] when writing to OpenLABEL
                    release = "r01"
                    if release == "r01":
                        IMAGE_WIDTH = 1
                        IMAGE_HEIGHT = 1
                    else:
                        IMAGE_WIDTH = 1920
                        IMAGE_HEIGHT = 1200
                    detection.bbox_2d = np.array(
                        [int(x_min * IMAGE_WIDTH), int(y_min * IMAGE_HEIGHT), int(x_max * IMAGE_WIDTH),
                         int(y_max * IMAGE_HEIGHT)])

            # NOTE: R0_S1 = color_body
            if "color_body" in label_obj['attributes']:
                detection.color = label_obj['attributes']['color_body']

            # NOTE: R0_S0 = Color Body
            if "Body Color" in label_obj['attributes']:
                detection.color = label_obj['attributes']['Body Color']["value"]

            # NOTE: R0_S0: value needed
            if "Occluded" in label_obj['attributes']:
                detection.occlusion_level = label_obj['attributes']['Occluded']["value"]

            # NOTE: R0_S0: value not needed
            if "Occluded" in label_obj['attributes']:
                detection.occlusion_level = label_obj['attributes']['Occluded']

            # NOTE: R0_S0 is using with_trailer
            if "with_trailer" in label_obj['attributes']:
                # R0_S1
                if label_obj['attributes']['with_trailer'] == 'true':
                    detection.has_trailer = True
                else:
                    detection.has_trailer = False
            if "electric" in label_obj['attributes']:
                if label_obj['attributes']['electric'] == True:
                    detection.is_electric = True
                elif label_obj['attributes']['electric'] == False:
                    detection.is_electric = False
                else:
                    detection.is_electric = None

            # NOTE: R0_S0 value needed
            if "Truck Type" in label_obj['attributes']:
                detection.sub_type = label_obj['attributes']['Truck Type']["value"]

            if "Bus Type" in label_obj['attributes']:
                detection.sub_type = label_obj['attributes']['Bus Type']["value"]

            if "Trailer Type" in label_obj['attributes']:
                detection.sub_type = label_obj['attributes']['Trailer Type']["value"]

            # NOTE: R0_S0 = Motorcycle Type
            if "Motorcycle Type" in label_obj['attributes']:
                detection.sub_type = label_obj['attributes']['Motorcycle Type']["value"]

            if "with_rider" in label_obj['attributes']:
                if label_obj['attributes']['with_rider'] == 'true':
                    detection.has_rider = True
                elif label_obj['attributes']['with_rider'] == 'false':
                    detection.has_rider = False
                else:
                    detection.has_rider = None

            # NOTE: R0_S1 = type
            if "type" in label_obj['attributes']:
                detection.sub_type = label_obj['attributes']['type']

            # NOTE: R0_S1 = number_of_trailers
            if "number_of_trailers" in label_obj['attributes']:
                detection.number_of_trailers = int(str(label_obj['attributes']['number_of_trailers']))

            # note: R0_S0 = Number of Trailers
            if "Number of Trailers" in label_obj['attributes']:
                detection.number_of_trailers = int(str(label_obj['attributes']['Number of Trailers']["value"]))

            # NOTE: R0_S1 = Has Flashing Light
            if "Has Flashing Lights" in label_obj['attributes']:
                if label_obj['attributes']['Has Flashing Lights']["value"] == 'true':
                    detection.has_flashing_light = True
                elif label_obj['attributes']['Has Flashing Lights']["value"] == 'false':
                    detection.has_flashing_light = False
                else:
                    detection.has_flashing_light = None

            if "velocity" in label_obj:
                # velocity vector (vx, vy, vz) in m/s
                detection.velocity = np.array(
                    [label_obj["velocity"]["x"], label_obj["velocity"]["y"], label_obj["velocity"]["z"]])
                # speed (float): Speed of the object in m/s
                detection.speed = np.linalg.norm(detection.velocity)

            detections.append(detection)
        # sort detections by id
        detections.sort(key=lambda x: x.id)
        save_to_openlabel(detections, file_name_label, Path(output_folder_path_detections),
                          frame_properties=frame_properties, coordinate_systems=coordinate_systems)
