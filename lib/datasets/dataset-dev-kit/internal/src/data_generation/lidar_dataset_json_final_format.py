#!/usr/bin/env python
import os
# import json
import simplejson as json
import copy
import sys
from pathlib import Path
from decimal import *
import numpy as np
from scipy.spatial.transform import Rotation as R

SUBSET = "04_R1_S4"
weather_type = "SUNNY"  # S04: SUNNY, S05: SUNNY, S07: SUNNY, S08: SUNNY, S09: RAINY
time_of_day = "DUSK"  # S04: DUSK, S05: DUSK, S07: DAY, S08: DAY, S09: NIGHT
# R01_S04, R01_S05, R01_S08, R01_S09
sensor_ids = ['s110_lidar_ouster_south', 's110_lidar_ouster_north', 's110_lidar_valeo_north_west']
# R01_S07
# sensor_ids = ['s050_lidar_valeo_south']
coordinate_system_common_road = "common_road"
coordinate_system_hd_map_origin = "hd_map_origin"
import open3d as o3d


def get_num_points(point_cloud, position_3d, rotation_yaw, dimensions):
    # TODO: check whether the other sin needs to be negated instead
    obb = o3d.geometry.OrientedBoundingBox(position_3d, np.array(
        [[np.cos(rotation_yaw), -np.sin(rotation_yaw), 0], [np.sin(rotation_yaw), np.cos(rotation_yaw), 0], [0, 0, 1]]),
                                           dimensions)
    num_points = len(obb.get_point_indices_within_bounding_box(point_cloud.points))
    return num_points


for sensor_id in sensor_ids:
    # sensor_id = 's050_lidar_valeo_south'
    # TODO: use ArgumentParser
    input_folder_labels = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/05_labels_original/" + sensor_id + "/"
    output_folder_labels_openlabel = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/05_labels_openlabel/" + sensor_id + "/"
    input_folder_path_point_clouds = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/04_point_clouds/" + sensor_id + "/"
    input_file_path_calibration_data = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/06_calibration/" + sensor_id + ".json"

    sensor_station = sensor_id.split("_")[0]
    coordinate_system_sensor = sensor_id
    coordinate_system_sensor_station_base = sensor_id.split("_")[0] + "_base"

    if not os.path.exists(output_folder_labels_openlabel):
        print("creating output directory....")
        path = Path(output_folder_labels_openlabel)
        path.mkdir(parents=True)

    for frame_id, filename in enumerate(sorted(os.listdir(input_folder_labels))):
        print("processing file: ", filename)
        if sensor_id in filename:

            json_input_file = open(os.path.join(input_folder_labels, filename))
            json_data = json.load(json_input_file)

            json_calib_file = open(input_file_path_calibration_data, )
            json_calib_data = json.load(json_calib_file)

            point_cloud = o3d.io.read_point_cloud(
                os.path.join(input_folder_path_point_clouds, filename.replace(".json", ".pcd")))

            output_json_data = {}
            output_json_data["openlabel"] = {
                "metadata": {
                    "schema_version": "1.0.0"
                },
                "coordinate_systems": {}
            }

            if sensor_station == "s050":
                output_json_data["openlabel"]["coordinate_systems"]["hd_map_origin"] = {
                    "type": "scene_cs",
                    "parent": "",
                    "children": [
                        sensor_station + "_base"
                    ]
                }
                output_json_data["openlabel"]["coordinate_systems"][sensor_station + "_base"] = {
                    "type": "local_cs",
                    "parent": "hd_map_origin",
                    "children": [
                        "common_road"
                    ],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_hd_map_origin_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }
                output_json_data["openlabel"]["coordinate_systems"]["common_road"] = {
                    "type": "scene_cs",
                    "parent": sensor_station + "_base",
                    "children": [
                        sensor_id
                    ],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(json_calib_data["transformation_sensor_station_base_to_common_road"],
                                                dtype=float).ravel().tolist()
                    }
                }
                output_json_data["openlabel"]["coordinate_systems"][sensor_id] = {
                    "type": "sensor_cs",
                    "parent": "common_road",
                    "children": [],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(json_calib_data["transformation_common_road_to_sensor"],
                                                dtype=float).ravel().tolist()
                    }
                }
            if sensor_station == "s110":
                output_json_data["openlabel"]["coordinate_systems"]["hd_map_origin"] = {
                    "type": "scene_cs",
                    "parent": "",
                    "children": [
                        sensor_station + "_base"
                    ]
                }
                output_json_data["openlabel"]["coordinate_systems"][sensor_station + "_base"] = {
                    "type": "local_cs",
                    "parent": "hd_map_origin",
                    "children": [
                        sensor_id
                    ],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_hd_map_origin_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }
                if "ouster" in sensor_id:
                    output_json_data["openlabel"]["coordinate_systems"][sensor_id] = {
                        "type": "sensor_cs",
                        "parent": sensor_station + "_base",
                        "children": ["s110_camera_basler_south1_8mm", "s110_camera_basler_south2_8mm"],
                        "pose_wrt_parent": {
                            "matrix4x4": np.asarray(json_calib_data["transformation_sensor_station_base_to_sensor"],
                                                    dtype=float).ravel().tolist()
                        }
                    }
                    output_json_data["openlabel"]["coordinate_systems"]["s110_camera_basler_south1_8mm"] = {
                        "type": "sensor_cs",
                        "parent": sensor_id,
                        "children": [],
                        "pose_wrt_parent": {
                            "matrix4x4": np.asarray(
                                json_calib_data["transformation_" + sensor_id + "_to_s110_camera_basler_south1_8mm"],
                                dtype=float).ravel().tolist()
                        }
                    }
                    output_json_data["openlabel"]["coordinate_systems"]["s110_camera_basler_south2_8mm"] = {
                        "type": "sensor_cs",
                        "parent": sensor_id,
                        "children": [],
                        "pose_wrt_parent": {
                            "matrix4x4": np.asarray(
                                json_calib_data["transformation_" + sensor_id + "_to_s110_camera_basler_south2_8mm"],
                                dtype=float).ravel().tolist()
                        }
                    }
                else:
                    output_json_data["openlabel"]["coordinate_systems"][sensor_id] = {
                        "type": "sensor_cs",
                        "parent": sensor_station + "_base",
                        "children": [],
                        "pose_wrt_parent": {
                            "matrix4x4": np.asarray(json_calib_data["transformation_sensor_station_base_to_sensor"],
                                                    dtype=float).ravel().tolist()
                        }
                    }

            objects_map = {}
            frame_map = {}
            parts = filename.split("_")
            seconds = Decimal(float(parts[0]))
            nano_seconds = Decimal(float(parts[1]) / 1000000000.0)
            timestamp = round(seconds + nano_seconds, 9)
            frame_map[frame_id] = {
                "frame_properties": {
                    "timestamp": timestamp,
                    "point_cloud_file_name": filename.replace(".json", ".pcd"),
                    "weather_type": weather_type,
                    "time_of_day": time_of_day,
                    "transforms": {
                        coordinate_system_sensor_station_base + "_to_" + coordinate_system_hd_map_origin: {
                            "src": coordinate_system_sensor_station_base,
                            "dst": coordinate_system_hd_map_origin,
                            "transform_src_to_dst": {
                                "matrix4x4": np.asarray(
                                    json_calib_data["transformation_sensor_station_base_to_hd_map_origin"],
                                    dtype=float).ravel().tolist()
                            }
                        }
                    }
                }
            }
            if sensor_station == "s050":
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_common_road + "_to_" + coordinate_system_sensor_station_base] = {
                    "src": coordinate_system_common_road,
                    "dst": coordinate_system_sensor_station_base,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(json_calib_data["transformation_common_road_to_sensor_station_base"],
                                                dtype=float).ravel().tolist()
                    }
                }
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_sensor + "_to_" + coordinate_system_common_road] = {
                    "src": coordinate_system_sensor,
                    "dst": coordinate_system_common_road,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(json_calib_data["transformation_sensor_to_common_road"],
                                                dtype=float).ravel().tolist()
                    }
                }

            if sensor_station == "s110":
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_sensor + "_to_" + coordinate_system_sensor_station_base] = {
                    "src": coordinate_system_sensor,
                    "dst": coordinate_system_sensor_station_base,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }
            for label in json_data['labels']:
                category = label['category']
                if (category.upper() != 'LICENSE PLATE LOCATION'):
                    # cuboid_data = label.pop('cuboid')
                    center = label['center']
                    dimensions = label['dimensions']
                    rotation = label['rotation']
                else:
                    print("License plate detected in LiDAR point cloud. Exiting...")
                    sys.exit(0)
                attributes = label['attributes']

                # TODO: calculate num points within bounding box
                position_3d = np.array([center["x"], center["y"], center["z"]])
                if float(rotation["_x"]) == 0.0 and float(rotation["_y"]) == 0.0 and float(
                        rotation["_z"]) == 0.0 and float(rotation["_w"]) == 0.0:
                    print("found zero norm quaternion. Continue with next label...")
                    continue
                roll, pitch, yaw = R.from_quat(
                    [rotation["_x"], rotation["_y"], rotation["_z"], rotation["_w"]]).as_euler('xyz', degrees=False)
                dimensions = np.array([dimensions["length"], dimensions["width"], dimensions["height"]])
                num_points = get_num_points(point_cloud, position_3d, yaw, dimensions)

                # pedestrian attributes: occlusion_level
                if category == "Pedestrian":
                    object_attributes = {
                        "text": [{
                            "name": "occlusion_level",
                            "val": attributes["Occluded"]["value"]
                        }],
                        "num": [{
                            "name": "num_points",
                            "val": num_points
                        }]
                    }
                elif category == "Car" or category == "Bus" or category == "Van" or category == "Truck" or category == "Emergency Vehicle" or category == "Other" or category == "Other Vehicles":
                    object_attributes = {
                        "text": [],
                        "num": []
                    }
                    if "Body Color" in attributes:
                        body_color_attribute = {
                            "name": "body_color",
                            "val": attributes["Body Color"]["value"].lower()
                        }
                        object_attributes["text"].append(body_color_attribute)
                    if "Occluded" in attributes:
                        occlusion_attribute = {
                            "name": "occlusion_level",
                            "val": attributes["Occluded"]["value"]
                        }
                        object_attributes["text"].append(occlusion_attribute)
                    if "Number of Trailers" in attributes:
                        num_of_trailers_attribute = {
                            "name": "number_of_trailers",
                            "val": int(attributes["Number of Trailers"]["value"])
                        }
                        object_attributes["num"].append(num_of_trailers_attribute)

                    if "Has Flashing Lights" in attributes:
                        flashing_light_attribute = {
                            "name": "has_flashing_lights",
                            "val": bool(attributes["Has Flashing Lights"]["value"])
                        }
                        object_attributes["num"].append(flashing_light_attribute)
                    # add num points
                    num_points_attribute = {
                        "name": "num_points",
                        "val": num_points
                    }
                    object_attributes["num"].append(num_points_attribute)
                elif category == "Trailer":
                    object_attributes = {
                        "text": [
                            {
                                "name": "body_color",
                                "val": attributes["Body Color"]["value"].lower()
                            },
                            {
                                "name": "occlusion_level",
                                "val": attributes["Occluded"]["value"]
                            }],
                        "num": [{
                            "name": "num_points",
                            "val": num_points
                        }]
                    }
                elif category == "Bicycle" or category == "Motorcycle":
                    object_attributes = {
                        "text": [{
                            "name": "occlusion_level",
                            "val": attributes["Occluded"]["value"]
                        }],
                        "boolean": [{
                            "name": "has_rider",
                            "val": bool(attributes["Has Rider"]["value"])
                        }],
                        "num": [{
                            "name": "num_points",
                            "val": num_points
                        }]
                    }
                else:
                    print("Unknown category: ", category)
                    sys.exit()
                # car,bus,van,truck, emergency_vehicle, other attributes: body_color, number_of_trailers, occlusion_level
                # trailer attributes: body_color, occlusion_level
                # bicycle, motorcycle attributes: has_rider, occlusion_level
                category = category.replace(" ", "_")
                if category == "Other_Vehicles":
                    category = "Other"
                objects_map[label["id"]] = {
                    "object_data": {
                        "name": category.upper() + "_" + label["id"].split("-")[0],
                        "type": category.upper(),
                        "cuboid": {
                            "name": "shape3D",
                            "val": [center["x"], center["y"], center["z"], rotation["_x"], rotation["_y"],
                                    rotation["_z"], rotation["_w"], dimensions[0], dimensions[1],
                                    dimensions[2]],
                            "attributes": object_attributes
                        }
                    }
                }

            frame_map[frame_id]["objects"] = objects_map
            output_json_data["openlabel"]["frames"] = frame_map

            if sensor_station == "s110" and "ouster" in sensor_id:
                output_json_data["openlabel"]["streams"] = {
                    "s110_camera_basler_south1_8mm": {
                        "description": "Basler RGB camera",
                        "uri": "",
                        "type": "camera",
                        "stream_properties": {
                            "intrinsics_pinhole": {
                                "width_px": 1920,
                                "height_px": 1200,
                                "camera_matrix_3x4": np.array(
                                    json_calib_data["intrinsic_camera_matrix_s110_camera_basler_south1_8mm"]).tolist()
                            }
                        }
                    },
                    "s110_camera_basler_south2_8mm": {
                        "description": "Basler RGB camera",
                        "uri": "",
                        "type": "camera",
                        "stream_properties": {
                            "intrinsics_pinhole": {
                                "width_px": 1920,
                                "height_px": 1200,
                                "camera_matrix_3x4": np.array(
                                    json_calib_data["intrinsic_camera_matrix_s110_camera_basler_south2_8mm"]).tolist()
                            }
                        }
                    }
                }

            with open(os.path.join(output_folder_labels_openlabel, filename), 'w', encoding='utf-8') as json_writer:
                # json.dumps(output_json_data, json_writer, ensure_ascii=True, indent=4)
                json_string = json.dumps(output_json_data, ensure_ascii=True, indent=4)
                json_writer.write(json_string)
