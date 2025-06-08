#!/usr/bin/env python
import os
# import json
import simplejson as json
import copy
import sys
from pathlib import Path
from decimal import *
import numpy as np
import argparse


def update_cam_id(cam_id):
    if cam_id == 's040_n_cam_near':
        return "s040_camera_basler_north_16mm"
    elif cam_id == 's040_n_cam_far':
        return "s040_camera_basler_north_50mm"
    elif cam_id == 's050_s_cam_near':
        return "s050_camera_basler_south_16mm"
    elif cam_id == 's050_s_cam_far':
        return "s050_camera_basler_south_50mm"
    elif cam_id == 'm090_n_cam_16_k':
        return "m090_camera_basler_north_16mm"
    elif cam_id == 'm090_o_cam_50_m':
        return "m090_camera_basler_east_50mm"
    elif cam_id == 'm090_w_cam_16_b':
        return "m090_camera_basler_west_16mm"
    elif cam_id == 'm090_w_cam_50_k':
        return "m090_camera_basler_west_50mm"
    elif cam_id == 's110_n_cam_16_b':
        return "s110_camera_basler_north_16mm"
    elif cam_id == 's110_n_cam_50_m':
        return "s110_camera_basler_north_50mm"
    elif cam_id == 's110_o_cam_16_b':
        return "s110_camera_basler_east_16mm"
    elif cam_id == 's110_s1_cam_8_b':
        return "s110_camera_basler_south1_8mm"
    elif cam_id == 's110_s2_cam_8_b':
        return "s110_camera_basler_south2_8mm"
    else:
        return cam_id
        # print("Error: Unknown camera ID: "+cam_id)
        # sys.exit(0)


sensor_ids = ['s040_camera_basler_north_50mm', 's040_camera_basler_north_16mm', 's050_camera_basler_south_16mm',
              's050_camera_basler_south_50mm']
# sensor_ids = ['s050_lidar_valeo_south']

# r01_s04, r01_s05
# sensor_ids = ['m090_camera_basler_west_50mm', 's110_camera_basler_east_16mm', 's110_camera_basler_east_50mm',
#               's110_camera_basler_north_16mm', 's110_camera_basler_north_50mm', 's110_camera_basler_south1_8mm',
#               's110_camera_basler_south2_8mm']
# sensor_ids = ['s110_lidar_ouster_south', 's110_lidar_ouster_north', 's110_lidar_valeo_north_west']

# r01_s08, r01_s09 (corrected labels)
# TODO: add missing calibration data for remaining 3 cameras on m090
# sensor_ids = ['m090_camera_basler_east_50mm', 'm090_camera_basler_north_16mm', 'm090_camera_basler_west_16mm',
#               'm090_camera_basler_west_50mm',
#               's110_camera_basler_east_16mm', 's110_camera_basler_north_16mm', 's110_camera_basler_north_50mm',
#               's110_camera_basler_south1_8mm',
#               's110_camera_basler_south2_8mm']
# r01_s08, r01_s09 (original labels)
# sensor_ids = ['m090_n_cam_16_k', 'm090_o_cam_50_m', 'm090_w_cam_16_b', 'm090_w_cam_50_k', 's110_n_cam_16_b',
#              's110_n_cam_50_m', 's110_o_cam_16_b', 's110_s1_cam_8_b', 's110_s2_cam_8_b']
# sensor_ids = ['s110_camera_basler_south1_8mm', 's110_camera_basler_south2_8mm','m090_camera_basler_west_50mm','s110_camera_basler_east_16mm', 's110_camera_basler_east_50mm', 's110_camera_basler_north_16mm', 's110_camera_basler_north_50mm']

# sensor_ids = ['m090_n_cam_16_k', 'm090_o_cam_50_m', 'm090_w_cam_16_b', 'm090_w_cam_50_k', 's110_camera_basler_north_16mm',
#            's110_camera_basler_north_50mm', 's110_camera_basler_east_16mm', 's110_camera_basler_south1_8mm',
#            's110_camera_basler_south2_8mm']

SUBSET = "r01_s03"

if SUBSET == "r01_s01" or SUBSET == "r01_s03":
    WEATHER_TYPE = "SNOWY"
elif SUBSET == "r01_s02":
    WEATHER_TYPE = "FOGGY"
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s04" or SUBSET == "r01_s05" or SUBSET == "r01_s06" or SUBSET == "r01_s07" or SUBSET == "r01_s08":
    WEATHER_TYPE = "SUNNY"
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s09":
    WEATHER_TYPE = "RAINY"
    TIME_OF_DAY = "NIGHT"

if SUBSET == "r01_s01":
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s02":
    TIME_OF_DAY = "NIGHT"
elif SUBSET == "r01_s03":
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s04" or SUBSET == "r01_s05":
    TIME_OF_DAY = "DUSK"
elif SUBSET == "r01_s06" or SUBSET == "r01_s07":
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s08":
    TIME_OF_DAY = "DAY"
elif SUBSET == "r01_s09":
    TIME_OF_DAY = "NIGHT"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

coordinate_system_common_road = "common_road"
coordinate_system_hd_map_origin = "hd_map_origin"

for sensor_id in sensor_ids:
    input_folder_path_labels = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/04_labels_original/" + sensor_id + "/"
    sensor_id = update_cam_id(sensor_id)
    output_folder_path_labels_openlabel = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/04_labels_openlabel/" + sensor_id + "/"

    input_file_path_calibration_data = "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/" + SUBSET + "/05_calibration/" + sensor_id + ".json"

    sensor_station = sensor_id.split("_")[0]
    coordinate_system_sensor = sensor_id
    coordinate_system_sensor_station_base = sensor_id.split("_")[0] + "_base"

    if not os.path.exists(output_folder_path_labels_openlabel):
        print("creating output directory for corrected labels (in openlabel format)....")
        path = Path(output_folder_path_labels_openlabel)
        path.mkdir(parents=True)

    for frame_id, filename_label in enumerate(sorted(os.listdir(input_folder_path_labels))):
        print("processing file: ", filename_label)

        parts = filename_label.split("_")
        filename_new = parts[0] + "_" + parts[1] + "_" + sensor_id + ".json"
        # os.rename(os.path.join(input_folder, filename),os.path.join(input_folder,filename_new))

        json_label_input_file = open(os.path.join(input_folder_path_labels, filename_label))
        json_label_data = json.load(json_label_input_file)

        json_calib_file = open(input_file_path_calibration_data, )
        json_calib_data = json.load(json_calib_file)

        output_json_data = {}
        output_json_data["openlabel"] = {
            "metadata": {
                "schema_version": "1.0.0"
            },
            "coordinate_systems": {
                "hd_map_origin": {
                    "type": "scene_cs",
                    "parent": "",
                    "children": [
                        sensor_station + "_base"
                    ]
                }
            }
        }

        if sensor_station == "s040" or sensor_station == "s050":
            output_json_data["openlabel"]["coordinate_systems"][sensor_station + "_base"] = {
                "type": "scene_cs",
                "parent": "hd_map_origin",
                "children": [
                    "common_road"
                ],
                "pose_wrt_parent": {
                    "matrix4x4": np.asarray(json_calib_data["transformation_hd_map_origin_to_sensor_station_base"],
                                            dtype=float).ravel().tolist()
                }
            }
            output_json_data["openlabel"]["coordinate_systems"][sensor_id + "_south_driving_direction"] = {
                "type": "sensor_cs",
                "parent": "common_road",
                "children": [],
                "pose_wrt_parent": {
                    "matrix4x4": np.asarray(
                        json_calib_data["transformation_common_road_to_sensor_south_driving_direction"],
                        dtype=float).ravel().tolist()
                }
            }
            output_json_data["openlabel"]["coordinate_systems"][sensor_id + "_north_driving_direction"] = {
                "type": "sensor_cs",
                "parent": "common_road",
                "children": [],
                "pose_wrt_parent": {
                    "matrix4x4": np.asarray(
                        json_calib_data["transformation_common_road_to_sensor_north_driving_direction"],
                        dtype=float).ravel().tolist()
                }
            }
            output_json_data["openlabel"]["coordinate_systems"]["common_road"] = {
                "type": "scene_cs",
                "parent": sensor_station + "_base",
                "children": [
                    sensor_id + "_south_driving_direction",
                    sensor_id + "_north_driving_direction"
                ],
                "pose_wrt_parent": {
                    "matrix4x4": np.asarray(json_calib_data["transformation_sensor_station_base_to_common_road"],
                                            dtype=float).ravel().tolist()
                }
            }
        if sensor_station == "m090" or sensor_station == "s110":
            output_json_data["openlabel"]["coordinate_systems"][sensor_station + "_base"] = {
                "type": "scene_cs",
                "parent": "hd_map_origin",
                "children": [
                    sensor_id
                ],
                "pose_wrt_parent": {
                    "matrix4x4": np.asarray(json_calib_data["transformation_hd_map_origin_to_sensor_station_base"],
                                            dtype=float).ravel().tolist()
                }
            }
            if sensor_id == "m090_camera_basler_west_50mm" or sensor_id == "s110_camera_basler_north_16mm" or sensor_id == "s110_camera_basler_north_50mm" or sensor_id == "s110_camera_basler_south2_8mm":
                output_json_data["openlabel"]["coordinate_systems"][sensor_id] = {
                    "type": "sensor_cs",
                    "parent": sensor_station + "_base",
                    "children": [],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_station_base_to_sensor"],
                            dtype=float).ravel().tolist()
                    }
                }
            if sensor_id == "s110_camera_basler_east_16mm" or sensor_id == "s110_camera_basler_east_50mm" or sensor_id == "s110_camera_basler_south1_8mm":
                output_json_data["openlabel"]["coordinate_systems"][sensor_id + "_east_driving_direction"] = {
                    "type": "sensor_cs",
                    "parent": sensor_station + "_base",
                    "children": [],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_station_base_to_sensor_east_driving_direction"],
                            dtype=float).ravel().tolist()
                    }
                }
                output_json_data["openlabel"]["coordinate_systems"][sensor_id + "_west_driving_direction"] = {
                    "type": "sensor_cs",
                    "parent": sensor_station + "_base",
                    "children": [],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_station_base_to_sensor_west_driving_direction"],
                            dtype=float).ravel().tolist()
                    }
                }
            if sensor_id == "s110_camera_basler_east_50mm":
                output_json_data["openlabel"]["coordinate_systems"][sensor_id + "_back"] = {
                    "type": "sensor_cs",
                    "parent": sensor_station + "_base",
                    "children": [],
                    "pose_wrt_parent": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_station_base_to_sensor_back"],
                            dtype=float).ravel().tolist()
                    }
                }
        objects_map = {}
        frame_map = {}
        seconds = Decimal(float(parts[0]))
        nano_seconds = Decimal(float(parts[1]) / 1000000000.0)
        timestamp = round(seconds + nano_seconds, 9)
        frame_map[frame_id] = {
            "frame_properties": {
                "timestamp": timestamp,
                "image_file_name": filename_new.replace(".json", ".png"),
                "weather_type": WEATHER_TYPE,
                "time_of_day": TIME_OF_DAY,
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

        if sensor_station == "s040" or sensor_station == "s050":
            frame_map[frame_id]["frame_properties"]["transforms"][
                coordinate_system_common_road + "_to_" + coordinate_system_sensor_station_base] = {
                "src": coordinate_system_common_road,
                "dst": coordinate_system_sensor_station_base,
                "transform_src_to_dst": {
                    "matrix4x4": np.asarray(
                        json_calib_data["transformation_common_road_to_sensor_station_base"],
                        dtype=float).ravel().tolist()
                }
            }
            frame_map[frame_id]["frame_properties"]["transforms"][
                coordinate_system_sensor + "_south_driving_direction_to_" + coordinate_system_common_road] = {
                "src": coordinate_system_sensor + "_south_driving_direction",
                "dst": coordinate_system_common_road,
                "transform_src_to_dst": {
                    "matrix4x4": np.asarray(
                        json_calib_data["transformation_sensor_south_driving_direction_to_common_road"],
                        dtype=float).ravel().tolist()
                }
            }
            frame_map[frame_id]["frame_properties"]["transforms"][
                coordinate_system_sensor + "_north_driving_direction_to_" + coordinate_system_common_road] = {
                "src": coordinate_system_sensor + "_north_driving_direction",
                "dst": coordinate_system_common_road,
                "transform_src_to_dst": {
                    "matrix4x4": np.asarray(
                        json_calib_data["transformation_sensor_north_driving_direction_to_common_road"],
                        dtype=float).ravel().tolist()
                }
            }
        if sensor_station == "m090":
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
        if sensor_station == "s110":
            if sensor_id == "s110_camera_basler_east_16mm" or sensor_id == "s110_camera_basler_east_50mm" or sensor_id == "s110_camera_basler_south1_8mm":
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_sensor + "_east_driving_direction_to_" + coordinate_system_sensor_station_base] = {
                    "src": coordinate_system_sensor + "_east_driving_direction",
                    "dst": coordinate_system_sensor_station_base,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_east_driving_direction_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_sensor + "_west_driving_direction_to_" + coordinate_system_sensor_station_base] = {
                    "src": coordinate_system_sensor + "_west_driving_direction",
                    "dst": coordinate_system_sensor_station_base,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_west_driving_direction_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }
            if sensor_id == "s110_camera_basler_east_50mm":
                frame_map[frame_id]["frame_properties"]["transforms"][
                    coordinate_system_sensor + "_back_to_" + coordinate_system_sensor_station_base] = {
                    "src": coordinate_system_sensor + "_back",
                    "dst": coordinate_system_sensor_station_base,
                    "transform_src_to_dst": {
                        "matrix4x4": np.asarray(
                            json_calib_data["transformation_sensor_back_to_sensor_station_base"],
                            dtype=float).ravel().tolist()
                    }
                }

            if sensor_id == "s110_camera_basler_north_16mm" or sensor_id == "s110_camera_basler_north_50mm" or sensor_id == "s110_camera_basler_south2_8mm":
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

        for label in json_label_data['labels']:
            category = label['category']
            cuboid_2d = None
            if (category.upper() != "LICENSE PLATE LOCATION"):
                # cuboid_data = label.pop('cuboid')
                cuboid_2d = label['cuboid']
            else:
                license_plate_keypoints = label["rectangle"]["points"]
            attributes = label['attributes']

            # pedestrian attributes: occlusion_level
            if category == "Pedestrian":
                object_attributes = {
                    "text": [{
                        "name": "occlusion_level",
                        "val": attributes["Occluded"]["value"]
                    }]
                }
            elif category == "Car" or category == "Bus" or category == "Van" or category == "Truck" or category == "Emergency Vehicle" or category == "Other" or category == "Other Vehicles":
                object_attributes = {
                    "text": [],
                    "num": [],
                    "boolean": []
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
                        "number_of_trailers": int(attributes["Number of Trailers"]["value"])
                    }
                    object_attributes["num"].append(num_of_trailers_attribute)

                if "Has Flashing Lights" in attributes:
                    flashing_light_attribute = {
                        "name": "has_flashing_lights",
                        "val": bool(attributes["Has Flashing Lights"]["value"])
                    }
                    object_attributes["boolean"].append(flashing_light_attribute)
                if category == "Bus" and "Bus Type" in attributes:
                    bus_type_attribute = {
                        "name": "sub_type",
                        "val": attributes["Bus Type"]["value"]
                    }
                    object_attributes["text"].append(bus_type_attribute)
                if category == "Truck" and "Truck Type" in attributes:
                    truck_type_attribute = {
                        "name": "sub_type",
                        "val": attributes["Truck Type"]["value"]
                    }
                    object_attributes["text"].append(truck_type_attribute)
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
                        }]
                }
                if category == "Trailer" and "Trailer Type" in attributes:
                    trailer_type_attribute = {
                        "name": "sub_type",
                        "val": attributes["Trailer Type"]["value"]
                    }
                    object_attributes["text"].append(trailer_type_attribute)
            elif category == "Bicycle" or category == "Motorcycle":
                object_attributes = {
                    "text": [{
                        "name": "occlusion_level",
                        "val": attributes["Occluded"]["value"]
                    }],
                    "boolean": [{
                        "name": "has_rider",
                        "val": bool(attributes["Has Rider"]["value"])
                    }]
                }
                if category == "Motorcycle" and "Motorcycle Type" in attributes:
                    motorcycle_type_attribute = {
                        "name": "sub_type",
                        "val": attributes["Motorcycle Type"]["value"]
                    }
                    object_attributes["text"].append(motorcycle_type_attribute)
            elif category == "License Plate Location":
                object_attributes = {
                    "text": [{
                        "name": "occlusion_level",
                        "val": attributes["Occluded"]["value"]
                    }]
                }
            # license plate category -> pass
            # print("Unknown category: ", category)
            # sys.exit()
            # car,bus,van,truck, emergency_vehicle, other attributes: body_color, number_of_trailers, occlusion_level
            # trailer attributes: body_color, occlusion_level
            # bicycle, motorcycle attributes: has_rider, occlusion_level
            category = category.replace(" ", "_")
            if category == "Other_Vehicles":
                category = "Other"

            # add keypoints 2d
            if cuboid_2d is not None and not "p5" in cuboid_2d["points"]:
                # uncomplete (truncated) object. Do not include this label and continue with next one
                continue

            x_positions_2d = []
            y_positions_2d = []
            keypoints_attributes = {}
            if category.upper() == "LICENSE_PLATE_LOCATION":
                for i in range(4):
                    x_positions_2d.append(license_plate_keypoints[i]["x"] * IMAGE_WIDTH)
                    y_positions_2d.append(license_plate_keypoints[i]["y"] * IMAGE_HEIGHT)
                keypoints_attributes["points2d"] = {
                    "name": "license_plate_keypoints",
                    "val": [
                        {
                            "point2d": {
                                "name": "top_left_corner",
                                "val": [int(license_plate_keypoints[0]["x"] * IMAGE_WIDTH),
                                        int(license_plate_keypoints[0]["y"] * IMAGE_HEIGHT)]
                            }
                        },
                        {
                            "point2d": {
                                "name": "top_right_corner",
                                "val": [int(license_plate_keypoints[1]["x"] * IMAGE_WIDTH),
                                        int(license_plate_keypoints[1]["y"] * IMAGE_HEIGHT)]
                            }
                        },
                        {
                            "point2d": {
                                "name": "bottom_right_corner",
                                "val": [int(license_plate_keypoints[2]["x"] * IMAGE_WIDTH),
                                        int(license_plate_keypoints[2]["y"] * IMAGE_HEIGHT)]
                            }
                        },
                        {
                            "point2d": {
                                "name": "bottom_left_corner",
                                "val": [int(license_plate_keypoints[3]["x"] * IMAGE_WIDTH),
                                        int(license_plate_keypoints[3]["y"] * IMAGE_HEIGHT)]
                            }
                        }
                    ]
                }
            else:
                for i in range(8):
                    x_positions_2d.append(cuboid_2d["points"]["p" + str(i + 1)]["x"])
                    y_positions_2d.append(cuboid_2d["points"]["p" + str(i + 1)]["y"])

                keypoints_attributes["points2d"] = {
                    "name": "projected_bounding_box",
                    "val": [
                        {
                            "point2d": {
                                "name": "projected_2d_point_bottom_left_front",
                                "val": [int(cuboid_2d["points"]["p1"]["x"]), int(cuboid_2d["points"]["p1"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_bottom_left_back",
                                "val": [int(cuboid_2d["points"]["p2"]["x"]), int(cuboid_2d["points"]["p2"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_bottom_right_back",
                                "val": [int(cuboid_2d["points"]["p3"]["x"]), int(cuboid_2d["points"]["p3"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_bottom_right_front",
                                "val": [int(cuboid_2d["points"]["p4"]["x"]), int(cuboid_2d["points"]["p4"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_top_left_front",
                                "val": [int(cuboid_2d["points"]["p5"]["x"]), int(cuboid_2d["points"]["p5"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_top_left_back",
                                "val": [int(cuboid_2d["points"]["p6"]["x"]), int(cuboid_2d["points"]["p6"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_top_right_back",
                                "val": [int(cuboid_2d["points"]["p7"]["x"]), int(cuboid_2d["points"]["p7"]["y"])]
                            }
                        },
                        {
                            "point2d": {
                                "name": "projected_2d_point_top_right_front",
                                "val": [int(cuboid_2d["points"]["p8"]["x"]), int(cuboid_2d["points"]["p8"]["y"])]
                            }
                        }

                    ]
                }

            width = max(x_positions_2d) - min(x_positions_2d)
            x_pos_center = min(x_positions_2d) + width / 2.0
            height = max(y_positions_2d) - min(y_positions_2d)
            y_pos_center = min(y_positions_2d) + height / 2.0

            objects_map[label["id"]] = {
                "object_data": {
                    "name": category.upper() + "_" + label["id"].split("-")[0],
                    "type": category.upper(),
                    "bbox": [{
                        "name": "shape",
                        "val": [int(x_pos_center), int(y_pos_center), int(width), int(height)]
                    }],
                    "keypoints_2d": {
                        "name": "keypoints_2d",
                        "attributes": keypoints_attributes
                    },
                    "cuboid": {
                        "name": "shape3D",
                        "val": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "attributes": object_attributes
                    }

                }
            }

        frame_map[frame_id]["objects"] = objects_map
        output_json_data["openlabel"]["frames"] = frame_map

        camera_matrix_3x4 = np.array(json_calib_data["intrinsic_camera_matrix"])
        camera_matrix_optimal_3x4 = np.array(json_calib_data["optimal_intrinsic_camera_matrix"])

        output_json_data["openlabel"]["streams"] = {
            sensor_id: {
                "description": "Basler RGB camera",
                "uri": "",
                "type": "camera",
                "stream_properties": {
                    "intrinsics_pinhole": {
                        "width_px": 1920,
                        "height_px": 1200,
                        "camera_matrix_3x4": camera_matrix_3x4.tolist(),
                        "camera_matrix_optimal_3x4": camera_matrix_optimal_3x4.tolist(),
                        "distortion_coeffs_1xN": json_calib_data["dist_coefficients"]
                    }
                }
            }
        }

        if sensor_station == "s040" or sensor_station == "s050":
            output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                "calibrated_camera_matrix_south_driving_direction_3x4"] = np.array(
                json_calib_data["calibrated_intrinsic_camera_matrix_south_driving_direction"]).tolist()
            output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                "calibrated_camera_matrix_north_driving_direction_3x4"] = np.array(
                json_calib_data["calibrated_intrinsic_camera_matrix_north_driving_direction"]).tolist()

        if sensor_station == "m090":
            output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                "calibrated_camera_matrix_3x4"] = np.array(
                json_calib_data["calibrated_intrinsic_camera_matrix"]).tolist()

        if sensor_station == "s110":
            if sensor_id == "s110_camera_basler_east_16mm" or sensor_id == "s110_camera_basler_east_50mm" or sensor_id == "s110_camera_basler_south1_8mm":
                output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                    "calibrated_camera_matrix_east_driving_direction_3x4"] = np.array(
                    json_calib_data["calibrated_intrinsic_camera_matrix_east_driving_direction"]).tolist()
                output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                    "calibrated_camera_matrix_west_driving_direction_3x4"] = np.array(
                    json_calib_data["calibrated_intrinsic_camera_matrix_west_driving_direction"]).tolist()
            if sensor_id == "s110_camera_basler_east_50mm":
                output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                    "calibrated_camera_matrix_back_3x4"] = np.array(
                    json_calib_data["calibrated_intrinsic_camera_matrix_back"]).tolist()
            if sensor_id == "s110_camera_basler_north_16mm" or sensor_id == "s110_camera_basler_north_50mm" or sensor_id == "s110_camera_basler_south2_8mm":
                output_json_data["openlabel"]["streams"][sensor_id]["stream_properties"]["intrinsics_pinhole"][
                    "calibrated_camera_matrix_3x4"] = np.array(
                    json_calib_data["calibrated_intrinsic_camera_matrix"]).tolist()
        with open(os.path.join(output_folder_path_labels_openlabel, filename_new), 'w',
                  encoding='utf-8') as json_writer:
            json_string = json.dumps(output_json_data, ensure_ascii=True, indent=4)
            json_writer.write(json_string)
