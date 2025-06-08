import json
import numpy as np
import pandas as pd

with open(
        "/mnt/hdd_data1/28_datasets/00_a9_dataset/r02_tum_traffic_intersection_dataset_train_val_test_nuscenes_format/nuscenes_infos_train.pkl",
        'rb') as input_file:
    data = pd.read_pickle(input_file)

for info in data["infos"]:
    info["cam_infos"]["s110_camera_basler_south1_8mm"]["ego_pose"] = np.array(
        info["cam_infos"]["s110_camera_basler_south1_8mm"]["ego_pose"]).tolist()
    info["cam_infos"]["s110_camera_basler_south1_8mm"]["calibrated_sensor"] = np.array(
        info["cam_infos"]["s110_camera_basler_south1_8mm"]["calibrated_sensor"]).tolist()
    info["cam_infos"]["s110_camera_basler_south1_8mm"]["camera_intrinsic"] = np.array(
        info["cam_infos"]["s110_camera_basler_south1_8mm"]["camera_intrinsic"]).tolist()

    info["cam_infos"]["s110_camera_basler_south2_8mm"]["ego_pose"] = np.array(
        info["cam_infos"]["s110_camera_basler_south2_8mm"]["ego_pose"]).tolist()
    info["cam_infos"]["s110_camera_basler_south2_8mm"]["calibrated_sensor"] = np.array(
        info["cam_infos"]["s110_camera_basler_south2_8mm"]["calibrated_sensor"]).tolist()
    info["cam_infos"]["s110_camera_basler_south2_8mm"]["camera_intrinsic"] = np.array(
        info["cam_infos"]["s110_camera_basler_south2_8mm"]["camera_intrinsic"]).tolist()

    if "s110_lidar_ouster_north" in info["lidar_infos"]:
        info["lidar_infos"]["s110_lidar_ouster_north"]["ego_pose"] = np.array(
            info["lidar_infos"]["s110_lidar_ouster_north"]["ego_pose"]).tolist()
        info["lidar_infos"]["s110_lidar_ouster_north"]["calibrated_sensor"] = np.array(
            info["lidar_infos"]["s110_lidar_ouster_north"]["calibrated_sensor"]).tolist()
    if "s110_lidar_ouster_south" in info["lidar_infos"]:
        info["lidar_infos"]["s110_lidar_ouster_south"]["ego_pose"] = np.array(
            info["lidar_infos"]["s110_lidar_ouster_south"]["ego_pose"]).tolist()
        info["lidar_infos"]["s110_lidar_ouster_south"]["calibrated_sensor"] = np.array(
            info["lidar_infos"]["s110_lidar_ouster_south"]["calibrated_sensor"]).tolist()

json.dump(data,
          open(
              "/mnt/hdd_data1/28_datasets/00_a9_dataset/r02_tum_traffic_intersection_dataset_train_val_test_nuscenes_format/nuscenes_infos_train.json",
              "w"))
