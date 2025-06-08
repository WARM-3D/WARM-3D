import argparse
import glob

from tqdm import tqdm

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_dataset",
        type=str,
        help="Folder path to dataset root.",
        default="",
    )
    args = arg_parser.parse_args()
    dataset_root_folder_path = args.input_folder_path_dataset
    dataset_sub_sets = ["train", "val", "test_full"]
    sensor_modalities = ["point_clouds", "images", "labels_point_clouds"]
    camera_sensors = ["s110_camera_basler_south2_8mm", "s110_camera_basler_south1_8mm"]
    lidar_sensors = ["s110_lidar_ouster_south", "s110_lidar_ouster_north"]

    for dataset_sub_set in dataset_sub_sets:
        for sensor_modality in sensor_modalities:
            if sensor_modality == "point_clouds":
                for lidar_sensor in lidar_sensors:
                    label_file_paths = sorted(
                        glob.glob(
                            f"{dataset_root_folder_path}/{dataset_sub_set}/{sensor_modality}/{lidar_sensor}/*.pcd"
                        )
                    )
                    with open(
                            f"../../../data_split/{dataset_sub_set}/{sensor_modality}/{lidar_sensor}/file_names.txt",
                            "w",
                    ) as file:
                        for file_path_point_cloud_label in tqdm(label_file_paths):
                            file_name = file_path_point_cloud_label.split("/")[-1]
                            file.write(f"{file_name}\n")
            elif sensor_modality == "images":
                for camera_sensor in camera_sensors:
                    image_file_paths = sorted(
                        glob.glob(
                            f"{dataset_root_folder_path}/{dataset_sub_set}/{sensor_modality}/{camera_sensor}/*.jpg"
                        )
                    )
                    with open(
                            f"../../../data_split/{dataset_sub_set}/{sensor_modality}/{camera_sensor}/file_names.txt",
                            "w",
                    ) as file:
                        for file_path_image in tqdm(image_file_paths):
                            file_name = file_path_image.split("/")[-1]
                            file.write(f"{file_name}\n")
            if sensor_modality == "labels_point_clouds":
                for lidar_sensor in lidar_sensors:
                    label_file_paths = sorted(
                        glob.glob(
                            f"{dataset_root_folder_path}/{dataset_sub_set}/{sensor_modality}/{lidar_sensor}/*.json"
                        )
                    )
                    with open(
                            f"../../../data_split/{dataset_sub_set}/{sensor_modality}/{lidar_sensor}/file_names.txt",
                            "w",
                    ) as file:
                        for file_path_point_cloud_label in tqdm(label_file_paths):
                            file_name = file_path_point_cloud_label.split("/")[-1]
                            file.write(f"{file_name}\n")
