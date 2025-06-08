import argparse
import glob
import os
import pathlib
from tqdm import tqdm


def read_file_names(data_split_root):
    file_names_test_set = {}
    # iterate all folders in data_split_root
    for sensor_modality in sorted(os.listdir(data_split_root + "/test_full")):
        # iterate all files in folder
        for sensor_id in sorted(os.listdir(data_split_root + "/test_full/" + sensor_modality)):
            if sensor_modality not in file_names_test_set:
                file_names_test_set[sensor_modality] = {}
            if sensor_id not in file_names_test_set[sensor_modality]:
                file_names_test_set[sensor_modality][sensor_id] = []
            with open(
                    os.path.join(data_split_root, "test_full", sensor_modality, sensor_id, "file_names.txt"), "r"
            ) as f:
                file_names = f.readlines()
                # stip all file names
                file_names = [file_name.strip() for file_name in file_names]
                file_names_test_set[sensor_modality][sensor_id] = file_names
    return file_names_test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_dataset",
        type=str,
        help="Folder path to dataset root that contains all sequence data.",
        default="",
    )
    parser.add_argument(
        "--output_folder_path_dataset_without_test_set",
        type=str,
        help="Folder path to dataset that will not contain the test set.",
        default="",
    )
    parser.add_argument(
        "--output_folder_path_test_set",
        type=str,
        help="Folder path to dataset that will not contain the test set.",
        default="",
    )
    args = parser.parse_args()
    input_folder_path_dataset = args.input_folder_path_dataset
    output_folder_path_dataset_without_test_set = args.output_folder_path_dataset_without_test_set
    output_folder_path_test_set = args.output_folder_path_test_set
    subsets = ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]
    sensor_modalities = ["point_clouds", "images", "labels_point_clouds"]
    camera_sensors = ["s110_camera_basler_south2_8mm", "s110_camera_basler_south1_8mm"]
    lidar_sensors = ["s110_lidar_ouster_south", "s110_lidar_ouster_north"]
    data_split_root = str(pathlib.Path(__file__).parent.parent.parent.parent / "data_split")

    file_names_test_set = read_file_names(
        data_split_root,
    )

    # iterate all four sequences in dataset root
    for subset in tqdm(subsets):
        # iterate all sensor modalities
        for sensor_modality in tqdm(sensor_modalities):
            # iterate all sensors
            if sensor_modality == "images":
                sensors = camera_sensors
            elif sensor_modality == "point_clouds":
                sensors = lidar_sensors
            elif sensor_modality == "labels_point_clouds":
                sensors = lidar_sensors
            else:
                raise ValueError("Sensor modality not supported.")

            for sensor in tqdm(sensors):
                # iterate all files in sensors folder and check whether it belong to test set
                # if the file does not belong to test set, then copy it to output folder
                input_folder_path = os.path.join(input_folder_path_dataset, subset, sensor_modality, sensor)
                file_paths_sub_set = sorted(glob.glob(input_folder_path + "/*"))
                for file_path_sub_set in file_paths_sub_set:
                    file_name = os.path.basename(file_path_sub_set)
                    if file_name in file_names_test_set[sensor_modality][sensor]:
                        # save to test output folder
                        output_folder_path = os.path.join(output_folder_path_test_set, subset,
                                                          sensor_modality, sensor)
                        os.makedirs(output_folder_path, exist_ok=True)
                        input_file_path = os.path.join(input_folder_path, file_name)
                        output_file_path = os.path.join(output_folder_path, file_name)
                        os.system(f"cp {input_file_path} {output_file_path}")
                    # else:
                    # save to train/val output folder
                    # TODO: temp commented
                    # output_folder_path = os.path.join(output_folder_path_dataset_without_test_set, subset, sensor_modality, sensor)
                    # os.makedirs(output_folder_path, exist_ok=True)
                    # # copy file to output folder
                    # input_file_path = os.path.join(input_folder_path, file_name)
                    # output_file_path = os.path.join(output_folder_path, file_name)
                    # os.system(f"cp {input_file_path} {output_file_path}")
