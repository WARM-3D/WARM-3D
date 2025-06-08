import os

input_folder_path = "/home/walter/Downloads/mono3d-480/"
mapping = {
    "r04": {
        "min": 1646667310044372291,
        "max": 1646667340138817032,
        "south1": "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/r01_s04/03_c_images_anonymized_undistorted/s110_camera_basler_south1_8mm/",
        "south2": "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/r01_s04/03_c_images_anonymized_undistorted/s110_camera_basler_south2_8mm/"
    },
    "r05": {
        "min": 1646667395057589699,
        "max": 1646667425230744996,
        "south1": "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/r01_s05/03_c_images_anonymized_undistorted/s110_camera_basler_south1_8mm/",
        "south2": "/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/r01_s05/03_c_images_anonymized_undistorted/s110_camera_basler_south2_8mm/"
    },
    "r08": {
        "min": 1651673049964807393,
        "max": 1651673169956938940,
        "south1": "/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/r01_s08/03_c_images_anonymized_undistorted/s110_camera_basler_south1_8mm/",
        "south2": "/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/r01_s08/03_c_images_anonymized_undistorted/s110_camera_basler_south2_8mm/"
    },
    "r09": {
        "min": 1653330059125630248,
        "max": 1653330121048486640,
        "south1": "/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/r01_s09/03_a_images_jpeg/s110_camera_basler_south1_8mm",
        "south2": "/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/r01_s09/03_a_images_jpeg/s110_camera_basler_south2_8mm"
    }
}
for file in sorted(os.listdir(input_folder_path)):
    timestamp = file.split(".")[0]
    seconds = int(timestamp.split("_")[0])
    nano_seconds_remaining = int(timestamp.split("_")[1])
    nano_seconds = seconds * 1000000000 + nano_seconds_remaining
    key_found = None
    for key in mapping:
        if nano_seconds >= mapping[key]["min"] and nano_seconds <= mapping[key]["max"]:
            key_found = key
            break
    folder_path_south1 = mapping[key_found]["south1"]
    file_paths_south1 = [file for file in sorted(os.listdir(folder_path_south1))]
    found = False
    for file_path_south1 in file_paths_south1:
        if file.split(".")[0] in file_path_south1:
            found = True
    if found:
        # write sensor id to file name
        new_file_name = str(seconds) + "_" + str(nano_seconds_remaining) + "_s110_camera_basler_south1_8mm.json"
        os.rename(os.path.join(input_folder_path, file), os.path.join(input_folder_path, new_file_name))
    else:
        new_file_name = str(seconds) + "_" + str(nano_seconds_remaining) + "_s110_camera_basler_south2_8mm.json"
        os.rename(os.path.join(input_folder_path, file), os.path.join(input_folder_path, new_file_name))
