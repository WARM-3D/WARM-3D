#!/bin/bash

# Example:
#bash internal/src/converter/convert_pcd_binary_to_ascii.sh <INPUT_FOLDER_PATH_PCD_BINARY> <OUTPUT_FOLDER_PATH_PCD_ASCII>

# NOTE: to use pcl_convert_pcd_ascii_binary you first need to install the pcl-tools package:
# sudo apt install pcl-tools

INPUT_FOLDER_PATH_PCD_BINARY=$1
OUTPUT_FOLDER_PATH_PCD_ASCII=$2

# convert all sensors in input folder
#for sensor_id in ${INPUT_FOLDER_PATH_PCD_BINARY}/*/; do
#  sensor_id=${sensor_id%*/} #remove trailing slash
#  sensor_id="${sensor_id##*/}"
#  # create output folder if it does not exist
#  mkdir -p ${OUTPUT_FOLDER_PATH_PCD_ASCII}/${sensor_id}
#  for file_path in ${INPUT_FOLDER_PATH_PCD_BINARY}/${sensor_id}/*.pcd; do
#    file_name="${file_path##*/}"
#    pcl_convert_pcd_ascii_binary ${file_path} ${OUTPUT_FOLDER_PATH_PCD_ASCII}/${sensor_id}/${file_name} 0
#  done
#done

# convert current input folder
  # create output folder if it does not exist
for file_path in ${INPUT_FOLDER_PATH_PCD_BINARY}/*.pcd; do
  file_name="${file_path##*/}"
  pcl_convert_pcd_ascii_binary ${file_path} ${OUTPUT_FOLDER_PATH_PCD_ASCII}/${file_name} 0
done
