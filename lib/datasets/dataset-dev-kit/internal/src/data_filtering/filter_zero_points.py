import argparse
import os
import pandas as pd
import numpy as np

argparser = argparse.ArgumentParser(description="Filter point cloud data from lidar sensor.")
argparser.add_argument("-i", "--input_path", metavar="I", default="input", help="Path to the input folder")
argparser.add_argument("-o", "--output_path", metavar="O", default="output", help="Path to the output folder")
args = argparser.parse_args()
output_folder_path = args.output_path
input_folder_path = args.input_path

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

for file_name in sorted(os.listdir(input_folder_path)):
    header_lines = []
    data_lines = []
    num_points = 0
    with open(os.path.join(input_folder_path, file_name), "r") as file_reader:
        lines = file_reader.readlines()
        for idx, line in enumerate(lines):
            if idx <= 10:
                header_lines.append(line)
            else:
                parts = line.strip().split(" ")
                if len(parts) == 9:
                    # ouster lidar
                    if parts[0] == "0" and parts[1] == "0" and parts[2] == "0":
                        continue
                elif len(parts) == 4:
                    # robosense lidar
                    if parts[0] == "nan" and parts[1] == "nan" and parts[2] == "nan" and parts[3] == "0":
                        continue
                else:
                    raise Exception("unknown point format")
                data_lines.append(line)
    num_points = len(data_lines)
    # update width and num. points
    header_lines[6] = "WIDTH " + str(num_points) + "\n"
    header_lines[7] = "HEIGHT 1 \n"
    header_lines[9] = "POINTS " + str(num_points) + "\n"
    # write lines
    output_file_name_no_ext = os.path.splitext(file_name)[0]
    output_file_name = output_file_name_no_ext.replace(".", "_") + ".pcd"
    with open(os.path.join(output_folder_path, output_file_name), "w") as file_writer:
        for line in header_lines:
            file_writer.write(line)
        for line in data_lines:
            file_writer.write(line)
