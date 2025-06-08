import glob
import os
import numpy as np
import open3d as o3d
import json
import argparse

import pandas as pd
from pypcd import pypcd

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path",
        type=str,
        help="Path to pcd files",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path",
        type=str,
        help="output Path to pcd files",
        default="",
    )
    args = arg_parser.parse_args()
    input_folder_path = args.input_folder_path
    # load all point cloud files
    input_pcd_files = sorted(glob.glob(input_folder_path + "/*.pcd"))
    for input_pcd_file_path in input_pcd_files:
        point_cloud_array = np.array(pd.read_csv(input_pcd_file_path, sep=" ", skiprows=11, dtype=float).values)[:, :4]

        xyz = point_cloud_array[:, :3]
        intensities = point_cloud_array[:, 3]

        header = []
        header.append("# .PCD v0.7 - Point Cloud Data file format\n")
        header.append("VERSION 0.7\n")
        header.append("FIELDS x y z intensity\n")
        header.append("SIZE 4 4 4 4\n")
        header.append("TYPE F F F F\n")
        header.append("COUNT 1 1 1 1\n")
        header.append("WIDTH " + str(len(point_cloud_array)) + "\n")
        header.append("HEIGHT 1" + "\n")
        header.append("VIEWPOINT 0 0 0 1 0 0 0\n")
        header.append("POINTS " + str(len(point_cloud_array)) + "\n")
        header.append("DATA ascii\n")

        with open(os.path.join(args.output_folder_path, os.path.basename(input_pcd_file_path)), "w") as writer:
            for header_line in header:
                writer.write(header_line)
        df = pd.DataFrame(point_cloud_array)
        df.to_csv(
            os.path.join(args.output_folder_path, os.path.basename(input_pcd_file_path)),
            sep=" ",
            header=False,
            mode="a",
            index=False,
        )
