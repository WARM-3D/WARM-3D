# Author: Walter Zimmer
# Date: 2023-10-27
#
# This script extracts point clouds from a rosbag file and saves them as .pcd files.
# Example usage:
# python extract_point_clouds_from_rosbag.py --input_file_path_rosbag /home/user/Downloads/test.bag --output_folder_path_point_clouds /home/user/Downloads/pcd


import os
import argparse
import numpy as np

np.float = float
import pandas as pd
import rosbag
import ros_numpy


def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[..., 0] = np.round(cloud_array['x'], 6)
    points[..., 1] = np.round(cloud_array['y'], 6)
    points[..., 2] = np.round(cloud_array['z'], 6)
    points[..., 3] = np.round(cloud_array['intensity'], 6)
    return points


def convert_pc_msg_to_np(pc_msg):
    point_cloud_array_source = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
    pc_np = get_xyzi_points(point_cloud_array_source, True)
    return pc_np


def write_point_cloud_with_intensities(output_file_path, point_cloud_array, header=None):
    # update num points
    if header is not None:
        header[2] = "FIELDS x y z intensity rgb\n"
        header[3] = "SIZE 4 4 4 4 4\n"
        header[4] = "TYPE F F F F F\n"
        header[5] = "COUNT 1 1 1 1 1\n"
        header[6] = "WIDTH " + str(len(point_cloud_array)) + "\n"
        header[7] = "HEIGHT 1" + "\n"
        header[9] = "POINTS " + str(len(point_cloud_array)) + "\n"
    else:
        # create pcd header
        header = ["# .PCD v0.7 - Point Cloud Data file format\n",
                  "VERSION 0.7\n",
                  "FIELDS x y z intensity\n",
                  "SIZE 4 4 4 4\n",
                  "TYPE F F F F\n",
                  "COUNT 1 1 1 1\n",
                  "WIDTH " + str(len(point_cloud_array)) + "\n",
                  "HEIGHT 1" + "\n",
                  "VIEWPOINT 0 0 0 1 0 0 0\n",
                  "POINTS " + str(len(point_cloud_array)) + "\n",
                  "DATA ascii\n"]
    with open(output_file_path, 'w') as writer:
        for header_line in header:
            writer.write(header_line)
    df = pd.DataFrame(point_cloud_array)
    # round values in data frame to 6 decimal places
    df = df.round(6)
    df.to_csv(output_file_path, sep=" ", header=False, mode='a', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_file_path_rosbag', type=str, help='path to input rosbag file')
    argparser.add_argument('--output_folder_path_point_clouds', type=str, help='path to output folder for point clouds')
    args = argparser.parse_args()
    input_file_path_rosbag = args.input_file_path_rosbag
    output_folder_path_point_clouds = args.output_folder_path_point_clouds
    # create output folder
    if not os.path.exists(output_folder_path_point_clouds):
        os.makedirs(output_folder_path_point_clouds)

    # read rosbag
    bag = rosbag.Bag(input_file_path_rosbag)
    # iterate all messages
    for topic, msg, timestamp_nano in bag.read_messages():
        if topic == "/ouster/points":
            timestamp_topic_seconds_int = int(timestamp_nano.to_sec())
            timestamp_topic_nano_remainder = int(timestamp_nano.to_nsec() % 1000000000)
            timestamp_topic_in_nanoseconds = int(
                timestamp_topic_seconds_int * 1000000000) + timestamp_topic_nano_remainder
            print("=====================================")
            print("topic: " + topic + ", timestamp: " + str(timestamp_topic_seconds_int) + "_" + str(
                timestamp_topic_nano_remainder))
            # print msg.header timestamp
            timestamp_header_seconds_int = int(msg.header.stamp.secs)
            timestamp_header_nano_remainder_int = int(msg.header.stamp.nsecs)
            timestamp_header_in_nanoseconds = int(
                timestamp_header_seconds_int * 1000000000) + timestamp_header_nano_remainder_int
            print("msg.header.stamp: " + str(timestamp_header_seconds_int) + "_" + str(
                timestamp_header_nano_remainder_int))

            # calculate difference between topic and msg.header timestamp
            timestamp_difference_in_nanoseconds = int(timestamp_topic_in_nanoseconds - timestamp_header_in_nanoseconds)
            timestamp_difference_in_secs = int(timestamp_difference_in_nanoseconds / 1000000000)
            timestamp_difference_nano_remainder = timestamp_difference_in_nanoseconds % 1000000000
            print("timestamp_difference: " + str(timestamp_difference_in_secs) + "_" + str(
                timestamp_difference_nano_remainder))
            # convert point cloud ROS msg to pcl PointCloud2
            pc_np = convert_pc_msg_to_np(msg)
            # normalize intensity values
            pc_np[:, 3] = pc_np[:, 3] / np.max(pc_np[:, 3])
            # save point cloud to hard drive as .pcd (ascii) using open3d
            output_file_name = str(timestamp_topic_seconds_int) + "_" + str(timestamp_topic_nano_remainder).zfill(
                9) + "_s110_lidar_ouster_south.pcd"
            write_point_cloud_with_intensities(
                os.path.join(output_folder_path_point_clouds, output_file_name), pc_np)
