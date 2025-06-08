import argparse
import json
import os

import numpy as np
import open3d as o3d
import copy

#################
# Usage:
#################
# python 01_manual_registration.py --input_file_path_point_cloud_source robosense/1688625741_092755795.pcd  --input_file_path_point_cloud_target ouster/1688625741_046595374.pcd --output_folder_path_transformation_matrix transformations --output_folder_path_point_cloud_registered registered_point_clouds

from src.utils.point_cloud_registration_utils import (
    read_point_cloud_with_intensity,
    write_point_cloud_with_intensity,
    filter_point_cloud,
)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    color_blue = (94, 199, 255)
    color_blue_normalized = [x / 255 for x in color_blue]
    target_temp.paint_uniform_color(color_blue_normalized)
    # target_temp.paint_uniform_color([0, 0.651, 0.929]) #
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # set background color to gray
    vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
    # vis.add_geometry([source_temp, target_temp])
    vis.add_geometry(source_temp)
    # add target point cloud
    vis.add_geometry(target_temp)
    vis.run()  # user picks points
    vis.destroy_window()
    # o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    # set background color to gray
    vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def merge_point_clouds(source, intensities_source, target, intensities_target, transformation_matrix):
    source.transform(transformation_matrix)
    source += target
    # merge intensities
    intensities_merged = np.concatenate((intensities_source, intensities_target))
    return source, intensities_merged


def manual_registration(
        input_file_path_point_cloud_source,
        input_file_path_point_cloud_target,
        output_folder_path_point_clouds_registered,
        output_folder_path_transformation_matrices,
):
    point_cloud_array_source, header_source = read_point_cloud_with_intensity(input_file_path_point_cloud_source)
    # normalize intensities
    point_cloud_array_source[:, 3] *= 1 / point_cloud_array_source[:, 3].max()
    point_cloud_array_source = filter_point_cloud(point_cloud_array_source, 200)
    intensities_source = point_cloud_array_source[:, 3]

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_cloud_array_source[:, :3])

    point_cloud_array_target, header_target = read_point_cloud_with_intensity(input_file_path_point_cloud_target)
    # normalize intensities
    point_cloud_array_target[:, 3] *= 1 / point_cloud_array_target[:, 3].max()
    point_cloud_array_target = filter_point_cloud(point_cloud_array_target, 200)
    intensities_target = point_cloud_array_target[:, 3]

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point_cloud_array_target[:, :3])

    print("Visualization of two point clouds before manual alignment")
    print("0) Please inspect both point clouds and press Q to continue.")

    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert len(picked_id_source) >= 3 and len(picked_id_target) >= 3
    assert len(picked_id_source) == len(picked_id_target)
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_matrix = reg_p2p.transformation
    inlier_rmse = reg_p2p.inlier_rmse
    fitness = reg_p2p.fitness
    # set nd correspondence_set size
    print("inlier_rmse: ", inlier_rmse)
    print("fitness: ", fitness)
    # save transformation matrix to json file
    transformation_matrix_json = {"transformation_matrix": transformation_matrix.tolist()}
    json.dump(transformation_matrix_json, open(
        os.path.join(output_folder_path_transformation_matrices,
                     os.path.basename(input_file_path_point_cloud_source).replace(".pcd", ".json")),
        "w"), indent=4)

    draw_registration_result(source, target, transformation_matrix)

    pcd_merged, intensities_merged = merge_point_clouds(
        source, intensities_source, target, intensities_target, transformation_matrix
    )

    # add intensities to merged point cloud
    point_cloud_merged = np.concatenate(
        (pcd_merged.points, intensities_merged.reshape((len(intensities_merged), 1))), axis=1
    )
    input_file_name_point_cloud_source = os.path.basename(input_file_path_point_cloud_source)
    input_file_name_point_cloud_source_without_extension = os.path.splitext(input_file_name_point_cloud_source)[0]
    # remove timestamp in front file name
    input_file_name_point_cloud_source_without_extension = input_file_name_point_cloud_source_without_extension[21:]

    input_file_name_point_cloud_target = os.path.basename(input_file_path_point_cloud_target)
    input_file_name_point_cloud_target_without_extension = os.path.splitext(input_file_name_point_cloud_target)[0]

    output_file_name_point_cloud_merged = input_file_name_point_cloud_target_without_extension + "_and_" + input_file_name_point_cloud_source_without_extension + ".pcd"
    write_point_cloud_with_intensity(
        os.path.join(output_folder_path_point_clouds_registered, output_file_name_point_cloud_merged),
        point_cloud_merged, header_source)


if __name__ == "__main__":
    print("Manual registration")
    argparser = argparse.ArgumentParser(description="Demo for manual ICP using correspondences")
    argparser.add_argument(
        "--input_file_path_point_cloud_source",
        type=str,
        default="data/point_clouds/point_cloud_source.ply",
        help="Path to the point cloud source",
    )
    argparser.add_argument(
        "--input_file_path_point_cloud_target",
        type=str,
        default="data/point_clouds/point_cloud_target.ply",
        help="Path to the point cloud target",
    )
    argparser.add_argument(
        "--output_folder_path_point_clouds_registered",
        type=str,
        default="data/point_clouds/",
        help="Folder path to the point cloud registered",
    )
    argparser.add_argument(
        "--output_folder_path_transformation_matrices",
        type=str,
        default="data/point_clouds/",
        help="Output folder path to the transformation matrix",
    )
    args = argparser.parse_args()
    # create output folder if not exists
    if not os.path.exists(args.output_folder_path_point_clouds_registered):
        os.makedirs(args.output_folder_path_point_clouds_registered)
    manual_registration(
        args.input_file_path_point_cloud_source,
        args.input_file_path_point_cloud_target,
        args.output_folder_path_point_clouds_registered,
        args.output_folder_path_transformation_matrices,
    )
