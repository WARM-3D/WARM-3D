import struct

import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_file_path_point_cloud",
        type=str,
        help="Path to point cloud",
        default="input.bin",
    )
    args = arg_parser.parse_args()
    pose_s110_base = np.array(
        [
            [0.9999999999999998, 0.0, 0.0, 0.0],
            [0.0, 0.9999999999999998, 0.0, 0.0],
            [0.0, 0.0, 0.9999999999999998, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformation_s110_base_to_s110_lidar_ouster_south = np.array(
        [
            [0.21479485, -0.9761028, 0.03296187, -15.87257873],
            [0.97627128, 0.21553835, 0.02091894, 2.30019086],
            [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    s110_base_to_lidar_ouster_north = np.array(
        [
            [-0.06821837, -0.997359, 0.0249256, -2.02963586],
            [0.99751775, -0.06774959, 0.01919179, 0.56416412],
            [-0.01745241, 0.02617296, 0.99950507, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose_camera_south2 = np.array(
        [
            [0.89247588, 0.29913535, -0.33764603, -19.30710024],
            [0.45096262, -0.60979520, 0.65175343, 5.27535533],
            [-0.01093244, -0.73393995, -0.67912637, 6.37107736],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
    )

    lidar_south_to_camera_south2 = np.asarray(
        [
            [0.49709212, -0.19863714, 0.64202357, -0.03734614],
            [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
            [0.01173726, -0.77546627, -0.70523436, 0.54322305],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # R2
    # lidar_south_to_camera_south1 = np.asarray(
    #     [
    #         [-0.10087585, -0.51122875, 0.88484734, 1.90816304],
    #         [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
    #         [0.01956882, -0.93122171, -0.45454375, 0.72290242],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ],
    #     dtype=np.float32,
    # )

    # R3
    camera_south1_to_lidar_south = np.asarray(
        [
            [-0.41205, 0.910783, -0.0262516, 15.0787],
            [0.453777, 0.230108, 0.860893, 2.52926],
            [0.790127, 0.342818, -0.508109, 3.67868],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    lidar_south_to_camera_south1 = np.linalg.inv(camera_south1_to_lidar_south)
    # R3
    camera_north_to_lidar_south = np.array(
        [
            [0.564606, 0.824831, 0.0295749, 12.9357],
            [0.458378, -0.343161, 0.819836, -7.22675],
            [0.686374, -0.449328, -0.571835, -6.75106],
            [0, 0, 0, 1],
        ]
    )
    lidar_south_to_camera_north = np.linalg.inv(camera_north_to_lidar_south)

    # R3
    camera_east_to_lidar_south = np.array(
        [
            [0.642324, -0.766428, -0.00297113, -1.87022],
            [0.74505, 0.625308, -0.232142, 13.4961],
            [0.179778, 0.146897, 0.972677, 6.99675],
            [0, 0, 0, 1],
        ]
    )
    lidar_south_to_camera_east = np.linalg.inv(camera_east_to_lidar_south)

    transformation_s110_camera_basler_east_8mm_to_s110_base = np.array(
        [
            [-0.9765868395975734, 0.21111945754584283, -0.04131246023141921, -15.577517063270534],
            [0.0963831031171547, 0.25771474278459316, -0.9614017936247983, 8.5],
            [-0.19232379509132863, -0.9428741623339884, -0.2720291746203862, 0.3380540568048218],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformation_s110_base_to_s110_camera_basler_east_8mm = np.linalg.inv(
        transformation_s110_camera_basler_east_8mm_to_s110_base
    )
    transformation_s110_lidar_ouster_south_to_s110_camera_basler_east_8mm = (
            transformation_s110_base_to_s110_camera_basler_east_8mm @ transformation_s110_base_to_s110_lidar_ouster_south
    )

    list_pcd = []
    size_float = 4

    if ".bin" in args.input_file_path_point_cloud:
        with open(args.input_file_path_point_cloud, "rb") as f:
            byte = f.read(size_float * 5)
            while byte:
                x, y, z, intensity, timestamp = struct.unpack("fffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 5)
        np_pcd = np.asarray(list_pcd)

    elif ".pcd" in args.input_file_path_point_cloud:
        pcd = o3d.io.read_point_cloud(args.input_file_path_point_cloud)
        np_pcd = np.asarray(pcd.points)
    else:
        raise Exception("Invalid file type")

    # make homogeneous
    np_pcd = np.hstack((np_pcd, np.ones((np_pcd.shape[0], 1))))
    # transform point cloud from s110_lidar_south coordinate system to s110_base coordinate system
    np_pcd = np_pcd @ transformation_s110_base_to_s110_lidar_ouster_south.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd[:, :3])

    # visualize pose in open3d
    s110_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    s110_frame.transform(pose_s110_base)

    lidar_south_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    lidar_south_frame.transform(transformation_s110_base_to_s110_lidar_ouster_south)

    lidar_north_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    lidar_north_frame.transform(s110_base_to_lidar_ouster_north)

    camera_south2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    s110_base_to_s110_camera_basler_south2 = (
            transformation_s110_base_to_s110_lidar_ouster_south @ lidar_south_to_camera_south2
    )
    camera_south2_frame.transform(s110_base_to_s110_camera_basler_south2)

    camera_south1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    s110_base_to_s110_camera_basler_south1 = (
            transformation_s110_base_to_s110_lidar_ouster_south @ lidar_south_to_camera_south1
    )
    camera_south1_frame.transform(s110_base_to_s110_camera_basler_south1)

    camera_north_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    s110_base_to_s110_camera_basler_north = (
            transformation_s110_base_to_s110_lidar_ouster_south @ lidar_south_to_camera_north
    )
    camera_north_frame.transform(s110_base_to_s110_camera_basler_north)

    camera_east_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # rotate lidar north by 180 degrees
    translation = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ]
    )
    # rotate roll by -180 degree
    rotation_roll_minus_180 = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate roll by -90 degree
    rotation_roll_minus_90 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate pitch by -90 degree
    rotation_pitch_minus_90 = np.array(
        [
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate pitch by +90 degree
    rotation_pitch_plus_90 = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate yaw by -90 degree
    rotation_yaw_minus_90 = np.array(
        [
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # rotate yaw by +90 degree
    rotation_yaw_plus_90 = np.array(
        [
            [0, -1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate yaw by +45 degree
    rotation_yaw_plus_45 = np.array(
        [
            [0.707, -0.707, 0, 0],
            [0.707, 0.707, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate yaw by +135 degree
    rotation_yaw_plus_135 = np.array(
        [
            [-0.707, -0.707, 0, 0],
            [0.707, -0.707, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate roll by 45 degree
    rotation_roll_plus_45 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.707, -0.707, 0],
            [0, 0.707, 0.707, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate roll by -45 degree
    rotation_roll_minus_45 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.707, 0.707, 0],
            [0, -0.707, 0.707, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate roll by -30 degree
    rotation_roll_minus_30 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.866, 0.5, 0],
            [0, -0.5, 0.866, 0],
            [0, 0, 0, 1],
        ]
    )

    # rotate pitch by 180 degree
    rotation_pitch_plus_180 = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    lidar_to_camera_frame = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    transformation_s110_base_to_s110_camera_basler_east_8mm = (
            transformation_s110_base_to_s110_lidar_ouster_south
            @ lidar_to_camera_frame
            @ rotation_yaw_plus_90
            @ rotation_pitch_plus_90
            @ rotation_roll_minus_30
            @ translation
    )

    camera_east_frame.transform(transformation_s110_base_to_s110_camera_basler_east_8mm)

    s110_camera_basler_south2_intrinsics = {
        "width": 1920,
        "height": 1200,
        "fx": 1400.3096617691212,
        "fy": 1403.041082755918,
        "cx": 967.7899705163408,
        "cy": 581.7195041357244,
    }

    s110_camera_basler_8mm_pinhole = o3d.camera.PinholeCameraIntrinsic(
        s110_camera_basler_south2_intrinsics["width"],
        s110_camera_basler_south2_intrinsics["height"],
        s110_camera_basler_south2_intrinsics["fx"],
        s110_camera_basler_south2_intrinsics["fy"],
        s110_camera_basler_south2_intrinsics["cx"],
        s110_camera_basler_south2_intrinsics["cy"],
    )

    s110_camera_basler_south2_8mm_frustum = o3d.geometry.LineSet.create_camera_visualization(
        s110_camera_basler_8mm_pinhole, extrinsic=np.linalg.inv(s110_base_to_s110_camera_basler_south2), scale=20
    )
    s110_camera_basler_south2_8mm_frustum.paint_uniform_color([0, 0, 0])

    s110_camera_basler_south1_8mm_frustum = o3d.geometry.LineSet.create_camera_visualization(
        s110_camera_basler_8mm_pinhole, extrinsic=np.linalg.inv(s110_base_to_s110_camera_basler_south1), scale=20
    )
    s110_camera_basler_south1_8mm_frustum.paint_uniform_color([0, 0, 0])

    s110_camera_basler_north_8mm_frustum = o3d.geometry.LineSet.create_camera_visualization(
        s110_camera_basler_8mm_pinhole, extrinsic=np.linalg.inv(s110_base_to_s110_camera_basler_north), scale=20
    )
    s110_camera_basler_north_8mm_frustum.paint_uniform_color([0, 0, 0])

    s110_camera_basler_east_8mm_frustum = o3d.geometry.LineSet.create_camera_visualization(
        s110_camera_basler_8mm_pinhole,
        extrinsic=np.linalg.inv(transformation_s110_base_to_s110_camera_basler_east_8mm),
        scale=20,
    )
    s110_camera_basler_east_8mm_frustum.paint_uniform_color([0, 0, 0])

    # camera_east_frame.transform(transformation_s110_base_to_s110_camera_basler_east_8mm)

    rotation_matrix_initial_camera_to_lidar = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float
    )
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix_initial_camera_to_lidar

    # camera_east_frame.transform(transformation)

    o3d.visualization.draw_geometries(
        [
            pcd,
            s110_frame,
            lidar_south_frame,
            lidar_north_frame,
            camera_south2_frame,
            camera_south1_frame,
            camera_north_frame,
            camera_east_frame,
            s110_camera_basler_south2_8mm_frustum,
            s110_camera_basler_south1_8mm_frustum,
            s110_camera_basler_north_8mm_frustum,
            s110_camera_basler_east_8mm_frustum,
        ]
    )

    # calculate the transformation matrix from s110_camera_basler_east to s110_lidar_ouster_south
    s110_camera_basler_east_to_s110_lidar_ouster_south = (
            np.linalg.inv(transformation_s110_base_to_s110_camera_basler_east_8mm)
            @ transformation_s110_base_to_s110_lidar_ouster_south
    )
    print("s110_camera_basler_east_to_s110_lidar_ouster_south=", s110_camera_basler_east_to_s110_lidar_ouster_south)

    s110_lidar_ouster_south_to_s110_camera_basler_east = np.linalg.inv(
        s110_camera_basler_east_to_s110_lidar_ouster_south
    )
    print("s110_lidar_ouster_south_to_s110_camera_basler_east=", s110_lidar_ouster_south_to_s110_camera_basler_east)

    print(
        "euler rotation values: ",
        R.from_matrix(s110_lidar_ouster_south_to_s110_camera_basler_east[:3, :3]).as_euler("xyz", degrees=True),
    )

    translation_vector_s110_camera_basler_east_8mm_hd_map = np.array([675.055, -191.907, 783.561])

    rotation_quaternion_hd_map_to_s110_camera_basler_east_8mm = np.array([0.388271, 0.486407, -0.61756, 0.48091])
    rotation_matrix_hd_map_to_s110_camera_basler_east_8mm = R.from_quat(
        rotation_quaternion_hd_map_to_s110_camera_basler_east_8mm
    ).as_matrix()

    # rotate translation vector
    translation_vector_hd_map_to_s110_camera_basler_east_8mm = np.matmul(
        rotation_matrix_hd_map_to_s110_camera_basler_east_8mm, translation_vector_s110_camera_basler_east_8mm_hd_map
    )
    print(
        "translation_vector_hd_map_to_s110_camera_basler_east_8mm = ",
        repr(translation_vector_hd_map_to_s110_camera_basler_east_8mm),
    )
    # [-354.93734093, -897.01619744, -419.37669934]

    transformation_hd_map_to_s110_camera_basler_east_8mm = np.eye(4)
    transformation_hd_map_to_s110_camera_basler_east_8mm[
    0:3, 0:3
    ] = rotation_matrix_hd_map_to_s110_camera_basler_east_8mm
    transformation_hd_map_to_s110_camera_basler_east_8mm[0:3, 3] = translation_vector_s110_camera_basler_east_8mm_hd_map
    print(
        "transformation_hd_map_to_s110_camera_basler_east_8mm = ",
        repr(transformation_hd_map_to_s110_camera_basler_east_8mm),
    )

    transformation_hd_map_to_s110_base = np.array(
        [[0.0, -1.0, -0.0, -433.63036414], [1.0, 0.0, 0.0, -169.78483588], [0.0, 0.0, 1.0, 5.08], [0.0, 0.0, 0.0, 1.0]]
    )

    transformation_s110_base_to_s110_camera_basler_east_8mm = np.matmul(
        transformation_hd_map_to_s110_base, transformation_hd_map_to_s110_camera_basler_east_8mm
    )
    print(
        "transformation_s110_base_to_s110_camera_basler_east_8mm = ",
        repr(transformation_s110_base_to_s110_camera_basler_east_8mm),
    )

    # transformation_s110_base_to_s110_camera_basler_east_8mm =  array([
    #        [ 2.16265895e-01,  6.42684661e-02,  9.74216930e-01, -2.41723364e+02],
    #        [-2.35943108e-01,  9.71696129e-01, -1.17252860e-02, 5.05270164e+02],
    #        [-9.47396386e-01, -2.27323991e-01,  2.25308435e-01, 7.88641000e+02],
    #        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
