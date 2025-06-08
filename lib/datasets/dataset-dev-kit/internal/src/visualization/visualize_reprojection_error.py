import argparse
import os
import numpy as np
import cv2


def project(position_3d, projection_matrix):
    position_3d_homogeneous = np.array([position_3d[0], position_3d[1], position_3d[2], 1])
    points = np.matmul(projection_matrix, position_3d_homogeneous)
    if points[2] > 0:
        pos_x = int(points[0] / points[2])
        pos_y = int(points[1] / points[2])
        # if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
        return [pos_x, pos_y]
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--input_file_path_image",
        default="images",
        help="Input file path to image. Default: images/image.png",
    )
    argparser.add_argument(
        "--input_file_path_corners_2d",
        help="Input file path to 2D corners. Default: corners_2d",
    )
    argparser.add_argument(
        "--input_file_path_corners_3d",
        help="Input file path to 3D corners. Default: corners_3d",
    )
    argparser.add_argument(
        "--output_file_path_image",
        help="Output file path of visualization. Default: output/output.png",
    )

    args = argparser.parse_args()
    input_file_path_image = args.input_file_path_image
    input_file_path_corners_2d = args.input_file_path_corners_2d
    input_file_path_corners_3d = args.input_file_path_corners_3d
    output_file_path_image = args.output_file_path_image
    error_total = 0

    # TODO: load projection matrix from file (a9 dataset)
    # projection matrix s110_lidar_ouster_south to s110_camera_basler_north_8mm
    # 20 correspondences (10.73 px reprojection error)
    # projection_matrix = np.array(
    #     [
    #         [-185.2891049687059, -1504.063395597006, -525.9215327879701, -23336.12843138125],
    #         [-240.2665682659353, 220.6722195428702, -1567.287260600104, 6362.243306159624],
    #         [0.6863989233970642, -0.4493367969989777, -0.5717979669570923, -6.750176429748535],
    #     ]
    # )
    # R3, projection matrix s110_lidar_ouster_south to s110_camera_basler_south1_8mm, 20 correspondences, 11 px reprojection error
    # projection_matrix = np.array(
    #     [
    #         [1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
    #         [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
    #         [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875],
    #     ]
    # )

    # R3, projection matrix s110_lidar_ouster_south to s110_camera_basler_east_8mm, 22 correspondences, 400 px reprojection error
    # projection_matrix = np.array(
    #     [
    #         [-2666.70160799, -655.44528859, -790.96345758, -33010.77350141],
    #         [430.89231274, 66.06703744, -2053.70223986, 6630.65222157],
    #         [-0.00932524, -0.96164431, -0.27414094, 11.41820108],
    #     ]
    # )
    # R3, projection matrix s110_lidar_ouster_south to s110_camera_basler_east_8mm, 22 correspondences, 400 px reprojection error
    # projection_matrix = np.array(
    #     [
    #         [1279.2768504, -862.92893566, -443.65575713, -16164.33154748],
    #         [-57.00779242, -67.92430641, -1461.78659387, -806.92191484],
    #         [0.790127, 0.342818, -0.508109, 3.67868],
    #     ]
    # )
    # R3, projection matrix vehicle_lidar_robosense to vehicle_camera_basler_16mm, 15 correspondences
    projection_matrix = np.array(
        [[1030.352144979763, -2677.507853013934, 214.2943474545664, 478.1953176491571],
         [589.9235492355729, -39.65458848865501, -2604.420668353988, -14.43355243962748],
         [0.9855244159698486, 0.07738696783781052, 0.1508407741785049, -0.01369299273937941]]
    )
    img = cv2.imread(os.path.join(input_file_path_image))
    # read 2d corners from txt file
    with open(input_file_path_corners_2d, "r") as f:
        data = f.readlines()
        corners_2d = []
        for line in data:
            line = line.rstrip("\n").split(" ")
            if len(line) == 2:
                corners_2d.append([float(line[0]), float(line[1])])
        corners_2d = np.asarray(corners_2d)

    # read 3d corners from txt file
    with open(input_file_path_corners_3d, "r") as f:
        data = f.readlines()
        corners_3d = []
        for line in data:
            line = line.rstrip("\n").split(" ")
            if len(line) == 3:
                corners_3d.append([float(line[0]), float(line[1]), float(line[2])])
        corners_3d = np.asarray(corners_3d)

    # project 3d corners to 2d corners
    corners_2d_projected = []
    for corner_3d in corners_3d:
        corner_2d = project(corner_3d, projection_matrix)
        corners_2d_projected.append(corner_2d)

    # draw 2d corners
    for corner_2d in corners_2d:
        cv2.circle(img, (int(corner_2d[0]), int(corner_2d[1])), 5, (0, 0, 255), -1)
    # draw 2d corners projected
    for corner_2d_projected in corners_2d_projected:
        cv2.circle(img, (int(corner_2d_projected[0]), int(corner_2d_projected[1])), 5, (0, 255, 0), -1)
    # draw lines between 2d corners and 2d corners projected
    for i in range(len(corners_2d)):
        cv2.line(
            img,
            (int(corners_2d[i][0]), int(corners_2d[i][1])),
            (int(corners_2d_projected[i][0]), int(corners_2d_projected[i][1])),
            (255, 0, 0),
            2,
        )
        error = np.sqrt(
            (corners_2d[i][0] - corners_2d_projected[i][0]) ** 2 + (corners_2d[i][1] - corners_2d_projected[i][1]) ** 2
        )
        error_total += error
    error_total = error_total / len(corners_2d)
    print("Average reprojection error: ", error_total)
    cv2.imwrite(os.path.join(output_file_path_image), img)
