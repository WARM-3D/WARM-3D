import numpy as np

if __name__ == '__main__':
    # non-optimal intrinsic 16mm
    intrinsic_matrix_camera = np.array(
        [[2788.86072, 0, 907.839058],
         [0, 2783.31261, 589.071478],
         [0, 0, 1]]
    )

    extrinsic_matrix_lidar_to_camera = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.23],
        [1.0, 0.0, 0.0, -0.01]
    ])
    projection_matrix = intrinsic_matrix_camera @ extrinsic_matrix_lidar_to_camera
    print(repr(projection_matrix))
