import os

input_path = '/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/09_R1_S9/04_point_clouds/s110_lidar_ouster_south/'
output_path = '/mnt/hdd_data2/28_datasets/00_a9_dataset/01_R1_sequences/09_R1_S9/04_point_clouds/s110_lidar_ouster_south_corrected/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

for file in sorted(os.listdir(input_path)):
    header_lines = []
    data_lines = []
    num_points = 0
    with open(os.path.join(input_path, file), "r") as file_reader:
        lines = file_reader.readlines()
        for idx, line in enumerate(lines):
            if idx <= 10:
                header_lines.append(line)
            else:
                data_lines.append(line)
            if idx == 9:
                num_points = int(line.split(" ")[1])
    # update width and height
    header_lines[6] = "WIDTH " + str(num_points) + "\n"
    header_lines[7] = "HEIGHT 1\n"
    # write lines
    with open(os.path.join(output_path, file), "w") as file_writer:
        for line in header_lines:
            file_writer.write(line)
        for line in data_lines:
            file_writer.write(line)
