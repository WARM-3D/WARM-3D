import glob
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    argparser = ArgumentParser(description='VizLabel Argument Parser')
    argparser.add_argument('-i', '--input_folder_path_labels')
    argparser.add_argument('-t', '--test_file_names')
    argparser.add_argument('-o', '--output_folder_path_labels')
    # parse
    args = argparser.parse_args()
    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels
    test_file_names = args.test_file_names

    # create output folder if it does not exist
    if not os.path.exists(output_folder_path_labels):
        os.makedirs(output_folder_path_labels)

    # read all file names from txt file
    with open(test_file_names) as f:
        test_file_names = f.readlines()

    # remove whitespace characters like `\n` at the end of each line
    # remove extension from file names
    # test_file_names = [os.path.splitext(x.strip())[0] for x in test_file_names]
    # use only timestamp
    test_file_names = [x.split('_')[0] + '_' + x.split('_')[1] for x in test_file_names]

    # get all input file paths using glob
    input_file_paths_labels = sorted(glob.glob(input_folder_path_labels + '/*.json'))
    # iterate all input labels and copy them to output folder if they are not in test_file_names_south2
    for input_file_path_label in input_file_paths_labels:
        # get file name from input file path
        file_name = os.path.basename(input_file_path_label)
        # remove file extension
        # file_name = os.path.splitext(file_name)[0]
        timestamp_str = file_name.split('_')[0] + '_' + file_name.split('_')[1]
        # check if file name is in test_file_names_south2
        if timestamp_str not in test_file_names:
            # copy file to output folder
            os.system('cp ' + input_file_path_label + ' ' + output_folder_path_labels)
