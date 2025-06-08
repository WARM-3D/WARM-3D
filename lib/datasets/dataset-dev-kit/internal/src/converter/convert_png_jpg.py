import argparse
import glob
import os
import cv2

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_folder_path", type=str, help="Path to input folder", default="")
    argparser.add_argument("--output_folder_path", type=str, help="Path to output folder", default="")
    args = argparser.parse_args()

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    # read all images from input folder and convert them from png to jpg
    for file_path in sorted(glob.glob(args.input_folder_path + "/*.png")):
        image = cv2.imread(file_path)
        # save image as jpg
        file_name = os.path.basename(file_path)
        cv2.imwrite(os.path.join(args.output_folder_path, file_name.replace(".png", ".jpg")), image)
