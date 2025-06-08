import argparse
import glob
import os
import json


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder_path_labels", type=str, help="Path to labels input folder", default="")
    arg_parser.add_argument(
        "--input_folder_path_point_clouds", type=str, help="Folder Path to registered point clouds", default=""
    )
    arg_parser.add_argument("--input_folder_path_images1", type=str, help="Folder Path to south1 images", default="")
    arg_parser.add_argument("--input_folder_path_images2", type=str, help="Folder Path to south2 images", default="")
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    args = arg_parser.parse_args()
    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    label_file_paths = sorted(glob.glob(args.input_folder_path_labels + "/*.json"))
    image1_file_paths = sorted(glob.glob(args.input_folder_path_images1 + "/*.png"))
    image2_file_paths = sorted(glob.glob(args.input_folder_path_images2 + "/*.png"))
    point_cloud_file_paths = sorted(glob.glob(args.input_folder_path_point_clouds + "/*.pcd"))
    # iterate over all files in input folder
    for file_path_label, file_path_point_cloud, file_path_image1, file_path_image2 in zip(
            label_file_paths, point_cloud_file_paths, image1_file_paths, image2_file_paths
    ):
        file_name_label = os.path.basename(file_path_label)
        file_name_point_cloud = os.path.basename(file_path_point_cloud)
        file_name_image1 = os.path.basename(file_path_image1)
        file_name_image2 = os.path.basename(file_path_image2)

        # load json file
        data_json = json.load(open(file_path_label))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # remove point_cloud_file_name from frame_properties
            # frame_obj["frame_properties"].pop("point_cloud_file_name", None)
            if "point_cloud_file_name" in frame_obj["frame_properties"]:
                frame_obj["frame_properties"].pop("point_cloud_file_name")
                # del frame_obj["frame_properties"]["point_cloud_file_name"]
            if "point_cloud_file_names" in frame_obj["frame_properties"]:
                frame_obj["frame_properties"].pop("point_cloud_file_names")
                # del frame_obj["frame_properties"]["point_cloud_file_name"]
            if "image_file_names" in frame_obj["frame_properties"]:
                frame_obj["frame_properties"].pop("image_file_names")
                # del frame_obj["frame_properties"]["image_file_names"]
            frame_obj["frame_properties"]["point_cloud_file_names"] = [file_name_point_cloud]
            frame_obj["frame_properties"]["image_file_names"] = [file_name_image1, file_name_image2]

        # write json file
        with open(os.path.join(args.output_folder_path_labels, file_name_label), "w") as f:
            json.dump(data_json, f)
