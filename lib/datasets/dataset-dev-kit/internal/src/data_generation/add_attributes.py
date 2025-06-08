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
    arg_parser.add_argument("--input_folder_path_images1", type=str, help="Folder Path to south1 images", default="")
    arg_parser.add_argument("--input_folder_path_images2", type=str, help="Folder Path to south2 images", default="")
    arg_parser.add_argument(
        "--input_folder_path_images3", type=str, help="Folder Path to north (8 mm) images", default=""
    )
    arg_parser.add_argument(
        "--input_folder_path_images4", type=str, help="Folder Path to east (8 mm) images", default=""
    )
    arg_parser.add_argument(
        "--input_folder_path_images_vehicle", type=str, help="Folder Path to south2 images", default=""
    )
    arg_parser.add_argument(
        "--input_folder_path_point_clouds_vehicle", type=str, help="Folder Path to vehicle point clouds", default=""
    )
    arg_parser.add_argument(
        "--input_folder_path_point_clouds_infrastructure",
        type=str,
        help="Folder Path to infrastructure point clouds",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_point_clouds_registered",
        type=str,
        help="Folder Path to registered point clouds",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_transforms",
        type=str,
        help="Folder Path to transformation matrices",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_with_transforms",
        type=str,
        help="Folder Path to labels with transformation matrices",
        default="",
    )
    arg_parser.add_argument("--sensor_id", type=str, help="Sensor ID", default="")
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    args = arg_parser.parse_args()
    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    label_file_paths = sorted(glob.glob(args.input_folder_path_labels + "/*.json"))
    image1_file_paths = sorted(
        glob.glob(args.input_folder_path_images1 + "/*.jpg") + glob.glob(args.input_folder_path_images1 + "/*.png")
    )
    image2_file_paths = sorted(
        glob.glob(args.input_folder_path_images2 + "/*.jpg") + glob.glob(args.input_folder_path_images2 + "/*.png")
    )
    image3_file_paths = sorted(
        glob.glob(args.input_folder_path_images3 + "/*.jpg") + glob.glob(args.input_folder_path_images3 + "/*.png")
    )
    image4_file_paths = sorted(
        glob.glob(args.input_folder_path_images4 + "/*.jpg") + glob.glob(args.input_folder_path_images4 + "/*.png")
    )
    image_vehicle_file_paths = sorted(
        glob.glob(args.input_folder_path_images_vehicle + "/*.jpg")
        + glob.glob(args.input_folder_path_images_vehicle + "/*.png")
    )
    point_cloud_vehicle_file_paths = sorted(glob.glob(args.input_folder_path_point_clouds_vehicle + "/*.pcd"))
    point_cloud_infrastructure_file_paths = sorted(
        glob.glob(args.input_folder_path_point_clouds_infrastructure + "/*.pcd")
    )
    point_cloud_registered_file_paths = sorted(glob.glob(args.input_folder_path_point_clouds_registered + "/*.pcd"))

    if args.input_folder_path_transforms != "":
        transforms_file_paths = sorted(glob.glob(args.input_folder_path_transforms + "/*.json"))
    else:
        transforms_file_paths = [""] * len(label_file_paths)

    if args.input_folder_path_labels_with_transforms != "":
        input_folder_path_labels_with_transforms = sorted(
            glob.glob(args.input_folder_path_labels_with_transforms + "/*.json"))
    else:
        input_folder_path_labels_with_transforms = [""] * len(label_file_paths)
    # iterate over all files in input folder
    for (
            file_path_label,
            file_path_image1,
            file_path_image2,
            file_path_image3,
            file_path_image4,
            file_path_image_vehicle,
            file_path_point_cloud_vehicle,
            file_path_point_cloud_infrastructure,
            file_path_point_cloud_registered,
            file_path_transform,
            file_path_label_with_transform,
    ) in zip(
        label_file_paths,
        image1_file_paths,
        image2_file_paths,
        image3_file_paths,
        image4_file_paths,
        image_vehicle_file_paths,
        point_cloud_vehicle_file_paths,
        point_cloud_infrastructure_file_paths,
        point_cloud_registered_file_paths,
        transforms_file_paths,
        input_folder_path_labels_with_transforms
    ):
        file_name_image1 = os.path.basename(file_path_image1)
        file_name_image2 = os.path.basename(file_path_image2)
        file_name_image3 = os.path.basename(file_path_image3)
        file_name_image4 = os.path.basename(file_path_image4)
        file_name_image_vehicle = os.path.basename(file_path_image_vehicle)
        file_name_point_cloud_vehicle = os.path.basename(file_path_point_cloud_vehicle)
        file_name_point_cloud_infrastructure = os.path.basename(file_path_point_cloud_infrastructure)
        file_name_point_cloud_registered = os.path.basename(file_path_point_cloud_registered)
        file_name_label = os.path.basename(file_path_label)

        # load transformation matrix from json file that contains only transformation matrix
        if file_path_transform != "":
            transform_json = json.load(open(file_path_transform))

        if file_path_label_with_transform != "":
            label_with_transform_json = json.load(open(file_path_label_with_transform))

        # load json file
        data_json = json.load(open(file_path_label))
        # iterate over all frames
        for frame_id, frame_obj in data_json["openlabel"]["frames"].items():
            # add frame_properties
            frame_obj["frame_properties"] = {}
            # add timestamp to frame properties
            # extract timestamp from file_name_point_cloud_registered file name
            file_name_point_cloud_registered_without_extension = os.path.splitext(file_name_point_cloud_registered)[0]
            timestamp_sec = int(file_name_point_cloud_registered_without_extension.split("_")[0])
            timestamp_nsec = int(file_name_point_cloud_registered_without_extension.split("_")[1])
            # calc timestamp in seconds with decimal point
            timestamp_sec_with_decimal_point = timestamp_sec + timestamp_nsec / 1e9
            frame_obj["frame_properties"]["timestamp"] = timestamp_sec_with_decimal_point
            frame_obj["frame_properties"]["image_file_names"] = [
                file_name_image1,
                file_name_image2,
                file_name_image3,
                file_name_image4,
                file_name_image_vehicle,
            ]
            frame_obj["frame_properties"]["point_cloud_file_names"] = [
                file_name_point_cloud_vehicle,
                file_name_point_cloud_infrastructure,
                file_name_point_cloud_registered,
            ]

            # add transforms
            if file_path_transform != "":
                frame_obj["frame_properties"]["transforms"] = {
                    "vehicle_lidar_robosense_to_s110_lidar_ouster_south": {
                        "src": "vehicle_lidar_robosense",
                        "dst": "s110_lidar_ouster_south",
                        "transform_src_to_dst": {
                            "matrix4x4": transform_json["transformation_matrix"]
                        }
                    }
                }
            if file_path_label_with_transform != "":
                for frame_id, label_with_transform_frame_obj in label_with_transform_json["openlabel"][
                    "frames"].items():
                    frame_obj["frame_properties"]["transforms"] = {
                        "vehicle_lidar_robosense_to_s110_lidar_ouster_south": {
                            "src": "vehicle_lidar_robosense",
                            "dst": "s110_lidar_ouster_south",
                            "transform_src_to_dst": {
                                "matrix4x4": label_with_transform_frame_obj["frame_properties"]["transforms"][
                                    "vehicle_lidar_robosense_to_s110_lidar_ouster_south"]["transform_src_to_dst"][
                                    "matrix4x4"]
                            }
                        }
                    }
                    break

            if args.sensor_id is not None:
                # iterate over all objects
                for object_id, label in frame_obj["objects"].items():
                    attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["text"], "sensor_id")
                    # iterate over all attributes
                    if attribute is None:
                        # add attribute
                        label["object_data"]["cuboid"]["attributes"]["text"].append(
                            {"name": "sensor_id", "val": args.sensor_id}
                        )
                    else:
                        # update attribute
                        attribute["val"] = args.sensor_id
                    # add score
                    # label["object_data"]["cuboid"]["attributes"]["num"].append({"name": "score", "val": -1.0})
                    # TODO: remove score attribute for labels (but keep it for detections)
                    # for attr in label["object_data"]["cuboid"]["attributes"]["num"]:
                    #     if attr["name"] == "score":
                    #         label["object_data"]["cuboid"]["attributes"]["num"].remove(attr)
                    #         break
                    # remove overlap attribute
                    for attr in label["object_data"]["cuboid"]["attributes"]["text"]:
                        if attr["name"] == "overlap":
                            label["object_data"]["cuboid"]["attributes"]["text"].remove(attr)
                            break

        # write json file
        with open(
                os.path.join(args.output_folder_path_labels,
                             file_name_point_cloud_registered.replace(".pcd", ".json")),
                "w") as f:
            json.dump(data_json, f)
