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
    arg_parser.add_argument(
        "--input_folder_path_labels1",
        type=str,
        help="Path to labels (merged by late fusion) with occlusion attribute",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels2", type=str, help="Path to labels (test sequence manually labeled)", default=""
    )
    arg_parser.add_argument("--output_folder_path_labels", type=str, help="Path to labels output folder", default="")
    args = arg_parser.parse_args()
    # create output folder
    if not os.path.exists(args.output_folder_path_labels):
        os.makedirs(args.output_folder_path_labels)

    label_file_paths1 = sorted(glob.glob(args.input_folder_path_labels1 + "/*.json"))
    label_file_paths2 = sorted(glob.glob(args.input_folder_path_labels2 + "/*.json"))

    # iterate over all files in input folder
    for file_path_label1, file_path_label2 in zip(label_file_paths1, label_file_paths2):
        file_name_label1 = os.path.basename(file_path_label1)
        file_name_label2 = os.path.basename(file_path_label2)

        # load json file
        data_json1 = json.load(open(file_path_label1))
        data_json2 = json.load(open(file_path_label2))

        # iterate over all objects in the test sequence
        for frame_id2, frame_obj2 in data_json2["openlabel"]["frames"].items():
            for object_track_id2, object_json2 in frame_obj2["objects"].items():
                # find the same object track id in the merged labels
                same_uuid_found = False
                for frame_id1, frame_obj1 in data_json1["openlabel"]["frames"].items():
                    for object_track_id1, object_json1 in frame_obj1["objects"].items():
                        if object_track_id1 == object_track_id2:
                            same_uuid_found = True
                            # set the occlusion level attribute
                            occlusion_attribute1 = get_attribute_by_name(
                                object_json1["object_data"]["cuboid"]["attributes"]["text"], "occlusion_level"
                            )
                            if occlusion_attribute1 is not None:
                                occlusion_attribute2 = get_attribute_by_name(
                                    object_json2["object_data"]["cuboid"]["attributes"]["text"], "occlusion_level"
                                )
                                if occlusion_attribute2 is None:
                                    if occlusion_attribute1["val"] == "":
                                        print("occlusion value is empty")
                                    object_json2["object_data"]["cuboid"]["attributes"]["text"].append(
                                        {"name": "occlusion_level", "val": occlusion_attribute1["val"]}
                                    )
                                else:
                                    if occlusion_attribute1["val"] == "":
                                        print("occlusion value is empty")
                                    occlusion_attribute2["val"] = occlusion_attribute1["val"]
                            break
                if not same_uuid_found:
                    print("same uuid not found")
        # write json file
        with open(os.path.join(args.output_folder_path_labels, file_name_label2), "w") as f:
            json.dump(data_json2, f)
