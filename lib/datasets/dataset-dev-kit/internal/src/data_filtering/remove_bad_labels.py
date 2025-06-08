import argparse
import glob
import os

# dataset cleaning:
#
# 1) manual inspection -> check qualitative visualization of labels and IDs -> note IDs with bad labels and remove them
# 1) filter by dimensions
# 2) filter by location (e.g. traffic sign)
# 3) check jira board (issues)
# 5) use proAnno labeling tool to correct classes (OR: manually correct classes within json file)
#
#
# 5.1) s04_lidar_ouster_south sign labeled as pedestrian bug
# 5.2) s110_lidar_ouster_north, wrong object labeled:
#
# remove_wrong_labels_r01_s08
# start: 3070_061
# end: 3075_560
# start: 3077_061
# end: 3079_961
#
# 5.3)
# r01_s08_ouster_north_rotation_fix
# 3104_257 → quaternion is [0,0,0,0] not valid
import json
import sys

import numpy as np

dimension_values_mapping = {
    "CAR": [[2.0, 6.0], [1.5, 2.2], [1.2, 1.75]],  # length  # width  # height
    "VAN": [[2.5, 7.8], [1.6, 2.6], [1.75, 2.5]],
    "TRUCK": [[2.5, 4.0], [1.6, 3.5], [1.8, 4.0]],
    "TRAILER": [[2.5, 15.0], [1.6, 3.5], [1.8, 4.0]],
    "BUS": [[4.0, 15.0], [1.6, 3.5], [1.8, 4.0]],
    "BICYCLE": [[1.5, 2.5], [0.5, 1.5], [1.0, 2.0]],
    "PEDESTRIAN": [[0.3, 1.3], [0.3, 1.3], [1.0, 2.0]],
    "MOTORCYCLE": [[1.5, 2.5], [0.5, 1.5], [1.0, 2.0]],
    "EMERGENCY_VEHICLE": [[2.0, 15.0], [1.5, 3.5], [1.2, 4.0]],
    "OTHER": [[2.0, 20.0], [1.5, 4.0], [1.2, 4.0]],
}

# bad uuids_short in test set (merged lidar labels in south1 projected)
# ids_south1 = ["3a1", "16d", "7f4", "2ba"]

# bad uuids_short in test set (merged lidar labels in south2 projected)
# bdb only first 5 occurences
# ids_south2 = ["912", "bdb", "e2a", "e01", "516"]

# bad uuids_short in train set (original south lidar labels)
bad_uuids_south = {
    "a59c708b": {
        "start": "1688626532_044026624_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json",
        "end": "1688626541_943776817_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json",
    },
    # "8af110f4": {
    #     "start": "1688626895_140223818_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json",
    #     "end": "1688626899_940260020_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered",
    # },
    # "bdb": {
    #     "start": "1651673079_959346053_s110_lidar_ouster_south.json",
    #     "end": "1651673086_555100388_s110_lidar_ouster_south.json",
    # },
    # "1b5": {
    #     "start": "1651673147_357208657_s110_lidar_ouster_south.json",
    #     "end": "1651673147_357208657_s110_lidar_ouster_south.json",
    # },
    # "16d8": {
    #     "start": "1651673140_056908277_s110_lidar_ouster_south.json",
    #     "end": "1651673149_859458198_s110_lidar_ouster_south.json",
    # },
    # "7f4e": {
    #     "start": "1651673140_056908277_s110_lidar_ouster_south.json",
    #     "end": "1651673149_551364362_s110_lidar_ouster_south.json",
    # },
}


# bad uuids_short in train set (original north lidar labels)
# bad_uuids_north = {
#     "912": {
#         "start": "1651673050_054497235_s110_lidar_ouster_north.json",
#         "end": "1651673059_357982485_s110_lidar_ouster_north.json",
#     },
#     "e01": {
#         "start": "1651673160_050708878_s110_lidar_ouster_north.json",
#         "end": "1651673169_954070886_s110_lidar_ouster_north.json",
#     },
#     "3a11": {
#         "start": "1651673070_061175935_s110_lidar_ouster_north.json",
#         "end": "1651673075_560447451_s110_lidar_ouster_north.json",
#     },
#     "3a118157": {
#         "start": "1651673079_961918429_s110_lidar_ouster_north.json",
#         "end": "1651673079_961918429_s110_lidar_ouster_north.json",
#     },
#     "1da": {
#         "start": "1651673122_654947738_s110_lidar_ouster_north.json",
#         "end": "1651673122_654947738_s110_lidar_ouster_north.json",
#     },
#     "511f0fee": {  # bad uuid in r02_s01 north lidar
#         "start": "1646667337_841441557_s110_lidar_ouster_north.json",
#         "end": "1646667337_841441557_s110_lidar_ouster_north.json",
#     },
#     "d1868aa8": {  # bad uuid in r02_s01 north lidar (after tracking)
#         "start": "1646667327_143829313_s110_lidar_ouster_north.json",
#         "end": "1646667327_941832949_s110_lidar_ouster_north.json",
#     },
#     "22415819": {  # bad uuid in r02_s02 north lidar
#         "start": "1651673161_055366376_s110_lidar_ouster_north.json",
#         "end": "1651673169_756392277_s110_lidar_ouster_north.json",
#     },
#     "0adfc971": {  # bad uuid in r02_s03 north lidar
#         "start": "1651673080_257303218_s110_lidar_ouster_north.json",
#         "end": "1651673080_257303218_s110_lidar_ouster_north.json",
#     },
#     "00cbeb": {  # bad uuid in r02_s03 north lidar
#         "start": "1651673079_961346053_s110_lidar_ouster_north.json",
#         "end": "1651673080_257303218_s110_lidar_ouster_north.json",
#     },
#     "00cbeb6": {  # bad uuid in r02_s03 north lidar
#         "start": "1651673084_563209632_s110_lidar_ouster_north.json",
#         "end": "1651673084_563209632_s110_lidar_ouster_north.json",
#     },
# }
# bad uuids_short in train set (fused/merged south+north lidar labels)
# bad_uuids_fused = {
#     "1375c251": {
#         "start": "1651673050_054807704_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673059_849022272_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "912": {
#         "start": "1651673050_054807703_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673059_355479528_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "78d": {
#         "start": "1651673050_157887636_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673050_157887636_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "b2a": {
#         "start": "1651673052_555352309_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673059_950215382_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "668": {
#         "start": "1651673077_652163945_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673077_948146153_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "0ad": {
#         "start": "1651673079_959346053_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673080_254280069_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "bdb": {
#         "start": "1651673079_959346053_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673086_555100388_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "06f": {
#         "start": "1651673082_151183440_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673082_355618877_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "326": {
#         "start": "1651673089_560303798_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673089_961881019_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "f44": {
#         "start": "1651673095_749792588_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673095_749792588_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "f68": {
#         "start": "1651673096_452102370_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673096_548513834_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "63e": {
#         "start": "1651673101_950305777_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673102_051397238_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "983": {
#         "start": "1651673102_750055323_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673102_750055323_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "6b9": {
#         "start": "1651673135_060704289_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673135_060704289_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "c68": {
#         "start": "1651673136_960162211_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673138_160057053_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "fd6": {
#         "start": "1651673138_256533771_s110_lidar_ouster_south_and_north_merged",
#         "end": "1651673138_451939786_s110_lidar_ouster_south_and_north_merged",
#     },
#     "8fc": {
#         "start": "1651673137_657470504_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673138_160057053_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "1b5": {
#         "start": "1651673147_357208657_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673147_357208657_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "cac": {
#         "start": "1651673156_659465838_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673156_659465838_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "cac1": {
#         "start": "1651673151_859307223_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673152_260288650_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "e01": {
#         "start": "1651673160_064370252_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673169_960212879_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "e69": {
#         "start": "1651673056_155658197_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673057_754727301_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "3a11": {
#         "start": "1651673070_062381430_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673079_959346053_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "1da": {
#         "start": "1651673122_655252405_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673122_655252405_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "efb3": {
#         "start": "1651673123_057268748_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673123_649322956_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "8d35": {
#         "start": "1651673130_754776858_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673131_060391566_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "c68d": {
#         "start": "1651673132_559836707_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673132_947740404_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "eaf5": {
#         "start": "1651673141_158540632_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673141_158540632_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "16d8": {
#         "start": "1651673140_056908277_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673149_859458198_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "7f4e": {
#         "start": "1651673140_056908277_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673149_551364362_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "a9b6": {
#         "start": "1651673156_458573135_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673156_659465838_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "f983": {
#         "start": "1651673163_857991123_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673164_565164788_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "edb6": {
#         "start": "1651673169_156948619_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673169_960212879_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "74c": {
#         "start": "1651673067_651793773_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673067_651793773_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "74c6": {
#         "start": "1651673067_863525024_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673067_952588304_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "74c6f": {
#         "start": "1651673068_360526279_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673068_456081146_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "74c6fb": {
#         "start": "1651673068_753998775_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673068_850045230_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d098": {
#         "start": "1651673080_059525105_s110_lidar_ouster_south_and_north_merged",
#         "end": "1651673080_751611192_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d0985": {
#         "start": "1651673080_254280069_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673080_349070233_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac": {
#         "start": "1653330063_006138959_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330065_306645786_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac5": {
#         "start": "1653330066_100031266_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330066_100031266_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac55": {
#         "start": "1653330066_701712135_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330066_701712135_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac550": {
#         "start": "1653330063_104929881_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330064_404668166_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac5509": {
#         "start": "1653330065_802706385_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330065_802706385_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "2bac5509-": {
#         "start": "1653330066_902672512_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330066_902672512_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "feb9": {
#         "start": "1653330068_805999635_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330069_021056932_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "feb9b": {
#         "start": "1653330070_105049479_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1653330070_305237629_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d54": {
#         "start": "1651673060_054378646_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673060_150382297_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d54e": {
#         "start": "1651673060_359574696_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673060_359574696_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d54e8": {
#         "start": "1651673060_645868684_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673060_645868684_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "d54e8": {
#         "start": "1651673060_750285015_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673060_750285015_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "e4d8faa7": {
#         "start": "1651673152_558240384_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673152_558240384_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "9480950b": {
#         "start": "1651673080_059525105_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673169_960212879_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "00cbeb66": {
#         "start": "1651673084_557958816_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673084_557958816_s110_lidar_ouster_south_and_north_merged.json",
#     },
#     "f40": {
#         "start": "1651673090_756797591_s110_lidar_ouster_south_and_north_merged.json",
#         "end": "1651673084_557958816_s110_lidar_ouster_south_and_north_merged.json",
#     },
# }


def get_timestamp_by_filename(file_name):
    parts = file_name.split("_")
    timestamp_seconds = int(parts[0])
    timestamp_nano_seconds_remaining = int(parts[1])
    timestamp_nano_seconds = timestamp_seconds * 1e9 + timestamp_nano_seconds_remaining
    return timestamp_nano_seconds


def filter_by_dimensions(dimensions, category):
    # 1) filter by dimension
    # filter dimensions by min and max values
    if (
            dimensions[0] > dimension_values_mapping[category.upper()][0][0]
            or dimensions[0] < dimension_values_mapping[category.upper()][0][1]
    ):
        # length
        return False
    if (
            dimensions[0] > dimension_values_mapping[category.upper()][1][0]
            or dimensions[0] < dimension_values_mapping[category.upper()][1][1]
    ):
        # width
        return False
    if (
            dimensions[0] > dimension_values_mapping[category.upper()][2][0]
            or dimensions[0] < dimension_values_mapping[category.upper()][2][1]
    ):
        # height
        return False
    return True


def filter_by_location(location, category):
    # filter by location
    pass


def filter_by_uuid(uuid, input_folder_path_labels, timestamp_nano_seconds):
    # iterate all key value pairs in bad_uuids
    bad_uuids = None
    if "south" in input_folder_path_labels and not "north" in input_folder_path_labels:
        pass
        # bad_uuids = bad_uuids_south
        # bad_uuids.update(bad_uuids_fused)
    elif "north" in input_folder_path_labels and not "south" in input_folder_path_labels:
        pass
        # bad_uuids = bad_uuids_north
        # bad_uuids.update(bad_uuids_fused)
    elif "south" in input_folder_path_labels and "north" in input_folder_path_labels:
        bad_uuids = bad_uuids_south
        # bad_uuids = bad_uuids_fused
        # bad_uuids.update(bad_uuids_south)
        # bad_uuids.update(bad_uuids_north)
    else:
        print(
            "ERROR: No valid folder path given! Make sure the input folder path contains south or north within folder name. Exiting..."
        )
        sys.exit()
    for uuid_short, start_end_file_name in bad_uuids.items():
        if uuid_short in uuid:
            # found label that needs to be removed
            # check whether timestamp is in range
            start_file_name = start_end_file_name["start"]
            timestamp_nano_seconds_start = get_timestamp_by_filename(start_file_name)
            end_file_name = start_end_file_name["end"]
            timestamp_nano_seconds_end = get_timestamp_by_filename(end_file_name)
            if (
                    timestamp_nano_seconds >= timestamp_nano_seconds_start
                    and timestamp_nano_seconds <= timestamp_nano_seconds_end
            ):
                return True
    return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="dateset cleaning")
    argparser.add_argument(
        "--input_folder_path_labels",
        help="Input folder path to lidar labels in OpenLABEL format.",
    )
    argparser.add_argument(
        "--output_folder_path_labels",
        default="output",
        help="Output folder path.",
    )

    args = argparser.parse_args()

    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels

    # create output folder if not exists
    if not os.path.exists(output_folder_path_labels):
        os.makedirs(output_folder_path_labels)

    label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))

    for label_file_path in label_file_paths:
        valid_labels = {}
        labels = json.load(open(label_file_path))
        if "openlabel" in labels:
            num_detections_fused = 0
            for frame_idx, frame_obj in labels["openlabel"]["frames"].items():
                for box_uuid, box in frame_obj["objects"].items():
                    is_removed = False
                    category = box["object_data"]["type"]
                    if "cuboid" in box["object_data"]:
                        cuboid = box["object_data"]["cuboid"]["val"]
                        location = np.array([cuboid[0], cuboid[1], cuboid[2]])
                        category = box["object_data"]["type"]
                        dimensions = np.array([cuboid[7], cuboid[8], cuboid[9]])
                        # is_removed = filter_by_dimensions(dimensions, category)
                        # is_removed = is_removed or filter_by_location(location, category)
                        # extract file name from label file path
                        file_name = label_file_path.split("/")[-1]
                        # extract current timestamp from file name
                        timestamp_nano_seconds = get_timestamp_by_filename(file_name)
                        is_removed = is_removed or filter_by_uuid(
                            box_uuid, input_folder_path_labels, timestamp_nano_seconds
                        )
                        if is_removed:
                            print(label_file_path, ": Removing object with uuid: ", box_uuid)
                            continue
                        valid_labels[box_uuid] = box
                frame_obj["objects"] = valid_labels
            # write to file
            output_file_path = output_folder_path_labels + "/" + label_file_path.split("/")[-1]
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                json.dump(labels, outfile, indent=4)
