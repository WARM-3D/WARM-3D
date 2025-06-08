dataset = 's110_n'  # Options: 'kitti', 'a9', 'dair', 'a9_south1', 'a9_south2', 'snow'
result_type = 'ema'  # Options: 'gt', 'pred', 'pred_tum', 'ema', 'ema_proj', 'gt_proj'
epoch = 5

# Base directories for each dataset
base_dirs = {
    'kitti': '/data/carla_dataset_large/',
    'a9': '/data/tum_traffic_intersection_dataset_aligned/',
    'a9_south1': '/data/tum_traffic_intersection_dataset_south1/',
    'a9_south2': '/data/tum_traffic_intersection_dataset_south2/',
    'snow': "/data/tum_traffic_snow/",
    'dair': '/data/DAIR-V2X/single-infrastructure-side/',
    's110_n' : '/data/s110_n_dataset/',
}

# Mapping of result types to their label directories within each dataset
label_dirs = {
    'kitti': {
        'gt': lambda base: f'{base}training/label_2_aligned',
        'pred': '/workspace/outputs/monodetr_bl_carla_large_ext/outputs/data',
        'pred_tum': '/workspace/outputs/monodetr_bl_carla_large/outputs/data',
    },
    'a9': {
        'pred': '/workspace/outputs/monodetr_bl_tum/outputs/data',
        'gt': lambda base: f'{base}training/label_2',
        'ema': '/workspace/outputs/monodetr_ema_ext/outputs/data_target',
        'ema_proj': '/workspace/outputs/monodetr_ema/outputs/data_target_proj',
        'gt_proj': '/workspace/outputs/monodetr_ema/outputs/data_target_proj_gt',
        'visual': '/workspace/outputs/monodetr_ema_visual/outputs/data_target_epoch_' + str(epoch),
    },
    'dair': {
        'pred': '/workspace/outputs/monodetr_bl_tum/outputs/data',
        'ema': '/workspace/outputs/monodetr_dair/outputs/data_target',
        'ema_proj': '/workspace/outputs/monodetr_dair/outputs/data_target_proj',
        'gt_proj': '/workspace/outputs/monodetr_dair/outputs/data_target_proj_gt',
        'gt': lambda base: f'{base}training/label_2',
    },
    'a9_south1': {
        'gt': lambda base: f'{base}training/label_2',
        'tum_bl': '/workspace/outputs/monodetr_bl_tum/outputs/south1',
        'pred': '/workspace/outputs/monodetr_bl_carla_large_ext/outputs/south1',
        'ema': '/workspace/outputs/monodetr_ema_yolov9_0.3/outputs/south1',
    },
    'a9_south2': {
        'gt': lambda base: f'{base}training/label_2',
        'tum_bl': '/workspace/outputs/monodetr_bl_tum/outputs/south2',
        'pred': '/workspace/outputs/monodetr_bl_carla_large_ext/outputs/south2',
        'ema': '/workspace/outputs/monodetr_ema_yolov9_0.3/outputs/south2',
    },
    'snow': {
        'gt': lambda base: f'{base}training/label_2',
        'tum_bl': '/workspace/outputs/monodetr_bl_tum/outputs/snow',
        # 'tum_bl': '/workspace/outputs/monodetr_bl_tum/outputs/snow
        'pred': '/workspace/outputs/monodetr_ema_snow/outputs/snow_carla',
        'ema': '/workspace/outputs/monodetr_ema_snow/outputs/snow_ema',
    },
    's110_n': {
        'gt': lambda base: f'{base}training/label_2',
        'tum_bl': '/workspace/MonoDETR/outputs/warm3d_4class_4cam_1/outputs/data_target',
        # 'tum_bl': '/workspace/outputs/monodetr_bl_tum/outputs/snow
        'pred': '/workspace/MonoDETR/outputs/warm3d_4class_4cam_1/outputs/data_target_burn_in',
        'ema': '/workspace/MonoDETR/outputs/warm3d_4class_4cam_1/outputs/data_target',
    },
}

# Initialize base_dir and label_dir
base_dir = base_dirs.get(dataset, '')
result_path = label_dirs.get(dataset, {}).get(result_type, '')

# Check if the value is callable (i.e., a lambda function that expects base_dir as an argument)
if callable(result_path):
    label_dir = result_path(base_dir)
else:
    # If it's not callable, use the value directly as it's assumed to be a string
    label_dir = result_path

# Common directories
image_dir = f'{base_dir}training/image_2'
cal_dir = f'{base_dir}training/calib'

# Adjusted output directory mappings to account for different datasets
output_dir_mappings = {
    'kitti': {
        'gt': 'vis_gt_kitti',
        'pred': 'vis_pred_kitti',
        # Add more mappings for other result types as needed
    },
    'a9': {
        'gt': 'vis_gt_a9',
        'pred': 'train_on_tum_test_on_tum',
        'ema': 'vis_pred_ema_a9',
        'ema_proj': 'vis_pred_ema_proj_a9',
        'gt_proj': 'vis_gt_proj_a9',
        'visual': 'vis_ema_visual_' + str(epoch),
        # Add more mappings for other result types as needed
    },
    'dair': {
        'gt': 'vis_gt_dair',
        'pred': 'vis_pred_dair',
        'ema': 'vis_pred_ema_dair',
        'ema_proj': 'vis_pred_ema_proj_dair',
        'gt_proj': 'vis_gt_proj_dair',
        # Add more mappings for other result types as needed
    },
    'a9_south1': {
        'gt': 'vis_gt_south1',
        'tum_bl': 'train_on_tum_test_on_tum_south1',
        'pred': 'train_on_carla_test_on_tum_south1',
        'ema': 'train_on_carla_EMA_test_on_tum_south1',
        # Add more mappings for other result types as needed
    },
    'a9_south2': {
        'gt': 'vis_gt_south2',
        'tum_bl': 'train_on_tum_test_on_tum_south2',
        'pred': 'train_on_carla_test_on_tum_south2',
        'ema': 'train_on_carla_EMA_test_on_tum_south2',
        # Add more mappings for other result types as needed
    },
    'snow': {
        'gt': 'vis_gt_snow',
        'tum_bl': 'train_on_tum_test_on_snow',
        'pred': 'train_on_carla_test_on_snow',
        'ema': 'train_on_carla_EMA_test_on_snow',
        # Add more mappings for other result types as needed
    },
    's110_n':
        {
            
         "ema" : "train_on_carla_ema_on_s110_4_test_on_s110_n"   
        }
}

# Use the dataset and result type to find the correct output directory
out_box_dir_template = '/data/result_output/{}'
out_box_dir = out_box_dir_template.format(output_dir_mappings.get(dataset, {}).get(result_type, ""))

ground_project = False

# Validation list path adjustment based on result_type
val_list_suffix = 'ImageSets_yolo/val.txt' if result_type in ['ema',
                                                              'ema_proj'] and dataset is 'a9' else 'ImageSets/val.txt'
val_list = f'{base_dir}{val_list_suffix}'

val_list = base_dir + 'ImageSets/val.txt'

# Color list for visualization
color_list = {
    'car': (30, 144, 255),  # Bright Dodger Blue
    'bigcar': (255, 140, 0),  # Dark Orange
    'truck': (0, 128, 0),  # Dark Green
    'van': (148, 0, 211),  # Violet
    'bus': (255, 215, 0),  # Gold
    'cyclist': (135, 206, 235),  # Sky Blue
    'motorcycle': (107, 142, 35),  # Olive Drab
    'trailer': (221, 160, 221),  # Plum
    'pedestrian': (0, 255, 255)  # Cyan
}

thresh = -0.5
