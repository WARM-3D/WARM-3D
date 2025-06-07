import datetime
import os
import sys

import yaml

sys.path.append(os.getcwd())
from lib.helpers.utils_helper import create_logger
from lib.datasets.kitti.pd import PhotometricDistort
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import Extrinsics
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.utils import angles2class, class2angles
from pathlib import Path
import glob

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def save_kitti_format(detections, output_path, frame_id):
    """
    Save detections in KITTI format.

    Args:
        detections (list): List of detections with fields for KITTI format (type, bbox, dimensions, etc.).
        output_path (str): Directory to save the text file.
        frame_id (int): ID of the frame or image.
    """
    filename = os.path.join(output_path, f"{frame_id:06}.txt")
    with open(filename, 'w') as f:
        for det in detections:
            # Example fields: adjust according to your actual detection fields.
            object_type = det['type']
            truncated = det.get('truncated', 0.0)
            occluded = det.get('occluded', 0)
            alpha = det.get('alpha', -1.0)
            bbox = det['bbox']  # [x1, y1, x2, y2]
            dimensions = det['dimensions']  # [h, w, l]
            location = det['location']  # [x, y, z]
            rotation_y = det.get('rotation_y', -10.0)

            f.write(f"{object_type} {truncated:.2f} {occluded} {alpha:.2f} "
                    f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
                    f"{dimensions[0]:.2f} {dimensions[1]:.2f} {dimensions[2]:.2f} "
                    f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} "
                    f"{rotation_y:.2f}\n")

class LoadDetectSource(data.Dataset):
    def __init__(self, path, cfg):
        
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv

        # self.img_size = img_size
        # self.stride = stride
        self.files = images + videos
        
        self.vid_stride = cfg['detect']['vid_stride']
        # basic configuration
        # self.root_dir = cfg.get('target_eval_root_dir')
        self.dataset = cfg['dataset']['target_dataset_name']
        # cfg.get('target_dataset_name', 'tum')
        # self.num_classes = 3
        # self.max_objs = 50
        self.class_name = ['Car', 'Pedestrian', 'Cyclist', 'BigCar']
        # self.cls2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Motorcycle': 2, 'Van': 3, 'Truck': 3, 'Bus': 3,
        #                'Trailer': 3}
        self.cls2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Motorcycle': 2, 'Van': 3, 'Bus': 4, 'Truck': 5,
                       'Trailer': 5}
        self.resolution = np.array([960, 600])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        # self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(
                ['Van', 'Truck', 'Bus', 'Motorcycle', 'Trailer', 'Pedestrian', 'Cyclist'])

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # statistics
        if self.dataset == 'carla':
            #############################   Carla   ###############################
            # Overall Mean: [0.45303926 0.43145254 0.39440957 1.        ]
            # Overall Standard Deviation: [0.21221164 0.20120451 0.19594808 0.        ]

            self.mean = np.array(
                [0.45303926, 0.43145254, 0.39440957], dtype=np.float32)
            self.std = np.array(
                [0.21221164, 0.20120451, 0.19594808], dtype=np.float32)
            self.cls_mean_size = np.array([[1.5889325974143211, 1.9149214083951036, 4.276534553682457],
                                           [1.740576131103809, 0.38891169859355834, 0.38891169859355834],
                                           [1.6485547876898914, 0.790322758934293, 1.7703334617919204],
                                           [2.4277323432934774, 2.4430386748732063, 6.068817606371639],
                                           [3.260256481155415, 2.9435933401272982, 12.648112271151325],
                                           [3.5331199789537897, 3.0104627639025203, 6.707483320057572]]
                                        #    [3.2232471869812227, 2.855413839502194, 6.863092670309799]]
                                          )
        elif self.dataset == 'tum':
            #############################   Tum Traffic   ###############################
            # Overall Mean: [0.38749358 0.41994163 0.38898486]
            # Overall Standard Deviation: [0.12390958 0.12040959 0.11534844]

            self.mean = np.array(
                [0.38749358, 0.41994163, 0.38898486], dtype=np.float32)
            self.std = np.array(
                [0.12390958, 0.12040959, 0.11534844], dtype=np.float32)
            self.cls_mean_size = np.array([[1.600738818746034, 1.8974925101649465, 4.251457307939188],
                                           [1.7714340239912671, 0.7477153762268287, 0.826559432933489],
                                           [1.6566415500538019, 0.8801722282023738, 1.7309257265877247],
                                           [3.2177711842218817, 2.843558774762212, 6.909936294615725]])


        elif self.dataset == 'kitti':
            #############################   Kitti   ###############################
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                           [1.52563191462, 1.62856739989, 3.88311640418],
                                           [1.73698127, 0.59706367, 1.76282397]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(
                self.cls_mean_size, dtype=np.float32)
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'


    def __len__(self):
        return self.nf  # number of files

    def __iter__(self):
        self.count = 0
        return self
    
    # def __getitem__(self, item):
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, img0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, img0 = self.cap.read()
            
            img0 = Image.fromarray(img0.astype('uint8'), 'RGB')
            self.frame += 1
            # img0 = self._cv2_rotate(img0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = Image.open(path).convert("RGB")
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '
            


        img_size = np.array(img0.size)
        # features_size = self.resolution // self.downsample  # W * H

        # data augmentation for image
        center = np.array(img_size) / 2

        # # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(
            center, img_size, 0, self.resolution, inv=1)
        img = img0.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)


        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = img[:, :, :3]
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {
                # 'img_id': index,
                'img_size': img_size,
                'resolution': self.resolution,
                "cls_mean_size": self.cls_mean_size,
                'data_domain': 1}  # 1: target domain, 0: source domain
        
        return path, img, img0, self.cap, s, info
    
    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    cfg = yaml.load(open("configs/monodetr.yaml", 'r'), Loader=yaml.Loader)

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)
