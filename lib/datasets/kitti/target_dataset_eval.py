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

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TargetDatasetEval(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('target_eval_root_dir')
        self.dataset = cfg.get('target_dataset_name', 'tum')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Car', 'Pedestrian', 'Cyclist', 'Motorcycle','Van', 'Bus', 'Truck', 'Trailer']
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
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(
                ['Van', 'Truck', 'Bus', 'Motorcycle', 'Trailer', 'Pedestrian', 'Cyclist'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(
            self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(
            self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.gt_label_dir = os.path.join(self.data_dir, 'label_2')
        self.extrinsics_dir = os.path.join(self.data_dir, 'extrinsics')
        # data augmentation configuration
        self.data_augmentation = True if split in [
            'train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

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

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_extrinsics(self, idx):
        extrinsics_file = os.path.join(self.extrinsics_dir, '%06d.json' % idx)
        assert os.path.exists(extrinsics_file)
        return Extrinsics(extrinsics_file, self.dataset)

    def eval(self, results_dir, gt_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir, img_ids)
        gt_annos = kitti.get_label_annos(gt_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'BigCar': 3}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(
                gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)

        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample  # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False

        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(
                        np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * \
                                 np.clip(np.random.randn() * self.shift, -
                                 2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * \
                                 np.clip(np.random.randn() * self.shift, -
                                 2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = img[:, :, :3]
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': index,
                'img_size': img_size,
                'resolution': self.resolution,
                'data_domain': 1}  # 1: target domain, 0: source domain

        if self.split == 'test':
            calib = self.get_calib(index)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        extrinsics = self.get_extrinsics(index)

        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - \
                                                   x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:
                    object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi:
                    object.alpha += 2 * np.pi
                if object.ry > np.pi:
                    object.ry -= 2 * np.pi
                if object.ry < -np.pi:
                    object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        # heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_bin = np.zeros((self.max_objs, 3), dtype=np.int64)
        # heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_res = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        ground = np.zeros((self.max_objs, 4), dtype=np.float32)

        object_num = len(objects) if len(
            objects) < self.max_objs else self.max_objs

        for i in range(object_num):

            # filter objects by writelist
            if objects[i].cls_type not in self.cls2id.keys():
                # print('filter out ', objects[i].cls_type)
                continue

            # filter inappropriate samples
            # if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
            #     print('filter out ', objects[i].level_str, objects[i].cls_type, objects[i].trucation,
            #           objects[i].occlusion)
            #     continue

            # _, object_in_camera = get_3d_box(self.dataset.lower(), objects[i], calib.P2)
            obj_angles = np.array([objects[i].ry, objects[i].rp, objects[i].rr])

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            # real 3D center in 3D space

            center_3d = objects[i].pos
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            # project 3D center to image plane
            center_3d, _ = calib.rect_to_img(center_3d)
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - \
                   corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - \
                   corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale

            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale

            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]

            for j in range(len(obj_angles)):
                if obj_angles[j] > np.pi:
                    obj_angles[j] -= 2 * np.pi
                elif obj_angles[j] < -np.pi:
                    obj_angles[j] += 2 * np.pi
            heading_bin[i, :], heading_res[i, :] = angles2class(obj_angles)

            # encoding size_3d
            src_size_3d[i] = np.array(
                [objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            # if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
            #     mask_2d[i] = 1
            mask_2d[i] = 1

            calibs[i] = calib.P2 / 2
            ground[i] = extrinsics.extrinsics
            ground[i, 3] = ground[i, 3] / (-10)

        # collect return data
        inputs = img
        targets = {
            'calibs': calibs,
            'indices': indices,
            'img_size': img_size,
            'labels': labels,
            'boxes': boxes,
            'boxes_3d': boxes_3d,
            'depth': depth,
            'size_2d': size_2d,
            'size_3d': size_3d,
            'src_size_3d': src_size_3d,
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'mask_2d': mask_2d,
            'ground': ground}

        info = {'img_id': index,
                'img_size': img_size,
                'resolution': self.resolution,
                'data_domain': 1}  # 1: target domain, 0: source domain
        return inputs, calib.P2, targets, info


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    cfg = yaml.load(open("configs/monodetr.yaml", 'r'), Loader=yaml.Loader)

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    workers = 4


    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    cfg = cfg['dataset']
    test_target_set = TargetDatasetEval(split=cfg['test_split'], cfg=cfg)
    dataloader = DataLoader(dataset=test_target_set,
                            batch_size=cfg['batch_size'],
                            num_workers=workers,
                            worker_init_fn=my_worker_init_fn,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    results_dir = '/workspace/outputs/monodetr_ema/outputs/data_target_proj'
    gt_dir = '/workspace/outputs/monodetr_ema/outputs/data_target'
    # gt_dir = '/data/tum_traffic_intersection_dataset_aligned/training/label_2'

    dataloader.dataset.eval(results_dir=results_dir, gt_dir=gt_dir, logger=logger)
