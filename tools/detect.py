"""
MonoDETR Detection Script
Performs inference and visualization for monocular 3D object detection.
"""

import os
import sys
import torch
import yaml
import argparse
import datetime
import numpy as np
from pathlib import Path

from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detect_detections
from utils.visual_tools.draw_bounding_box import draw_detection
from lib.helpers.utils_helper import create_logger, set_random_seed
from lib.datasets.kitti.detect_image_loader import LoadDetectSource
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (Profile, check_file, cv2, increment_path)


def save_kitti_format(dets, output_path, frame_id):
    """
    Save detections in KITTI format to a text file.
    """
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, frame_id.replace('.jpg', '.txt'))
    with open(filename, 'w') as f:
        for det in dets:
            object_type = int(det[0])
            truncated = det[1]
            alpha = det[13]
            bbox = det[2:6]
            dimensions = det[6:9]
            location = det[9:12]
            rotation_y = det[14]
            f.write(f"{object_type} {truncated:.2f} 0 {alpha:.2f} "
                    f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
                    f"{dimensions[0]:.2f} {dimensions[1]:.2f} {dimensions[2]:.2f} "
                    f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} "
                    f"{rotation_y:.2f}\n")


def main():
    """
    Main detection function for MonoDETR.
    """
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        print(cfg['model_name'])
    cfg['model']['model_name'] = cfg['model_name']
    if args.evaluate_only:
        camera_id = args.camera_id
        cfg['dataset']['target_eval_root_dir'] = f"/data/tum_traffic_intersection_dataset_{camera_id}"

    source = cfg['detect']['source']
    project = cfg['detect']['project']

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / cfg['detect']['name'], exist_ok=cfg['detect']['exist_ok'])
    (save_dir / 'labels' if cfg['detect']['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(save_dir, f'train.log.{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    logger = create_logger(log_file)

    model, _, _ = build_model(cfg['model'])
    device = cfg['detect'].get('device') or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_checkpoint(model=model, optimizer=None, filename=cfg['detect']['checkpoint'], map_location=device, logger=logger)
    model = model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    dataset = LoadDetectSource(source, cfg)
    calibs = torch.tensor(cfg['detect']['calib']).unsqueeze(0).to(device)

    for path, im, im0s, _, _, info in dataset:
        im0s = np.array(im0s)
        inputs = torch.from_numpy(im).to(device)
        img_sizes = torch.from_numpy(info['img_size']).to(device)
        outputs = model(inputs, calibs, None, img_sizes)
        dets = extract_dets_from_outputs(outputs=outputs, K=cfg['detect']['max_objs'], topk=cfg['detect']['topk'])
        dets = dets.detach().cpu().numpy()
        dets = decode_detect_detections(
            dets=dets,
            info=info,
            calibs=calibs,
            cls_mean_size=info['cls_mean_size'],
            extrinsic=cfg['detect']['extrinsic'],
            threshold=cfg['detect']['threshold'],
            NMS=cfg['detect']['NMS'],
        )
        draw_img = draw_detection(im0s, dets, calibs, cfg)
        cv2.imshow('test', draw_img)
        cv2.waitKey(5)
        if cfg['detect'].get('save_text', True):
            output_text_path = cfg['detect']['text_save_path']
            frame_id = path.split('/')[-1]
            save_kitti_format(dets, output_text_path, frame_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', default="configs/monodetr_detect.yaml",
                        help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('-camera_id', type=str, help='Set the camera id for detection')
    parser.add_argument('-model_name', type=str, help='Set the model name for detection')
    args = parser.parse_args()
    main()
