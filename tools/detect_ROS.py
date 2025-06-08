import copy
import warnings
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import yaml
import argparse
import datetime
from cv_bridge import CvBridge, CvBridgeError
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detect_detections
from utils.visual_tools.draw_bounding_box import draw_detection
from lib.helpers.utils_helper import create_logger, set_random_seed
import torch
from pathlib import Path

import os
from utils.general import increment_path

warnings.filterwarnings("ignore")

def callback(data, args):
    model, device, cfg = args
    bridge = CvBridge()
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    im0s = np.array(cv_image)
    inputs = torch.from_numpy(im0s).to(device)
    calibs = torch.tensor(cfg['detect']['calib']).unsqueeze(0).to(device)
    img_sizes = torch.from_numpy(np.array([im0s.shape[0], im0s.shape[1]])).to(device)

    outputs = model(inputs, calibs, None, img_sizes)
    dets = extract_dets_from_outputs(outputs=outputs, K=cfg['detect']['max_objs'], topk=cfg['detect']['topk'])
    dets = dets.detach().cpu().numpy()
    
    cls_mean_size = cfg['detect']['cls_mean_size']
    extrinsic = cfg['detect']['extrinsic']
    dets = decode_detect_detections(
        dets=dets,
        info={"img_size": img_sizes, "cls_mean_size": cls_mean_size},
        calibs=calibs,
        cls_mean_size=cls_mean_size,
        extrinsic=extrinsic,
        threshold=cfg['detect']['threshold'],
        NMS=cfg['detect']['NMS'],
    )

    draw_img = draw_detection(im0s, dets, calibs, cfg)
    cv2.imshow('test', draw_img)
    cv2.waitKey(5)

def main():
    rospy.init_node('image_listener', anonymous=True)
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', default="configs/monodetr_detect.yaml",
                        help='settings of detection in yaml format')
    args = parser.parse_args()
    
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model, _ , _ = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    # print(cfg)
    project = cfg['detect']['project']

    save_dir = increment_path(Path(project) / cfg['detect']['name'], exist_ok=cfg['detect']['exist_ok'])  # increment run
    (save_dir / 'labels' if cfg['detect']['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    log_file = os.path.join(save_dir, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    load_checkpoint(model=model, optimizer=None, filename=cfg['detect']['checkpoint'], map_location=device,logger=logger)
    
    rospy.Subscriber("/s110/w/cam/8/image_raw/compressed", CompressedImage, callback, (model, device, cfg))
    rospy.spin()

if __name__ == '__main__':
    main()
