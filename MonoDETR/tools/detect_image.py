import copy
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import yaml
import argparse
import datetime
import numpy as np
from lib.helpers.model_helper import build_model
import time

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detect_detections
from utils.visual_tools.draw_bounding_box import draw_detection
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.datasets.kitti.detect_image_loader import LoadDetectSource
from utils.dataloaders import IMG_FORMATS, VID_FORMATS , LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from tools.tracker import Tracker
# from PIL import Image
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

def main():

    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    # tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)

    # if args.threshold is not None:
    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        print(cfg['model_name'])
    # cfg['model_name'] = cfg['model_name'] + str(args.threshold)
    cfg['model']['model_name'] = cfg['model_name']
    if args.evaluate_only:
        camera_id = args.camera_id
        cfg['dataset']['target_eval_root_dir'] = "/data/tum_traffic_intersection_dataset_" + camera_id
    # elif args.threshold is None:
    #     cfg['trainer']['threshold_increase_list'] = 1.0


    source = cfg['detect']['source']
    project = cfg['detect']['project']

    save_img = not cfg['detect']['nosave'] and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / cfg['detect']['name'], exist_ok=cfg['detect']['exist_ok'])  # increment run
    (save_dir / 'labels' if cfg['detect']['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # model_name = cfg['model_name']

    log_file = os.path.join(save_dir, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    # train_loader, test_loader = build_dataloader(cfg['dataset'])
    logger.info('Created train and test dataloader.')

    # build model
    model, _ , _ = build_model(cfg['model'])

    if cfg['detect']['device']:
        device = cfg['detect']['device']
        # gpu_ids = list(map(int, cfg['detect']['gpu_ids'].split(',')))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_ids = list(map(int, cfg['detect']['gpu_ids'].split(',')))

    load_checkpoint(model=model,
                    optimizer=None,
                    filename=cfg['detect']['checkpoint'],
                    map_location=device,
                    logger=logger)
    
    # if len(gpu_ids) == 1:
    model = model.to(device)
    # else:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
        
    torch.set_grad_enabled(False)
    model.eval()
    # imgsz = check_img_size(imgsz)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # if webcam:
    #     view_img = check_imshow(warn=True)
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    #     bs = len(dataset)
    # elif screenshot:
    #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # else:
    dataset = LoadDetectSource(source, cfg)
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    calibs = torch.tensor(cfg['detect']['calib']).unsqueeze(0).to(device)

    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s, info in dataset:
        # im0s = np.array(im0s.resize((1920, 1200), Image.Resampling.LANCZOS))

        im0s = np.array(im0s)
        
        inputs = torch.from_numpy(im).to(device)
        calibs = calibs.to(device)

        img_sizes = torch.from_numpy(info['img_size']).to(device)

        # start_time = time.time()
        ###dn
        targets = None
        outputs = model(inputs, calibs, targets, img_sizes)
        # end_time = time.time()
        # model_infer_time += end_time - start_time

        dets = extract_dets_from_outputs(outputs=outputs, K=cfg['detect']['max_objs'], topk=cfg['detect']['topk'])

        dets = dets.detach().cpu().numpy()

        # get corresponding calibs & transform tensor to numpy
        # calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
        # info = {key: val for key, val in info.items()}
        cls_mean_size = info['cls_mean_size']
        extrinsic = cfg['detect']['extrinsic']
        dets = decode_detect_detections(
            dets=dets,
            info=info,
            calibs=calibs,
            cls_mean_size=cls_mean_size,
            extrinsic=extrinsic,
            threshold=cfg['detect']['threshold'],
            NMS=cfg['detect']['NMS'],
            )

        draw_img = draw_detection(im0s, dets,calibs,cfg)
        

        cv2.imshow('test', draw_img)
        cv2.waitKey(5)  # 1 millisecond
        # time.sleep(0.5)
        save_path = cfg['detect']['save_path']
        # Save results (image with detections)
        if cfg['detect']['save_img']:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, draw_img)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', default="configs/monodetr_detect.yaml",
                        help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    # parser.add_argument('-threshold', type=float, help='Set the threshold for detection')
    parser.add_argument('-camera_id', type=str, help='Set the camera id for detection')
    parser.add_argument('-model_name', type=str, help='Set the model name for detection')
    args = parser.parse_args()
    main()
