"""
Main training script for MonoDETR model.
This script handles the training and evaluation of the MonoDETR model for monocular 3D object detection.
"""

import copy
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


def main():
    """Main training and evaluation function."""
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # Configure model settings based on arguments
    if args.threshold is not None:
        cfg['trainer']['threshold_increase_list'] = args.threshold
        if args.model_name is not None:
            cfg['model_name'] = args.model_name
            print(cfg['model_name'])
        cfg['model']['model_name'] = cfg['model_name']
        if args.evaluate_only:
            camera_id = args.camera_id
            cfg['dataset']['target_eval_root_dir'] = "/data/tum_traffic_intersection_dataset_" + camera_id
    else:
        pass

    set_random_seed(cfg.get('random_seed', 444))

    # Setup output directory and logging
    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # Initialize data loaders
    train_loader, test_loader = build_dataloader(cfg['dataset'])
    logger.info('Created train and test dataloader.')

    # Initialize model and loss
    model, loss, matcher = build_model(cfg['model'])
    loss.dataloader = train_loader[1]
    logger.info('Created prime model.')

    # Initialize EMA model
    ema_model = copy.deepcopy(model).eval()
    logger.info('Created ema model.')
    for param in ema_model.parameters():
        param.requires_grad = False

    # Setup device and model parallelization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
        ema_model = ema_model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
        ema_model = torch.nn.DataParallel(ema_model, device_ids=gpu_ids).to(device)

    model_list = [model, ema_model]

    # Evaluation mode
    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.set_output_folder(camera_id)
        tester.set_tester_mode('target_eval')
        tester.test()
        return

    # Initialize optimizer and learning rate scheduler
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    # Initialize trainer and tester
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model_list,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      matcher_cfg=cfg['model'],
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.student_model,
                    dataloader=test_loader,
                    logger=logger,
                    loss=loss,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    # Start training
    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    # Run testing if not on test split
    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.set_tester_mode('target')
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', default="configs/monodetr.yaml",
                        help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('-threshold', type=float, help='Set the threshold for detection')
    parser.add_argument('-camera_id', type=str, help='Set the camera id for detection')
    parser.add_argument('-model_name', type=str, help='Set the model name for detection')
    args = parser.parse_args()
    main()
