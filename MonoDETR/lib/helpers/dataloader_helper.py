import itertools
import numpy as np
import torch.utils
import torch.utils.data
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.concat_dataset import CustomConcatDataset
from lib.datasets.kitti.target_dataset_eval import TargetDatasetEval
from lib.datasets.kitti.target_dataset_yolo import TargetDataset
from torch.utils.data import DataLoader
import torch

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        source_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_source_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)

        
        cfg['target_root_dir'], cfg['target_eval_root_dir'] = '/data/s110_o_dataset/', '/data/s110_o_dataset/'
        target_set_1 = TargetDataset(split=cfg['train_split'], cfg=cfg)
        test_target_set_1 = TargetDataset(split=cfg['test_split'], cfg=cfg)
        eval_target_set_1 = TargetDatasetEval(split=cfg['test_split'], cfg=cfg)

        
        cfg['target_root_dir'], cfg['target_eval_root_dir'] = '/data/s110_n_dataset/', '/data/s110_n_dataset/'
        target_set_2 = TargetDataset(split=cfg['train_split'], cfg=cfg)
        test_target_set_2 = TargetDataset(split=cfg['test_split'], cfg=cfg)
        eval_target_set_2 = TargetDatasetEval(split=cfg['test_split'], cfg=cfg)


        cfg['target_root_dir'], cfg['target_eval_root_dir'] = '/data/s110_s_dataset/', '/data/s110_s_dataset/'
        target_set_3 = TargetDataset(split=cfg['train_split'], cfg=cfg)
        test_target_set_3 = TargetDataset(split=cfg['test_split'], cfg=cfg)
        eval_target_set_3 = TargetDatasetEval(split=cfg['test_split'], cfg=cfg)

        cfg['target_root_dir'], cfg['target_eval_root_dir'] = '/data/s110_w_dataset/', '/data/s110_w_dataset/'
        target_set_4 = TargetDataset(split=cfg['train_split'], cfg=cfg)
        test_target_set_4 = TargetDataset(split=cfg['test_split'], cfg=cfg)
        eval_target_set_4 = TargetDatasetEval(split=cfg['test_split'], cfg=cfg)
        
        test_target_set = CustomConcatDataset([test_target_set_1,test_target_set_2,test_target_set_3,test_target_set_4])
        target_set = CustomConcatDataset([target_set_1,target_set_2,target_set_3,target_set_4])
        eval_target_set = CustomConcatDataset([eval_target_set_1,eval_target_set_2,eval_target_set_3,eval_target_set_4])
       
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    test_source_loader = DataLoader(dataset=test_source_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    test_target_loader = DataLoader(dataset=test_target_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    eval_target_loader = DataLoader(dataset=eval_target_set,
                                    batch_size=cfg['batch_size'],
                                    num_workers=workers,
                                    worker_init_fn=my_worker_init_fn,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
    source_loader = DataLoader(dataset=source_set,
                               batch_size=cfg['batch_size'],
                               num_workers=workers,
                               worker_init_fn=my_worker_init_fn,
                               shuffle=True,
                               pin_memory=False,
                               drop_last=True)
    target_loader = DataLoader(dataset=target_set,
                               batch_size=cfg['batch_size'],
                               num_workers=workers,
                               worker_init_fn=my_worker_init_fn,
                               shuffle=True,
                               pin_memory=False,
                               drop_last=True)

    train_loader = [source_loader, target_loader]
    test_loader = [test_source_loader, test_target_loader, eval_target_loader]

    return train_loader, test_loader
