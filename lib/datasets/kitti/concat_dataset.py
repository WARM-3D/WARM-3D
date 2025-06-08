import torch
import numpy as np

class CustomConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_offsets = np.cumsum([0] + [len(d) for d in datasets[:-1]])
        if len(datasets) > 0 and hasattr(datasets[0], 'std'):
            self.std = datasets[0].std
            self.mean = datasets[0].mean
            self.cls_mean_size = datasets[0].cls_mean_size
        else:
            self.std = None
            self.mean = None
            self.cls_mean_size = None

    def __getitem__(self, idx):
        for i, offset in enumerate(self.dataset_offsets):
            if idx < offset + len(self.datasets[i]):
                return self.datasets[i][idx - offset]
        raise IndexError("Index out of range")

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def get_calib(self, idx):
        for i, offset in enumerate(self.dataset_offsets):
            if idx < offset + len(self.datasets[i]):
                return self.datasets[i].get_calib(idx - offset)
        raise IndexError("Index out of range")
    
    def get_extrinsics(self, idx):
        for i, offset in enumerate(self.dataset_offsets):
            if idx < offset + len(self.datasets[i]):
                return self.datasets[i].get_extrinsics(idx - offset)
        raise IndexError("Index out of range")
    
    def eval(self, results_dir, logger):
        overall_result = []
        for dataset in self.datasets:
            result = dataset.eval(results_dir, logger)
            overall_result.append(result)
        # You might need to implement a way to combine or summarize the results from all datasets
        return overall_result