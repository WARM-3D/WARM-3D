"""
Trainer helper module for MonoDETR model.
This module provides the training functionality including burn-in phase, EMA updates, and main training loop.
"""

import copy
import numpy as np
import os
import torch
import tqdm
import wandb
from collections import OrderedDict
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.models.monodetr.matcher import build_matcher_v2, filter_high_cost_pairs_and_sort
from utils import misc


def wandb_log(detr_losses_dict_log_on_source=None, detr_losses_dict_log_on_target=None, epoch=0):
    """
    Log training metrics to Weights & Biases.
    
    Args:
        detr_losses_dict_log_on_source: Dictionary of losses for source domain
        detr_losses_dict_log_on_target: Dictionary of losses for target domain
        epoch: Current training epoch
    """
    logged_indices_source = set()
    logged_indices_target = set()

    if detr_losses_dict_log_on_source is not None:
        for key, val in detr_losses_dict_log_on_source.items():
            if key == "loss_detr":
                continue
            if any(char.isdigit() for char in key):
                index = int(key[-1])
                if index not in logged_indices_source:
                    logged_indices_source.add(index)
            wandb.log(data={f"loss/{key}/source": val})

    if detr_losses_dict_log_on_target is not None:
        for key, val in detr_losses_dict_log_on_target.items():
            if key == "loss_detr":
                continue
            if any(char.isdigit() for char in key):
                index = int(key[-1])
                if index not in logged_indices_target:
                    logged_indices_target.add(index)
            wandb.log(data={f"loss/{key}/target": val})


def print_losses(batch_idx, detr_losses_dict_log, data_domain="source"):
    """
    Print training losses for monitoring.
    
    Args:
        batch_idx: Current batch index
        detr_losses_dict_log: Dictionary of losses to print
        data_domain: Source of data (source/target)
    """
    flags = [True] * 6

    if batch_idx % 30 == 0:
        print(f"---- {batch_idx}, {data_domain} ----")
        print(f"loss_detr: {detr_losses_dict_log['loss_detr']:.2f}")

        for key, val in detr_losses_dict_log.items():
            if key == "loss_detr":
                continue
            if any(num in key for num in "012345"):
                index = int(key[-1])
                if flags[index]:
                    print()
                    flags[index] = False
            print(f"{key}: {val:.2f}, ", end="")
        print("\n")


class Trainer(object):
    """
    Trainer class for MonoDETR model.
    Handles the training process including burn-in phase, EMA updates, and main training loop.
    """
    
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 matcher_cfg,
                 model_name):
        """
        Initialize the trainer.
        
        Args:
            cfg: Configuration dictionary
            model: List containing student and teacher models
            optimizer: Optimizer for training
            train_loader: List containing source and target dataloaders
            lr_scheduler: Learning rate scheduler
            warmup_lr_scheduler: Warmup learning rate scheduler
            logger: Logger instance
            loss: Loss function
            matcher_cfg: Matcher configuration
            model_name: Name of the model
        """
        self.cfg = cfg
        self.student_model = model[0]
        self.teacher_model = model[1]
        self.optimizer = optimizer
        self.source_loader = train_loader[0]
        self.target_loader = train_loader[1]
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.student_epoch = 0
        self.teacher_epoch = 0
        self.epoch = 0
        self.student_best_result = 0
        self.student_best_epoch = 0
        self.teacher_best_result = 0
        self.teacher_best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.matcher = build_matcher_v2(matcher_cfg)
        self.model_name = model_name
        self.tester = None
        self.output_dir = os.path.join('./' + self.cfg['save_path'], model_name)
        self.wandb = self.cfg.get('wandb_log', False)

        # Load pretrained/resume model
        if self.cfg.get('pretrain_model'):
            assert os.path.exists(self.cfg['pretrain_model'])
            load_checkpoint(model=self.student_model,
                            optimizer=None,
                            filename=self.cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if self.cfg.get('resume_model', None):
            resume_student_model_path = os.path.join(self.output_dir, "student_checkpoint.pth")
            assert os.path.exists(resume_student_model_path)
            self.epoch, self.student_best_result, self.student_best_epoch = load_checkpoint(
                model=self.student_model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_student_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info(
                "Loading Student Model Checkpoint... Best Result:{}, Best Epoch:{}".format(self.student_best_result,
                                                                                           self.student_best_epoch))

        # Load burned-in teacher model
        if self.cfg.get('burned_in_teacher'):
            assert os.path.exists(self.cfg['burned_in_teacher'])
            self.teacher_epoch, self.teacher_best_result, self.teacher_best_epoch = load_checkpoint(
                model=self.student_model,
                optimizer=None,
                filename=self.cfg['burned_in_teacher'],
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch
            self.logger.info(
                "Loading Teacher Model Checkpoint... Best Result:{}, Best Epoch:{}".format(self.teacher_best_result,
                                                                                           self.teacher_best_epoch))

        if self.wandb:
            wandb.login()
            wandb.init(
                project=self.cfg.get('wandb_project', 'monodetr'),
                config={})

    @staticmethod
    def reset_random_seed(epoch):
        """Reset random seed for reproducibility."""
        np.random.seed(np.random.get_state()[1][0] + epoch)

    @staticmethod
    def update_lr_scheduler(epoch, warmup_scheduler, lr_scheduler):
        """Update learning rate scheduler."""
        if warmup_scheduler is not None and epoch < 5:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()

    @staticmethod
    def save_model_checkpoint(epoch, model, optimizer, best_result, best_epoch, output_dir, model_type, cfg):
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_name = os.path.join(output_dir, f"{model_type}_checkpoint")
        if cfg['save_all']:
            checkpoint_name += f"_epoch_{epoch}"
        save_checkpoint(get_checkpoint_state(model, optimizer, epoch, best_result, best_epoch), checkpoint_name)

    def train(self) -> None:
        """Main training function."""
        # Burn-in phase
        if not self.cfg.get('burned_in_teacher'):
            self.burn_in_model()
        else:
            self.logger.info("Teacher Model is burned in")

        # EMA update after burn-in
        self.ema_update(keep_rate=0.0)
        self.logger.info("EMA Updated after burning-in")

        # Main training loop
        self.main_training_loop()

    def burn_in_model(self):
        """Handle the burning-in of the teacher model."""
        progress_bar = tqdm.tqdm(range(self.teacher_epoch, self.cfg['burn_in_max_epoch']), dynamic_ncols=True,
                                 desc='Burn-in epochs')
        for epoch in progress_bar:
            self.reset_random_seed(epoch)
            self.burn_in_one_epoch(epoch)
            self.teacher_epoch += 1

            self.update_lr_scheduler(epoch, self.warmup_lr_scheduler, self.lr_scheduler)
            if (self.teacher_epoch % self.cfg['save_frequency']) == 0:
                self.save_model_checkpoint(self.teacher_epoch, self.student_model, self.optimizer,
                                           self.teacher_best_result, self.teacher_best_epoch, self.output_dir,
                                           'teacher_burn_in', self.cfg)
                if self.tester is not None:
                    self.teacher_best_result, self.teacher_best_epoch = self.test_and_save_checkpoint(
                        self.student_model, 'teacher', self.teacher_best_result, self.teacher_best_epoch,
                        epoch=self.teacher_epoch, pretrained=True)

    def main_training_loop(self):
        """Execute the main training loop for both student and teacher models."""
        progress_bar = tqdm.tqdm(range(self.epoch, self.cfg['max_epoch']), dynamic_ncols=True, desc='Training epochs')
        for epoch in progress_bar:
            self.reset_random_seed(epoch)
            self.train_one_epoch(epoch)
            self.epoch += 1

            self.update_lr_scheduler(epoch, self.warmup_lr_scheduler, self.lr_scheduler)
            if (self.epoch % self.cfg['ema_update_frequency']) == 0:
                self.ema_update(keep_rate=self.cfg['keep_rate'])

            if (self.epoch % self.cfg['save_frequency']) == 0:
                self.save_model_checkpoint(self.epoch, self.student_model, self.optimizer, self.student_best_result,
                                           self.student_best_epoch, self.output_dir, 'student', self.cfg)
                self.save_model_checkpoint(self.epoch, self.teacher_model, self.optimizer, self.teacher_best_result,
                                           self.teacher_best_epoch, self.output_dir, 'teacher', self.cfg)

        # Log final best results
        self.logger.info(f"Student Best Result: {self.student_best_result}, epoch: {self.student_best_epoch}")

    def test_and_save_checkpoint(self, model, model_type, current_best_result, current_best_epoch, epoch=None,
                                 pretrained=False, update_condition=True, eval=True):
        """
        Tests the model and saves a checkpoint if it outperforms the current best result.

        Parameters:
        - model: The model to be tested.
        - model_type: The type of the model ('teacher' or other).
        - current_best_result: The current best result for comparison.
        - current_best_epoch: The epoch of the current best result.
        - epoch: The current epoch. Defaults to self.epoch if not provided.
        - pretrained: Flag indicating if the model is pretrained.
        - update_condition: Condition to check before updating the best result.

        Returns:
        - Tuple of current best result and epoch.
        """
        epoch = epoch or self.epoch
        self.logger.info(f"Test Epoch {epoch}")
        tester_mode = 'source' if pretrained else 'teacher_burn_in' if model_type == 'teacher' else 'target'

        self.tester.set_tester_mode(tester_mode)
        self.tester.epoch = epoch
        self.tester.model = model
        self.tester.inference()
        if not eval:
            return 0, 0
        current_result = self.tester.evaluate()

        if  update_condition:
            current_best_result = current_result
            current_best_epoch = epoch
            ckpt_type = f"{model_type}_pretrained_checkpoint_best" if pretrained else f"{model_type}_checkpoint_best"
            ckpt_name = os.path.join(self.output_dir, ckpt_type)
            save_checkpoint(get_checkpoint_state(model, self.optimizer, epoch, current_best_result, current_best_epoch),
                            ckpt_name)

        self.logger.info(f"{model_type.capitalize()} Best Result:{current_best_result}, epoch:{current_best_epoch}")
        return current_best_result, current_best_epoch

    @torch.no_grad()
    def ema_update(self, keep_rate=0.99) -> None:
        student_model_dict = self.student_model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.teacher_model.state_dict().items():
            if key in student_model_dict.keys() and value.dtype.is_floating_point:
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1.0 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        self.teacher_model.load_state_dict(new_teacher_dict)

    def burn_in_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.student_model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.source_loader),
                                 leave=(self.epoch + 1 == self.cfg['burn_in_max_epoch']),
                                 desc='iters')

        for batch_idx, (source_inputs, source_calibs, source_targets, source_info) in enumerate(self.source_loader):
            detr_losses_dict_log_on_source = self.train_on_source(source_inputs=source_inputs,
                                                                  source_calibs=source_calibs,
                                                                  source_targets=source_targets,
                                                                  source_info=source_info,
                                                                  batch_idx=batch_idx, model=self.student_model)
            progress_bar.update()

        if self.wandb:
            wandb_log(detr_losses_dict_log_on_source=detr_losses_dict_log_on_source,
                      epoch=epoch)

        progress_bar.close()

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.student_model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.target_loader), leave=(self.epoch + 1 == self.cfg['max_epoch']),
                                 desc='iters')

        for batch_idx, (target_inputs, target_calibs, target_targets, target_info) in enumerate(self.target_loader):

            pseudo_label_targets, mask, cost_first_image = self.generate_pseudo_labels(
                target_inputs=target_info['img_weak_augmented'],
                target_calibs=target_calibs,
                target_targets=target_targets,
                target_info=target_info,
                batch_idx=batch_idx,
                model=self.teacher_model)

            target_info['epoch'] = epoch
            detr_losses_dict_log_on_target = self.train_on_target_with_pseudo_labels(target_inputs=target_inputs,
                                                                                     target_calibs=target_calibs,
                                                                                     target_targets=pseudo_label_targets,
                                                                                     target_info=target_info,
                                                                                     mask=mask,
                                                                                     batch_idx=batch_idx,
                                                                                     model=self.student_model)
            if batch_idx % self.cfg['ema_step'] == 0:
                self.ema_update(keep_rate=self.cfg['keep_rate'])
                self.logger.info("EMA Updated in Epoch {}, batch No.{}".format(self.epoch, batch_idx))

            progress_bar.update()

        if self.wandb:
            wandb_log(detr_losses_dict_log_on_source=None,
                      detr_losses_dict_log_on_target=detr_losses_dict_log_on_target,
                      epoch=epoch)

        progress_bar.close()

    @staticmethod
    def prepare_targets(targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d', 'ground']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                elif key == 'feature_extrinsic':
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list

    def train_on_source(self, source_inputs, source_calibs, source_targets, source_info, batch_idx, model):
        source_inputs = source_inputs.to(self.device)
        source_calibs = source_calibs.to(self.device)
        for key in source_targets.keys():
            source_targets[key] = source_targets[key].to(self.device)

        img_sizes = source_targets['img_size']
        source_targets = self.prepare_targets(source_targets, source_inputs.shape[0])

        self.optimizer.zero_grad()
        outputs = model(source_inputs, source_calibs, source_targets, img_sizes)

        detr_losses, detr_losses_dict_log = self.calculate_losses(outputs=outputs, targets=source_targets,
                                                                  batch_idx=batch_idx, info=source_info,
                                                                  data_domain="source")
        detr_losses.backward()
        self.optimizer.step()

        return detr_losses_dict_log

    def train_on_target(self, target_inputs, target_calibs, target_targets, target_info, batch_idx, model):
        target_inputs = target_inputs.to(self.device)
        target_calibs = target_calibs.to(self.device)
        for key in target_targets.keys():
            target_targets[key] = target_targets[key].to(self.device)

        img_sizes = target_targets['img_size']
        target_targets = self.prepare_targets(target_targets, target_inputs.shape[0])

        self.optimizer.zero_grad()
        outputs = model(target_inputs, target_calibs, target_targets, img_sizes)

        detr_losses, detr_losses_dict_log = self.calculate_losses(outputs=outputs, targets=target_targets,
                                                                  batch_idx=batch_idx, info=target_info,
                                                                  data_domain="target")
        detr_losses.backward()
        self.optimizer.step()

        return detr_losses_dict_log

    def train_on_target_with_pseudo_labels(self, target_inputs, target_calibs, target_targets, target_info, mask,
                                           batch_idx,
                                           model):
        target_inputs = target_inputs.to(self.device)
        target_calibs = target_calibs.to(self.device)
        for key in target_targets.keys():
            target_targets[key] = target_targets[key].to(self.device)

        img_sizes = torch.tensor(target_info['img_size']).to(self.device)
        target_targets = self.prepare_targets(target_targets, target_inputs.shape[0])

        self.optimizer.zero_grad()

        outputs = model(target_inputs, target_calibs, target_targets, img_sizes)

        detr_losses, detr_losses_dict_log = self.calculate_losses(outputs=outputs, targets=target_targets,
                                                                  batch_idx=batch_idx, info=target_info, mask=mask,
                                                                  data_domain="target")
        detr_losses.backward()
        self.optimizer.step()

        return detr_losses_dict_log

    def generate_pseudo_labels(self, target_inputs, target_calibs, target_targets, target_info, batch_idx, model):
        target_inputs = target_inputs.to(self.device)
        target_calibs = target_calibs.to(self.device)
        target_targets_cpu = copy.deepcopy(target_targets)
        for key in target_targets.keys():
            target_targets[key] = target_targets[key].to(self.device)

        img_sizes = target_targets['img_size']
        target_targets = self.prepare_targets(target_targets, target_inputs.shape[0])

        model.eval()
        with torch.no_grad():
            outputs = model(target_inputs, target_calibs, target_targets, img_sizes)

        pseudo_targets, mask, cost_first_image = self.extract_dets_from_outputs(outputs, target_targets,
                                                                                target_targets_cpu)

        return pseudo_targets, mask, cost_first_image

    @staticmethod
    def sort_src_data(indices, src):
        """
        Sorts the src data based on the given indices corresponding to target indices.
        
        Parameters:
        - indices: A list of tuples, where each tuple contains two tensors. The first tensor is the src indices,
                and the second tensor is the target indices.
        - src: A tensor of shape (batch_size, item_size, value_dim) representing the source data.
        
        Returns:
        - A tensor of shape (batch_size, topk, value_dim) with the src data sorted according to the target indices
        and remaining positions filled with zeros.
        """
        sorted_src = torch.zeros_like(src)

        for i, (src_idxs, target_idxs) in enumerate(indices):
            sorted_data = src[i, src_idxs]
            sorted_src[i, :sorted_data.size(0)] = sorted_data

        return sorted_src

    def extract_dets_from_outputs(self, outputs, targets, targets_cpu):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        with torch.no_grad():
            indices, cost_map = self.matcher(outputs_without_aux, targets)

        threshold = self.cfg['threshold_increase_list']

        sorted_indices, filtered_sorted_indices, mask = filter_high_cost_pairs_and_sort(indices, cost_map, threshold)

        boxes = self.sort_src_data(sorted_indices, outputs['pred_boxes'])
        heading = self.sort_src_data(sorted_indices, outputs['pred_angle'])
        size_3d = self.sort_src_data(sorted_indices, outputs['pred_3d_dim'])
        outputs_depths = self.sort_src_data(sorted_indices, outputs['pred_depth'])
        depth = outputs_depths[:, :, 0: 1]

        cost_first_image = None

        targets_cpu['boxes_3d'] = boxes
        targets_cpu['depth'] = depth
        targets_cpu['size_3d'] = size_3d
        targets_cpu['heading_bin'], targets_cpu['heading_res'] = (self.get_heading_angle(heading.cpu()))
        targets_cpu['feature_extrinsic'] = outputs['pred_feature_extrinsic'].detach().cpu()

        return targets_cpu, mask, cost_first_image

    @staticmethod
    def get_heading_angle(headings):
        # Optimized version that works directly with PyTorch tensors
        yaw_cls = headings[:, :, 0:12].argmax(dim=-1)
        yaw_res = torch.gather(headings[:, :, 12:24], -1, yaw_cls.unsqueeze(-1)).squeeze(-1)

        return yaw_cls, yaw_res

    def calculate_losses(self, outputs, targets, batch_idx, info, mask=None, data_domain=None):
        # code implementation
        detr_losses_dict = self.detr_loss(outputs, targets, mask, info, data_domain=data_domain)

        weight_dict = {}
        if data_domain == "source":
            weight_dict = self.detr_loss.weight_dict_source
        elif data_domain == "target":
            weight_dict = self.detr_loss.weight_dict_target

        detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if
                                     k in weight_dict]
        detr_losses = sum(detr_losses_dict_weighted)

        detr_losses_dict = misc.reduce_dict(detr_losses_dict)
        detr_losses_dict_log = {}
        detr_losses_log = 0
        for k in detr_losses_dict.keys():
            if k in weight_dict:
                detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                detr_losses_log += detr_losses_dict_log[k]
        detr_losses_dict_log["loss_detr"] = detr_losses_log

        print_losses(batch_idx, detr_losses_dict_log, data_domain=data_domain)

        return detr_losses, detr_losses_dict_log


def vis_match_2d(target_inputs, target_targets, cost_first_image):
    from utils import box_ops
    from PIL import Image
    import matplotlib.pyplot as plt
    # vis them

    cost_first_image = cost_first_image.diag().detach().cpu()

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    def rescale_bboxes(boxes, size):
        img_w, img_h = size
        b = box_ops.box_cxcylrtb_to_xyxy(boxes)
        b = b * torch.tensor([img_w / 2, img_h / 2, img_w / 2, img_h / 2], dtype=torch.float32)
        return b

    # convert boxes from [0; 1] to image scales
    boxes = rescale_bboxes(target_targets["boxes_3d"].detach().cpu(),
                           target_targets["img_size"][0].detach().cpu().tolist())

    image_tensor = target_inputs[0].cpu()
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    image_tensor = (image_tensor * 255).byte()
    image_tensor = image_tensor.permute(1, 2, 0)  # 从[3, 384, 1280]转置为[384, 1280, 3]
    image_np = image_tensor.numpy()
    pil_img = Image.fromarray(image_np)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for (xmin, ymin, xmax, ymax), c, cost in zip(boxes[0].tolist(), colors, cost_first_image):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))

        text = f'{cost:0.2f}'

        ax.text(xmin, ymin, text, fontsize=5,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
