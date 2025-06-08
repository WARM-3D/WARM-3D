import os
import shutil
import time
import torch
import tqdm
import wandb
from lib.helpers.decode_helper import decode_detections
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.save_helper import load_checkpoint
from utils import misc


def wandb_log(detr_losses_dict_log_on_val, epoch):
    logged_indices_val = set()

    for key, val in detr_losses_dict_log_on_val.items():
        if key == "loss_detr":
            continue

        if any(char.isdigit() for char in key):
            index = int(key[-1])
            if index not in logged_indices_val:
                logged_indices_val.add(index)
        wandb.log({'epoch': epoch, f"loss/{key}/val": val})


def print_losses(batch_idx, detr_losses_dict_log, loss_type="source"):
    flags = [True] * 6

    if batch_idx % 30 == 0:
        print(f"---- {batch_idx}, {loss_type} ----")
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


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, loss=None, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader_list = dataloader
        self.dataloader = self.dataloader_list[0]
        self.max_objs = self.dataloader.dataset.max_objs  # max objects per images, defined in dataset
        self.class_name = self.dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.result_dir = None
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.detr_loss = loss
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.tester_mode = 'source'
        self.epoch = 0
        self.wandb = self.cfg.get('wandb_log', False)

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir,
                                               "student_checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "student_checkpoint_best.pth")
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
                # checkpoint_path = '/workspace/outputs/monodetr_bl_carla_large_ext_vis_depth_MLP/checkpoint_best.pth'
            assert os.path.exists(checkpoint_path), "Checkpoint: {} not found".format(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            # self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes)
            # detr_losses, detr_losses_dict_log = self.calculate_losses(outputs=outputs, targets=targets,
            #                                                           batch_idx=batch_idx, loss_type='target')
            # if self.wandb:
            #     wandb_log(detr_losses_dict_log, epoch=self.epoch)

            if self.cfg['test_split'] != 'test':
                # self.calculate_losses(outputs, targets, batch_idx)
                pass
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        if self.tester_mode == 'source':
            output_dir = os.path.join(self.output_dir, 'outputs', 'data_source')
        elif self.tester_mode == 'target':
            output_dir = os.path.join(self.output_dir, 'outputs', 'data_target')
        elif self.tester_mode == 'teacher_burn_in':
            output_dir = os.path.join(self.output_dir, 'outputs', 'data_target_burn_in')
        elif self.tester_mode == 'target_eval':
            output_dir = os.path.join(self.output_dir, 'outputs', 'data_eval')
            
        if self.result_dir is not None:
            output_dir = self.result_dir

        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)),
                            exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

        if self.train_cfg['save_all']:
            save_sir = output_dir + f"_epoch_{self.epoch}"
            try:
                # This copies the entire directory tree including all files and sub-directories from the source to the destination.
                # If the destination directory already exists, it will raise a FileExistsError.
                shutil.copytree(output_dir, save_sir)
                print("Folder copied successfully")
            except Exception as e:
                # If an error occurs during copying, it will print the error message.
                print(f"An error occurred while copying: {e}")

    def evaluate(self):
        if self.tester_mode == 'source':
            results_dir = os.path.join(self.output_dir, 'outputs', 'data_source')
        elif self.tester_mode == 'target':
            results_dir = os.path.join(self.output_dir, 'outputs', 'data_target')
        elif self.tester_mode == 'teacher_burn_in':
            results_dir = os.path.join(self.output_dir, 'outputs', 'data_target_burn_in')
        elif self.tester_mode == 'target_eval':
            return
        if self.result_dir is not None:
            results_dir = self.result_dir
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result

    def calculate_losses(self, outputs, targets, batch_idx, loss_type, data_domain='target'):
        # code implementation
        detr_losses_dict = self.detr_loss(outputs, targets, data_domain=data_domain)

        weight_dict = self.detr_loss.weight_dict
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

        print_losses(batch_idx, detr_losses_dict_log, loss_type=loss_type)

        return detr_losses, detr_losses_dict_log

    def set_tester_mode(self, mode):
        if mode is 'source':
            self.dataloader = self.dataloader_list[0]
            self.tester_mode = 'source'
        elif mode is 'target':
            self.dataloader = self.dataloader_list[1]
            self.tester_mode = 'target'
        elif mode is 'teacher_burn_in':
            self.dataloader = self.dataloader_list[0]
            self.tester_mode = 'teacher_burn_in'
        elif mode is 'target_eval':
            self.dataloader = self.dataloader_list[2]
            self.tester_mode = 'target_eval'
        else:
            self.logger.error("Invalid mode for tester")
            
    def set_output_folder(self, camera_id):
        self.result_dir = os.path.join(self.output_dir, 'outputs', camera_id)
