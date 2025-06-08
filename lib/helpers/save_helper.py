import os

import torch


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result,
            'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            checkpoint_state = checkpoint['model_state']

            # Prepare the model for loading parameters
            model_dict = model.state_dict()

            filtered_checkpoint = {k: v for k, v in checkpoint_state.items() if
                                   k in model_dict and model_dict[k].size() == v.size()}

            model_dict.update(filtered_checkpoint)

            # print weight that doesn't match
            missing_keys = set(model_dict.keys()) - set(filtered_checkpoint.keys())
            if missing_keys:
                print("Warning: Missing keys in checkpoint:", missing_keys)

            # Fill missing keys in model_state with model_dict using ** operator for dictionary unpacking
            # model_state_updated = {**model_dict, **model_state}

            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                if logger:
                    logger.error(f"Error loading model state dict: {e}")
                raise e

        if optimizer is not None and 'optimizer_state' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except ValueError as e:
                if logger:
                    logger.error(f"Error loading optimizer state dict: {e}")
                raise e
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
