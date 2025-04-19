import os
import argparse
import functools
import os, shutil
import torch

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_ and log_path:
        try:
            with open(log_path, "a+") as f_log:
                f_log.write(s + "\n")
        except Exception as e:
             print(f"Error writing to log file {log_path}: {e}")

def get_logger(log_path, **kwargs):
    if not log_path:
         print("Warning: No log path provided to get_logger. File logging disabled.")
         return functools.partial(logging, log_path=None, log_=False, **kwargs)
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, log_file_path="log.txt", scripts_to_save=None, debug=False):
    if debug:
        print("Debug Mode : no experiment dir created")
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        if not os.path.exists(script_path): os.makedirs(script_path)
        for script in scripts_to_save:
            try:
                dst_file = os.path.join(script_path, os.path.basename(script))
                shutil.copyfile(script, dst_file)
            except Exception as e:
                 print(f"Warning: Could not copy script {script}: {e}")

    if not os.path.isabs(log_file_path):
         log_file_path = os.path.join(dir_path, log_file_path)

    return get_logger(log_path=log_file_path)

def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:
        for param_flag, param_config in params_config[params_category].items():
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)

def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config["dest"]: getattr(namespace, param_config["dest"], None)
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }

def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print("my rank={} local_rank={}".format(rank, local_rank)) # Original print
    torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "world_size": world_size,
    }

def set_up_env(env_params):
    assert torch.cuda.is_available()
    if env_params["distributed"]:
        env_params.update(
            _torch_distributed_init_process_group(local_rank=env_params["local_rank"])
        )
    else:
        env_params["rank"] = 0
        env_params["world_size"] = 1
    env_params["device"] = torch.device("cuda")

def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    # Log only on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("nb_parameters={:.2f}M".format(nb_parameters / 1e6))
    return grad_requiring_params

def _get_optimizer(model, optim_params):
    optim = optim_params["optim"]
    lr = optim_params["lr"]
    params = _get_grad_requiring_params(model)
    if not params:
        print("Warning: No trainable parameters found!")
        return None

    if optim.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif optim.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    else:
        raise RuntimeError("wrong type of optimizer - must be 'sgd' or 'adam'")

def _get_scheduler(optimizer, optim_params):
    lr_warmup = optim_params["lr_warmup"]
    if optimizer is None: return None
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1.0, ep / lr_warmup)
        )
    return None

def get_optimizer_and_scheduler(model, optim_params):
    optimizer = _get_optimizer(model=model, optim_params=optim_params)
    scheduler = _get_scheduler(optimizer=optimizer, optim_params=optim_params)
    return optimizer, scheduler

def _load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger, distributed):
    print("loading from a checkpoint at {}".format(checkpoint_path))

    map_location = 'cuda:%d' % torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}. Cannot resume.")
        return 0

    start_epoch = checkpoint_state.get("epoch", -1) + 1

    # Handle DDP checpoints
    model_state_dict = checkpoint_state.get("model")
    if model_state_dict:
        model_is_wrapped = hasattr(model, 'module')
        checkpoint_has_module_prefix = all(k.startswith('module.') for k in model_state_dict.keys())

        if checkpoint_has_module_prefix and not model_is_wrapped:
            print("Adjusting checkpoint: Removing 'module.' prefix from model state dict keys.")
            model_state_dict = {k.partition('module.')[2]: v for k, v in model_state_dict.items()}
        elif not checkpoint_has_module_prefix and model_is_wrapped:
            print("Adjusting checkpoint: Adding 'module.' prefix to model state dict keys.")
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}

        try:
            incompatible_keys = model.load_state_dict(model_state_dict, strict=True)
            print("Model state loaded successfully.")
        except Exception as e:
            print(f"ERROR loading model state_dict (strict=True): {e}. Check model architecture compatibility.")
    else:
        print("Warning: Checkpoint missing 'model' state dict.")

    if optimizer and "optimizer" in checkpoint_state:
        try:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
            print("Optimizer state loaded.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}.")

    if scheduler and "scheduler" in checkpoint_state:
        try:
            scheduler.load_state_dict(checkpoint_state["scheduler"])
            print("Scheduler state loaded.")
        except Exception as e:
             print(f"Warning: Could not load scheduler state: {e}.")

    if logger and "logger" in checkpoint_state:
        try:
             logger.load_state_dict(checkpoint_state["logger"])
             print("Logger state loaded.")
        except Exception as e:
             print(f"Warning: Could not load logger state: {e}.")

    return start_epoch

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger, distributed, resume):
    if resume and os.path.exists(checkpoint_path):
        return _load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            distributed=distributed,
        )
    elif resume:
        print(f"Resume requested, but checkpoint not found at '{checkpoint_path}'. Starting from scratch.")
    return 0

def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, logger): # Changed nb_batches_per_iter to epoch
    if checkpoint_path:
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        checkpoint_state = {
            "epoch": epoch,
            "model": model_state,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "logger": logger.state_dict() if logger else None,
        }
        try:
            tmp_path = checkpoint_path + ".tmp"
            torch.save(checkpoint_state, tmp_path)
            os.rename(tmp_path, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint to {checkpoint_path}: {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)


class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def log_iter(
        self, epoch_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model=None
    ):
        step = (epoch_no + 1) * nb_batches_per_iter
        self._log(title="step", value=step)
        self._log(title="epoch", value=epoch_no)
        self._log(title="train_loss", value=loss_train)
        self._log(title="val_loss", value=loss_val)
        self._log(title="elapsed_ms_per_batch", value=elapsed)