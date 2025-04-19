import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time
import datetime
import wandb # Assuming wandb is installed

# Use your config structure
from config import PARAMS_CONFIG
# Use your data loading
from data import get_train_val_test_data
# Use your model definition
from models import TransformerSeq
# Use your trainer logic (now expecting only h_cache)
from trainer import train_iteration, full_eval
# Use your utils
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    Logger,
)

def launch(
    env_params,
    model_params,
    # adapt_span_params, # Removed
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    # --- WandB Setup ---
    wandb_flag = bool(wandb_params.get("wandb_key")) # Check if key is provided
    if wandb_flag:
        try:
            wandb.login(key=wandb_params["wandb_key"])
            wandb.init(
                project=wandb_params["project_name"],
                name=wandb_params.get("run_name"), # Use generated name if None
                config={ # Log all relevant parameters
                    "env_params": env_params,
                    "model_params": model_params,
                    "optim_params": optim_params,
                    "data_params": data_params,
                    "trainer_params": trainer_params,
                }
            )
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"WandB initialization failed: {e}. Disabling WandB.")
            wandb_flag = False

    # --- Environment Setup ---
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]
    rank = env_params["rank"]
    world_size = env_params["world_size"]
    resume = trainer_params["resume"]

    # --- Create Output Directory and Logger ---
    output_dir = trainer_params["output_dir"]
    checkpoint_path = trainer_params["checkpoint_path"]
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
        trainer_params["checkpoint_path"] = checkpoint_path
    else:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    logger = Logger()
    logging = None
    if rank == 0:
        logging = create_exp_dir(output_dir, log_file_path="train_log.txt")
        logging(f"Output Directory: {output_dir}")
        logging(f"Checkpoint Path: {checkpoint_path}")
        logging(f"Start Time: {datetime.datetime.now()}")
        logging("--- Parameters ---")
        logging(f"env_params: {env_params}")
        logging(f"data_params: {data_params}")
        logging(f"model_params: {model_params}")
        logging(f"optim_params: {optim_params}")
        logging(f"trainer_params: {trainer_params}")
        logging(f"wandb_params: {wandb_params}")
        logging("------------------")

    # --- Data Loading ---
    global_batch_size = trainer_params["batch_size"] * world_size
    if rank == 0: logging(f"Global batch size: {global_batch_size}")

    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=global_batch_size,
        device=device,
    )
    model_params["vocab_size"] = data_params["vocab_size"]
    if rank == 0: logging(f"Vocabulary Size: {model_params['vocab_size']}")

    # --- Model Initialization ---
    model = TransformerSeq(
        **model_params,
        # adapt_span_params=None, # Removed
    )

    if rank == 0:
        logging(f"--- Model ---")
        logging(str(model))
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging(f"Total Parameters: {total_params / 1e6:.2f} M")
        logging(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        logging(f"-------------")

    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[env_params["local_rank"]],
            output_device=env_params["local_rank"],
            find_unused_parameters=False,
        )
    else:
        model = torch.nn.DataParallel(model)


    # --- Optimizer and Scheduler ---
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # --- Resume from Checkpoint ---
    start_epoch = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume,
    )
    if rank == 0: logging(f"Resuming from epoch: {start_epoch}")

    # --- Training Loop ---
    best_val_loss = None
    start_time = time.time()
    num_epochs = trainer_params["epochs"]
    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]

    # Initialize attention cache (h_cache) structure
    def initialize_attn_cache(current_batch_size, model_instance, device):
        attn_cache_list = []
        model_to_inspect = model_instance.module if hasattr(model_instance, 'module') else model_instance
        for layer in model_to_inspect.layers:
             attn_cache_list.append(
                 torch.zeros(current_batch_size, layer.attn.get_cache_size(), model_params["hidden_size"], device=device)
                 if layer.use_attn else None
             )
        return attn_cache_list

    # Initialize caches for train and valid - use per-device batch size
    per_device_batch_size = trainer_params["batch_size"]
    # Use hid_cache naming convention like reference
    hid_cache_train = initialize_attn_cache(per_device_batch_size, model, device)
    hid_cache_valid = initialize_attn_cache(per_device_batch_size, model, device)
    train_data_pos = 0
    valid_data_pos = 0


    for epoch_no in range(start_epoch, num_epochs):
        if rank == 0: logging(f"\n===== Starting Epoch {epoch_no} =====")
        epoch_start_time = time.time()

        # --- Training Iteration ---
        t_sta = time.time()
        loss_train, train_data_pos, hid_cache_train = train_iteration( # Pass/receive hid_cache_train
            model=model,
            load_balance_coeff=model_params["load_balance"],
            optimizer=optimizer,
            scheduler=scheduler,
            data=train_data,
            nb_batches_per_iter=nb_batches_per_iter,
            block_size=trainer_params["block_size"],
            eval_only=False,
            train_pos=train_data_pos,
            h_cache=hid_cache_train, # Pass only attention cache
            batch_split=trainer_params["batch_split"],
        )
        train_time_per_batch = (time.time() - t_sta) * 1000 / nb_batches_per_iter if nb_batches_per_iter > 0 else 0

        # --- Validation Iteration ---
        t_sta = time.time()
        val_batches = math.ceil((val_data.size(1) - 1) / trainer_params["block_size"])
        if val_batches > 0:
            loss_val, valid_data_pos, hid_cache_valid = train_iteration( # Pass/receive hid_cache_valid
                model=model,
                load_balance_coeff=model_params.get("load_balance", 0), # Use 0 during eval if needed
                optimizer=None,
                scheduler=None,
                data=val_data,
                nb_batches_per_iter=val_batches,
                block_size=trainer_params["block_size"],
                eval_only=True,
                train_pos=valid_data_pos,
                h_cache=hid_cache_valid, # Pass only attention cache
                batch_split=trainer_params["batch_split"],
            )
        else:
            loss_val = float('nan')
            if rank == 0: logging("Warning: Not enough validation data for one block.")

        val_time_per_batch = (time.time() - t_sta) * 1000 / val_batches if val_batches > 0 else 0

        # --- Collect Results (Distributed) ---
        if distributed:
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0, op=torch.distributed.ReduceOp.AVG)
            if rank == 0:
                loss_train = stats[0].item()
                loss_val = stats[1].item()
            else:
                continue # Non-rank-0 processes skip logging/saving

        # --- Logging (Rank 0 Only) ---
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            data_name_heuristic = os.path.basename(data_params["data_path"].strip('/'))
            is_char_level = "enwik8" in data_name_heuristic or "text8" in data_name_heuristic

            if is_char_level:
                train_metric = loss_train / math.log(2) if math.isfinite(loss_train) else float('inf')
                val_metric = loss_val / math.log(2) if math.isfinite(loss_val) else float('inf')
                metric_name = "BPC"
            else:
                try: train_metric = math.exp(loss_train)
                except: train_metric = float('inf') # Catch potential math errors
                try: val_metric = math.exp(loss_val)
                except: val_metric = float('inf')
                metric_name = "PPL"

            log_msg = (
                f"Epoch {epoch_no} | Train Loss: {loss_train:.4f} ({train_metric:.2f} {metric_name}) | "
                f"Val Loss: {loss_val:.4f} ({val_metric:.2f} {metric_name}) | "
                f"Train ms/batch: {train_time_per_batch:.1f} | Val ms/batch: {val_time_per_batch:.1f} | "
                f"Epoch Time: {epoch_time:.2f}s"
            )
            logging(log_msg)

            if wandb_flag:
                log_dict = {
                    "epoch": epoch_no,
                    "train_loss": loss_train,
                    f"train_{metric_name.lower()}": train_metric,
                    "val_loss": loss_val,
                    f"val_{metric_name.lower()}": val_metric,
                    "train_ms_per_batch": train_time_per_batch,
                    "val_ms_per_batch": val_time_per_batch,
                    "epoch_time_s": epoch_time,
                }
                if scheduler: log_dict["learning_rate"] = scheduler.get_last_lr()[0]
                elif optimizer: log_dict["learning_rate"] = optimizer.param_groups[0]['lr']
                wandb.log(log_dict)

            logger.log_iter(epoch_no, nb_batches_per_iter, loss_train, loss_val, train_time_per_batch, model)

            # --- Checkpoint Saving (Rank 0 Only) ---
            if (best_val_loss is None) or (math.isfinite(loss_val) and loss_val < best_val_loss):
                 if math.isfinite(loss_val):
                     best_val_loss = loss_val
                     logging(f"*** New best validation loss: {best_val_loss:.4f}. Saving checkpoint to {checkpoint_path} ***")
                     save_checkpoint(
                         checkpoint_path, epoch_no, model, optimizer, scheduler, logger
                     )
                 else:
                     logging(f"Warning: Current validation loss ({loss_val:.4f}) is not finite. Not saving checkpoint.")

    # --- End of Training ---
    if rank == 0:
        total_training_time = time.time() - start_time
        logging(f"\n===== Training Finished =====")
        logging(f"Total training time: {total_training_time / 3600:.2f} hours")
        logging(f"Best validation loss: {best_val_loss:.4f}")

        # --- Final Evaluation on Test Set (Rank 0 Only) ---
        logging(f"\n===== Final Evaluation on Test Set =====")
        if test_data.size(1) > 1:
            test_loss = full_eval(
                model=model,
                data=test_data,
                model_params=model_params,
                device=device,
            )

            if is_char_level:
                test_metric = test_loss / math.log(2) if math.isfinite(test_loss) else float('inf')
            else:
                try: test_metric = math.exp(test_loss)
                except: test_metric = float('inf')

            logging(f"Test Loss: {test_loss:.4f} ({test_metric:.2f} {metric_name})")
            if wandb_flag:
                wandb.log({
                    "final_test_loss": test_loss,
                    f"final_test_{metric_name.lower()}": test_metric,
                    "best_val_loss": best_val_loss,
                 })
        else:
            logging("No test data found or test data too small. Skipping final test evaluation.")

    if wandb_flag and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    params = get_params(params_config=PARAMS_CONFIG)
    launch(
        env_params=params["env_params"],
        model_params=params["model_params"],
        optim_params=params["optim_params"],
        data_params=params["data_params"],
        trainer_params=params["trainer_params"],
        wandb_params=params["wandb_params"],
        # adapt_span_params=None, # Removed
    )