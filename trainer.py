# --- START OF REVERTED FILE trainer.py ---
import os, sys
import argparse
import math, random
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Import BaseGate to check for gate loss attribute
from gates import BaseGate # Assuming gates.py defines BaseGate

def _train_step(model, load_balance_coeff, X, Y, h_cache, eval_only, loss_div=1):
    """Single training step, using only attention cache (h_cache)."""

    # Model forward pass (assuming signature: model(X, h_cache))
    # Momentum state is handled internally by MoE layers.
    # Returns: out_logits, updated_h_cache
    out, updated_h_cache = model(X, h_cache) # Pass only attention cache

    # Loss calculation
    out_flat = out.view(-1, out.size(-1))
    Y_flat = Y.view(-1)
    loss = F.nll_loss(out_flat, Y_flat) # Assumes model outputs log-softmax
    loss_value = loss.item() # Store base loss value for reporting

    if not eval_only:
        # Add auxiliary MoE load balancing loss
        if load_balance_coeff > 0:
            balance_loss = 0
            # Access model potentially wrapped in DDP/DP
            model_to_inspect = model.module if hasattr(model, 'module') else model
            for name, m in model_to_inspect.named_modules():
                # Check if it's a gate with a stored loss
                # Make sure BaseGate and get_loss are defined correctly
                if isinstance(m, BaseGate) and hasattr(m, 'get_loss'):
                     gate_loss = m.get_loss(clear=True) # Retrieve and clear loss
                     if gate_loss is not None:
                         balance_loss += gate_loss

            if isinstance(balance_loss, torch.Tensor): # Add only if found
                 loss += load_balance_coeff * balance_loss

        # Backward pass on the potentially combined loss
        (loss / loss_div).backward()

    # Return the NLL loss value (before aux loss), scaled for accumulation
    return loss_value / loss_div, updated_h_cache


def _train_batch(
    model,
    load_balance_coeff,
    optimizer,
    scheduler,
    X,
    Y,
    h_cache, # Only attention cache
    eval_only,
    batch_split,
):
    """Train on a batch, handling accumulation and attention cache."""

    if not eval_only:
        optimizer.zero_grad()

    if batch_split == 1:
        # Process batch in one step
        loss_value, updated_h_cache = _train_step(
            model, load_balance_coeff, X, Y, h_cache, eval_only
        )
    else:
        # Gradient accumulation over splits
        assert X.size(0) % batch_split == 0, "Batch size not divisible by batch_split"
        split_size = X.size(0) // batch_split
        loss_value = 0
        # Cache state evolves through the splits
        h_cache_list = [] # To store updated caches from splits

        for split_ind in range(batch_split):
            split_slice = slice(split_ind * split_size, (split_ind + 1) * split_size)

            # Prepare cache for the current split
            split_input_cache = [h[split_slice, :, :] if h is not None else None for h in h_cache]

            # Perform train step on the split (backward happens inside)
            split_loss_value, split_updated_cache = _train_step(
                model,
                load_balance_coeff,
                X[split_slice, :],
                Y[split_slice],
                split_input_cache,
                eval_only,
                batch_split, # Pass loss division factor
            )
            loss_value += split_loss_value # Accumulate average loss
            h_cache_list.append(split_updated_cache) # Store the updated cache

        # Combine the updated caches from all splits
        # Assumes h_cache is a list of tensors (or None)
        num_cache_layers = len(h_cache)
        updated_h_cache = []
        for l in range(num_cache_layers):
            if h_cache_list[0][l] is not None: # Check if cache exists for this layer
                 updated_h_cache.append(torch.cat([h_cache_list[i][l] for i in range(batch_split)], dim=0))
            else:
                 updated_h_cache.append(None)


    # Optimizer and scheduler steps after processing all splits
    if not eval_only:
        # Optional: Add gradient clipping here if needed
        # torch.nn.utils.clip_grad_norm_(model.parameters(), optim_params['grad_clip'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Adaptive span clamp removed

    # Return accumulated average loss and final cache state
    return loss_value, updated_h_cache


def train_iteration(
    model,
    load_balance_coeff,
    optimizer,
    scheduler,
    data,
    nb_batches_per_iter,
    block_size,
    eval_only,
    train_pos,
    h_cache, # Accepts only attention cache
    batch_split,
):
    """Single training iteration (epoch pass)."""
    if eval_only:
        model.eval()
    else:
        model.train()

    # Determine number of batches based on data size and block_size
    max_data_batches = math.ceil((data.size(1) - 1) / block_size)
    nb_batches_to_run = min(nb_batches_per_iter, max_data_batches)
    if nb_batches_to_run <= 0:
        print("Warning: Not enough data for one block in train_iteration. Skipping.")
        return 0.0, train_pos, h_cache

    total_loss = 0
    actual_batches_run = 0 # Track batches actually processed

    # Setup progress bar
    mode = "Eval" if eval_only else "Train"
    pbar = tqdm.tqdm(range(nb_batches_to_run), desc=f"{mode} Iteration", leave=False)

    for i in pbar:
        # Check if enough data remains for a full X and Y block
        if train_pos >= data.size(1) - block_size - 1:
            break # Exit loop if data runs out

        actual_batches_run += 1
        batch_start = train_pos
        batch_end = batch_start + block_size
        X = data[:, batch_start:batch_end].contiguous()
        # Target is shifted by one position
        Y = data[:, batch_start + 1 : batch_end + 1].contiguous()

        # Handle potential empty batch if data length is exactly block_size multiple
        if X.size(1) == 0 or Y.size(1) == 0:
            actual_batches_run -= 1 # Don't count this batch
            continue

        # Align X length if Y is shorter at the end of the dataset
        if Y.size(1) < X.size(1):
            X = X[:, :Y.size(1)]

        # Process the batch
        batch_loss, h_cache = _train_batch( # Pass and receive only h_cache
            model=model,
            load_balance_coeff=load_balance_coeff,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache, # Pass only attention cache
            eval_only=eval_only,
            batch_split=batch_split,
        )

        # Check for invalid loss and accumulate
        if not math.isfinite(batch_loss):
             print(f"Warning: Encountered non-finite loss ({batch_loss}) at batch {i+1}. Stopping iteration.")
             total_loss = float('nan')
             break
        total_loss += batch_loss
        train_pos += X.size(1) # Move data pointer by processed sequence length

        # Update progress bar display
        if total_tokens > 0:
            pbar.set_postfix(loss=f"{total_loss / actual_batches_run:.4f}")


    # Reset position and clear cache if data ended mid-iteration
    if train_pos >= data.size(1) - block_size - 1:
        train_pos = random.randrange(block_size // 2) # Random start within first half block
        print(f"Resetting data position to {train_pos} and clearing cache.")
        # Clear the attention cache
        if h_cache:
            for h_attn in h_cache:
                if h_attn is not None: h_attn.fill_(0)

    # Calculate average loss for the batches processed in this iteration
    avg_loss = total_loss / actual_batches_run if actual_batches_run > 0 else 0
    return avg_loss, train_pos, h_cache


def full_eval(model, data, model_params, device):
    """Full evaluation pass over the entire dataset."""
    model.eval() # Set model to evaluation mode
    train_pos = 0
    # Get necessary parameters from dict
    block_size = model_params["block_size"]
    hidden_size = model_params["hidden_size"]
    batch_size = data.size(0) # Batch size is dim 0 from batchified data

    # Initialize attention cache (h_cache)
    def initialize_eval_attn_cache(current_batch_size, model_instance, device):
        attn_cache_list = []
        model_to_inspect = model_instance.module if hasattr(model_instance, 'module') else model_instance
        for layer in model_to_inspect.layers: # Assumes model has .layers
            attn_cache_list.append(
                torch.zeros(current_batch_size, layer.attn.get_cache_size(), hidden_size, device=device)
                if layer.use_attn else None # Assumes layer has .use_attn
            )
        return attn_cache_list

    h_cache = initialize_eval_attn_cache(batch_size, model, device)

    total_loss = 0
    total_tokens = 0 # Use token count for accurate BPC/PPL

    # Calculate max possible batches
    max_batches = math.ceil((data.size(1) - 1) / block_size)
    pbar = tqdm.tqdm(range(max_batches), desc="Full Eval", leave=False)

    with torch.no_grad(): # Ensure no gradients are computed
        for i in pbar:
            # Prepare batch data
            batch_start = train_pos
            batch_end = batch_start + block_size
            X = data[:, batch_start:batch_end].contiguous()
            Y = data[:, batch_start + 1 : batch_end + 1].contiguous()

            # Check end of data / empty batch
            if X.size(1) == 0 or Y.size(1) == 0: break
            # Align X length if Y is shorter
            if Y.size(1) < X.size(1): X = X[:, :Y.size(1)]

            current_batch_tokens = Y.numel() # Number of target tokens in this batch
            if current_batch_tokens == 0: continue # Skip if somehow target is empty

            # Perform evaluation step using _train_batch in eval mode
            batch_loss_avg, h_cache = _train_batch( # Pass and receive only h_cache
                model=model,
                load_balance_coeff=0, # No balance loss during eval
                optimizer=None, # No optimizer needed
                scheduler=None, # No scheduler needed
                X=X,
                Y=Y,
                h_cache=h_cache, # Pass only attention cache
                eval_only=True,
                batch_split=1, # Process full batch at once
            )

            total_loss += batch_loss_avg * current_batch_tokens
            total_tokens += current_batch_tokens
            train_pos += X.size(1)

            if total_tokens > 0:
                 pbar.set_postfix(avg_loss=f"{total_loss/total_tokens:.4f}")

            if train_pos >= data.size(1) - 1:
                break

    final_avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return final_avg_loss