#!/bin/bash

mkdir -p /kaggle/working/checkpoints
args="
--data /kaggle/working/data/wikitext-103/ \
--architecture smsmsm \
--gate_name smoe \
--num-layers 3 \
--hidden-size 128 \
--inner-hidden-size 128 \
--num_heads 8 \
--mhmoe_num_heads 1 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--epochs 60 \
--batch-size 96 \
--batch-split 6 \
--nbatches 1000 \
--distributed \
--gamma1 0.8 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /kaggle/working/checkpoints/smoe.pt \
"
echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=2 --use_env train.py $args
echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=2 --use_env train.py $args --resume --full-eval-mode