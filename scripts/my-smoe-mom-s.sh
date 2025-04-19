%%writefile scripts/my-smoe-mom-s.sh
#!/bin/bash

# Create output directory
OUTPUT_DIR="/kaggle/working/output/smoe-mom-s"
mkdir -p $OUTPUT_DIR

# Define arguments based on config.py and user clarification
args="
--data /kaggle/working/data/wikitext-103/ \
--output-dir ${OUTPUT_DIR} \
--architecture mmm \
--gate_name smoe \
--num-layers 3 \
--hidden-size 128 \
--inner-hidden-size 128 \
--num-attn-heads 8 \
--num-heads 1 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--num-experts 16 \
--moe_top_k 2 \
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
--momentum_gamma 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--resume \
"

echo "Training (and evaluating at the end)..."
# Launch distributed training using torch.distributed.run
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node=2 --master_port 10013 train.py $args

echo "Script finished."