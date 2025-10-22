#!/bin/bash

MODEL=/mnt/nushare2/data/baliao/PLLMs/Gen-Verse/TraDo-4B-Instruct
DATA=hendrydong/fineweb-edu-10BT

for block_len in 4; do
    echo "Evaluating block_length=${block_len}"
    python sample/ppl.py \
        --model_name_or_path ${MODEL} \
        --tensor_parallel_size 1 \
        --eval_dataset ${DATA} \
        --block_length ${block_len} \
        --batch_size 16 \
        --max_samples 100 \
        --max_length 4096 \
        --tensor_parallel_size 1 \
done