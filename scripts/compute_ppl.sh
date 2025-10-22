#!/bin/bash

MODEL=/mnt/nushare2/data/baliao/PLLMs/Gen-Verse/TraDo-4B-Instruct
DATA=/mnt/nushare2/data/baliao/dlm/data/hendrydong/test.json
SAVE_DIR=/mnt/nushare2/data/baliao/dlm/00_start/ppl_results/Gen-Verse/TraDo-4B-Instruct

mkdir -p ${SAVE_DIR}

for block_len in 4; do
    echo "Evaluating block_length=${block_len}"
    python sample/ppl.py \
        --model_name_or_path ${MODEL} \
        --tensor_parallel_size 1 \
        --dataset_path ${DATA} \
        --block_length ${block_len} \
        --batch_size 16 \
        --max_samples 1 \
        --max_length 2048 \
        --tensor_parallel_size 1 \
        --output_file ${SAVE_DIR}/blocklen_${block_len}.json
done