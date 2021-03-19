#!/bin/bash

EXPERIMENT_DIR=./experiments/$(date +"%Y%m%d_%H%M")_$(git rev-parse --short HEAD)

python run_mlm.py \
    --model_name_or_path bert-base-cased \
    --train_file ./ing_ins_rec_doc_cased.txt \
    --do_train \
    --output_dir $EXPERIMENT_DIR
