#!/bin/bash

seq_len=96
pred_len=96
data_path="../Time-Series-Library/dataset"

model=LinearPFN
model_id=from_paper
ckpts_root=output
ckpt_file=best_model.pt
train_budget=1.0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seq_len) seq_len="$2"; shift ;;
        --pred_len) pred_len="$2"; shift ;;
        --data_path) data_path="$2"; shift ;;
        --model) model="$2"; shift ;;
        --model_id) model_id="$2"; shift ;;
        --ckpts_root) ckpts_root="$2"; shift ;;
        --ckpt_file) ckpt_file="$2"; shift ;;
        --train_budget) train_budget="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python run.py \
  --data traffic \
  --root_path $data_path/traffic/ \
  --data_path traffic.csv \
  --model_id $model_id \
  --model $model \
  --ckpts_root $ckpts_root\
  --ckpt_file $ckpt_file\
  --features S \
  --seq_len $seq_len --label_len 36 \
  --pred_len $pred_len \
  --train_budget $train_budget