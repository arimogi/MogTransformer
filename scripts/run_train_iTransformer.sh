#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

current_time=$(date "+%Y%m%d-%H%M%S")
var_time=$(date "+%Y%M%S")

echo "Var Time : $var_time"
echo "Current Time : $current_time"

mkdir -p ./logs/iTransformer
mkdir -p ./results/

model_name=iTransformer
random_seed=$var_time
log_file=./logs/iTransformer/log-$current_time.log

python -u run_anomaly.py \
  --use_gpu True \
  --random_seed $random_seed \
  --is_training 1 \
  --model $model_name \
  --data MSL \
  --root_path ./dataset/MSL/ \
  --anomaly_ratio 0.85 \
  --seq_len 100 \
  --moving_avg 100 \
  --win_size 100 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --d_model 512 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 5 \
  --dropout 0.0 \
  --train_epochs 1 \
  --batch_size 128 \
  --learning_rate 0.0003 \
  --gpu 0 \
  --des 'iTransformer_MSL' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --result_dir ./results/ \
  --run_chunk True \
  --itr 1 >> $log_file
