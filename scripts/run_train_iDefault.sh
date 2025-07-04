#!/bin/bash

# This script is used to run the FRNet model on the ETTh1 dataset for long-term forecasting.
# It sets up the necessary directories, defines parameters, and executes the model training.

export CUDA_VISIBLE_DEVICES=0

# Set the file name and current time for logging
current_time=$(date "+%Y%m%d-%H%M%S")
var_time=$(date "+%Y%M%S")
echo "Var Time : $var_time"
echo "Current Time : $current_time"

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=iTransformer

root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=$var_time
file_name=./results/log-$current_time.log

python -u run_anomaly.py \
  --use_gpu True \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --n_heads 1 \
  --d_model 16 \
  --dropout 0.05\
  --fc_dropout 0.1\
  --head_dropout 0.1\
  --patch_len 24\
  --stride 24\
  --des 'Exp' \
  --train_epochs 100\
  --patience 10\
  --kernel_size 25\
  --lradj type4\
  --pred_head_type 'truncation'\
  --aggregation_type 'avg'\
  --channel_attention 0\
  --global_freq_pred 0\
  --period_list 24 48 72\
  --emb 96\
  --model FRNet \
  --train_epochs 1 \
  --data MSL \
  --chunk_size 1024 \
  --e_layers 3 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --d_ff 128 \
  --dropout 0.0 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --d_model 512 \
  --moving_avg 100 \
  --win_size 100 \
  --gpu 0 \
  --des 'TA' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --result_dir ./results/ \
  --batch_size 128 \
  --learning_rate 0.0003 \
  --itr 1 \
  --is_training 1 >> $file_name
    
     



# for pred_len in 5
# do
#     file_name='logs/LongForecasting/'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len-$current_time'.log'
#     echo $random_seed >> $file_name
#     python -u run_longExp.py \
#      --use_gpu True \
#      --random_seed $random_seed \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id $model_id_name_$seq_len'_'$pred_len \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 7 \
#      --e_layers 1 \
#      --n_heads 1 \
#      --d_model 16 \
#      --d_ff 128 \
#      --dropout 0.05\
#      --fc_dropout 0.1\
#      --head_dropout 0.1\
#      --patch_len 24\
#      --stride 24\
#      --des 'Exp' \
#      --train_epochs 100\
#      --patience 10\
#      --kernel_size 25\
#      --lradj type4\
#      --pred_head_type 'truncation'\
#      --aggregation_type 'avg'\
#      --channel_attention 0\
#      --global_freq_pred 0\
#      --period_list 24 48 72\
#      --emb 96\
#      --itr 1 --batch_size 128 --learning_rate 0.0003 >> $file_name
#done  