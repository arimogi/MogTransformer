export CUDA_VISIBLE_DEVICES=0

python -u train_maelnet.py \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id Model_MaelNet_Training\
  --model MaelNet \
  --is_slow_learner true \
  --train_epochs 1 \
  --data MSL \
  --chunk_size 512 \
  --e_layers 1 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --d_ff 512 \
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
  --patch_size 5 \
  --channel 55 \
  --result_dir ./results/ \
  --itr 1  
  