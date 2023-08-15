
###
 # @Author: 
 # @Date: 2023-08-14 14:40:41
 # @LastEditors: peiqi yu
 # @LastEditTime: 2023-08-14 14:48:17
 # @FilePath: /ubuntu/projects/LTSF-Linear/scripts/EXP-LongForecasting/similar_repeat/ettm2.sh
### 
###
 # @Author: 
 # @Date: 2023-08-14 14:40:41
 # @LastEditors: peiqi yu
 # @LastEditTime: 2023-08-14 14:47:17
 # @FilePath: /ubuntu/projects/LTSF-Linear/scripts/EXP-LongForecasting/similar_repeat/ettm2.sh
### 
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=similar_repeat

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'96.log

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'192.log

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'336.log

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'720.log
