###
 # @Author: 
 # @Date: 2023-08-14 08:16:28
 # @LastEditors: peiqi yu
 # @LastEditTime: 2023-08-16 09:53:32
 # @FilePath: /ubuntu/projects/LTSF-Linear/scripts/EXP-LongForecasting/similar_repeat/exchange_rate.sh
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
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log 

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'192.log 

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32  --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'336.log 

python -u similar_repeat.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'720.log
