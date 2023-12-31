
###
 # @Author: 
 # @Date: 2023-08-13 12:15:14
 # @LastEditors: peiqi yu
 # @LastEditTime: 2023-08-15 03:17:07
 # @FilePath: /ubuntu/projects/LTSF-Linear/scripts/EXP-LongForecasting/DMLP/exchange_rate_DMLP.sh
### 
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DMLP

python -u run_MLP.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 5 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log 

python -u run_MLP.py \
  --is_training 1 \
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
  --itr 5 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'192.log 

python -u run_MLP.py \
  --is_training 1 \
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
  --itr 5 --batch_size 32  --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'336.log 

python -u run_MLP.py \
  --is_training 1 \
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
  --itr 5 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'720.log
