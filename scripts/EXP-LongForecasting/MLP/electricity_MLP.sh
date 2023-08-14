###
 # @Author: 
 # @Date: 2023-08-11 10:01:35
 # @LastEditors: peiqi yu
 # @LastEditTime: 2023-08-13 12:00:37
 # @FilePath: /ubuntu/projects/LTSF-Linear/scripts/EXP-LongForecasting/Linear/test.sh
### 
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MLP

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'96.log 
