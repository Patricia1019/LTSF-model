'''
Author: 
Date: 2023-08-14 03:35:06
LastEditors: peiqi yu
LastEditTime: 2023-08-16 10:57:44
FilePath: /ubuntu/projects/LTSF-Linear/similar_repeat.py
'''
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from data_provider.data_factory_all import data_provider
import pdb
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def sliding_windows(a, W):
    a = np.asarray(a)
    p = np.zeros(W-1,dtype=a.dtype)
    b = np.concatenate((p,a,p))
    s = b.strides[0]
    strided = np.lib.stride_tricks.as_strided
    return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))
class Corr_Similar_prediction():
    def __init__(self,data):
        self.data = data

    def predict(self,predict_x,pred_len):
        corr = -np.inf
        data_len = len(self.data)
        seq_len = len(predict_x)
        if data_len >= seq_len:
            corr = signal.correlate(self.data, predict_x, mode='valid')
            sum = signal.correlate(self.data**2, np.ones(predict_x.shape), mode='valid')
            corr = corr / sum**0.5
        else:
            try:
                corr = signal.correlate(self.data, predict_x[-(data_len//2):], mode='valid')
                sum = signal.correlate(self.data**2, np.ones([data_len//2,predict_x.shape[1]]), mode='valid')
                corr = corr / sum**0.5
            except:
                pdb.set_trace()
        max_index = np.argmax(corr)
        if (data_len-max_index-seq_len) > pred_len:
            output = self.data[max_index+seq_len:max_index+seq_len+pred_len]
        elif (data_len-max_index) > max(pred_len,seq_len):
            output = np.append(self.data[max_index+seq_len:],
                               predict_x[:pred_len-(data_len-max_index-seq_len)],axis=0)
        else:
            output = np.append(self.data[max_index+seq_len:],predict_x,axis=0)
            repeat_num = pred_len // len(output) + 1
            output = output.repeat(repeat_num,axis=0)
            output = output[:pred_len]
        assert len(output) == pred_len
        # shift
        shift = predict_x[-1] - output[0]
        output = torch.tensor(output) + shift
        return output

class NaiveNorm_Corr_Similar_prediction():
    def __init__(self,data):
        self.data = data 

    def predict(self,predict_x,pred_len):
        data_len = len(self.data)
        add_shift = np.array(predict_x[-1])
        if data_len >= len(predict_x):
            predict_x = predict_x - predict_x[0] # norm predict_x naively
            sum_x = predict_x.sum(0)
            seq_len = len(predict_x)
            s = self.data[:data_len-seq_len+1] 
            corr = signal.correlate(self.data, predict_x, mode='valid') -np.expand_dims(np.matmul(s,np.array(sum_x)),1)
            square_sum = signal.correlate(self.data**2, np.ones(predict_x.shape), mode='valid')
            mat=sliding_windows([1]*seq_len, W=data_len-seq_len+1)
            all_sum = np.matmul(mat,self.data)
            sum = square_sum - 2*np.expand_dims((s*all_sum).sum(1),1) + seq_len*np.expand_dims((s**2).sum(1),1)
            corr = corr / sum**0.5
        else:
            try:
                predict_x = predict_x[-(data_len//2):]
                predict_x = predict_x - predict_x[0] # norm predict_x naively
                sum_x = predict_x.sum(0)
                seq_len = len(predict_x)
                s = self.data[:data_len-seq_len+1] 
                corr = signal.correlate(self.data, predict_x, mode='valid') -np.expand_dims(np.matmul(s,np.array(sum_x)),1)
                square_sum = signal.correlate(self.data**2, np.ones(predict_x.shape), mode='valid')
                mat=sliding_windows([1]*seq_len, W=data_len-seq_len+1)
                all_sum = np.matmul(mat,self.data)
                sum = square_sum - 2*np.expand_dims((s*all_sum).sum(1),1) + seq_len*np.expand_dims((s**2).sum(1),1)
                corr = corr / sum**0.5
            except:
                pdb.set_trace()
        max_index = np.argmax(corr)
        sub_shift = self.data[max_index]
        if (data_len-max_index-seq_len) > pred_len:
            output = self.data[max_index+seq_len:max_index+seq_len+pred_len]
        elif (data_len-max_index) > max(pred_len,seq_len):
            output = np.append(self.data[max_index+seq_len:],
                               predict_x[:pred_len-(data_len-max_index-seq_len)],axis=0)
        else:
            output = np.append(self.data[max_index+seq_len:],predict_x,axis=0)
            repeat_num = pred_len // len(output) + 1
            output = output.repeat(repeat_num,axis=0)
            output = output[:pred_len]
        # shift
        out_shift = add_shift - output[0]
        output = output + out_shift
        assert len(output) == pred_len
        return output

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

setting = '{}_{}_{}'.format(args.model_id,
                            args.model,
                            args.data)

print('Args in experiment:')
print(args)

Exp = Exp_Main

train_dataset, train_dataloader = data_provider(args, flag='train')
test_dataset, test_dataloader = data_provider(args, flag='test')
preds = []
trues = []
inputx = []
folder_path = f'./test_results/{args.model}/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if args.model == 'similar_repeat': 
    similar_prediction = Corr_Similar_prediction(train_dataset.data_x)
elif args.model == 'norm_similar_repeat': 
    similar_prediction = NaiveNorm_Corr_Similar_prediction(train_dataset.data_x)
for i, (batch_x, batch_y) in enumerate(tqdm(test_dataloader)):
    pred_y = similar_prediction.predict(batch_x[0],args.pred_len)
    pred_y = np.expand_dims(pred_y,0)
    preds.append(pred_y)
    trues.append(batch_y[0].unsqueeze(0))
    inputx.append(batch_x[0].unsqueeze(0).detach().cpu().numpy())
    if i % 20 == 0:
        input = batch_x.detach().cpu().numpy()
        gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
        pd = np.concatenate((input[0, :, -1], pred_y[0, :, -1]), axis=0)
        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
inputx = np.concatenate(inputx, axis=0)

# result save
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
print('mse:{}, mae:{}'.format(mse, mae))
f = open("result.txt", 'a')
f.write(setting + "  \n")
f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
f.write('\n')
f.write('\n')
f.close()

# np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
np.save(folder_path + 'pred.npy', preds)
# np.save(folder_path + 'true.npy', trues)
# np.save(folder_path + 'x.npy', inputx)

torch.cuda.empty_cache()

