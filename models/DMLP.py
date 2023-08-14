'''
Author: 
Date: 2023-08-11 01:56:27
LastEditors: peiqi yu
LastEditTime: 2023-08-13 12:13:38
FilePath: /ubuntu/projects/LTSF-Linear/models/DMLP.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-MLP
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hid_len = 16
        self.hid_layers = 2

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.MLP_Seasonal = nn.ModuleList()
            self.NLP_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.MLP_Seasonal.append(self.create_MLP_modules())
                self.MLP_Trend.append(self.create_MLP_modules())

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.MLP_Seasonal = self.create_MLP_modules()
            self.MLP_Trend = self.create_MLP_modules()
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def create_MLP_modules(self):
        modules = nn.ModuleList([nn.Linear(self.seq_len, self.hid_len)])
        modules.append(nn.ReLU())
        for i in range(self.hid_layers):
            modules.append(nn.Linear(self.hid_len, self.hid_len))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.hid_len, self.pred_len))
        return modules

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # pdb.set_trace()
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_tmp = seasonal_init[:,i,:]
                trend_tmp = trend_init[:,i,:]
                for layer in self.MLP_Seasonal[i]:
                    seasonal_tmp = layer(seasonal_tmp)
                for layer in self.MLP_Trend[i]:
                    trend_tmp = layer(trend_tmp)
                seasonal_output[:,i,:] = seasonal_tmp
                trend_output[:,i,:] = trend_tmp
        else:
            seasonal_output = seasonal_init
            trend_output = trend_init
            for layer in self.MLP_Seasonal:
                seasonal_output = layer(seasonal_output)
            for layer in self.MLP_Trend:
                trend_output = layer(trend_output)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
