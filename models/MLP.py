'''
Author: 
Date: 2023-08-13 08:44:12
LastEditors: peiqi yu
LastEditTime: 2023-08-13 11:58:22
FilePath: /ubuntu/projects/LTSF-Linear/models/MLP.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Model(nn.Module):
    """
    Multiple Linear layers
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hid_len = 16
        self.hid_layers = 2
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual

        if self.individual:
            self.MLP = nn.ModuleList()
            for i in range(self.channels):
                self.MLP.append(self.create_MLP_modules())
        else:
            self.MLP = self.create_MLP_modules()

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
        if self.individual:
            out = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                tmp = x[:,:,i]
                for layer in self.MLP[i]:
                    tmp = self.MLP[i](tmp)
                out[:,:,i] = tmp
        else:
            inp = x.permute(0,2,1)
            for layer in self.MLP:
                inp = layer(inp)
            out = inp.permute(0,2,1)
        return out # [Batch, Output length, Channel]
