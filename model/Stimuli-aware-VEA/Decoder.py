#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from torchvision import models
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")

class Attention(nn.Module):
    def __init__(self,feat_size,hidden_size,att_size,dropout=0.5):
        super(Attention,self).__init__()

        self.feats_att=weight_norm(nn.Linear(feat_size, att_size))
        self.hiddens_att=weight_norm(nn.Linear(hidden_size, att_size))
        # self.hiddens_att = weight_norm(nn.Linear(hidden_size+feat_size, feat_size))
        self.full_att = weight_norm(nn.Linear(att_size, 1))
        # self.full_att = weight_norm(nn.Linear(feat_size, 1))
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats, hiddens):

        att1=self.feats_att(feats)
        # att1=feats
        att2=self.hiddens_att(hiddens)

        att=self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)
        # att = self.full_att(self.dropout(self.tanh(att1 + att2.unsqueeze(1)))).squeeze(2)
        alpha = self.softmax(att)
        atted_feats = (feats * alpha.unsqueeze(2)).sum(dim=1)

        return atted_feats, alpha


class Decoder(nn.Module):
    def __init__(self,feat_size,hidden_size):
        super(Decoder,self).__init__()
        
        self.minval=-0.1
        self.maxval=0.1
        self.fc_size=1024
        self.dropout=0.5
        self.batch_size=None
        self.hidden_size=hidden_size
        self.feat_size=feat_size
        self.maxlen=None

        self.LSTM1=nn.LSTMCell(self.feat_size+self.hidden_size,self.hidden_size,bias=True)
        self.LSTM2=nn.LSTMCell(self.feat_size+self.hidden_size,self.hidden_size,bias=True)
        # self.LSTM1 = nn.LSTMCell(self.feat_size, self.hidden_size, bias=True)
        # self.LSTM2=nn.LSTMCell(self.hidden_size,self.hidden_size,bias=True)
        self.Att=Attention(self.feat_size,self.hidden_size,self.hidden_size)
        # self.Att1=Attention(self.feat_size,self.hidden_size,self.hidden_size)
        # self.Att2 = Attention(self.hidden_size, self.hidden_size, self.hidden_size)
        ''' 
        self.f2h=nn.Sequential(nn.Linear(feat_size, self.fc_size),
                               #nn.ReLU(),
                               nn.Dropout(p=self.dropout),
                               nn.Linear(self.fc_size, hidden_size)) 
        self.classfier=nn.Sequential(nn.Linear(hidden_size, self.fc_size),
                                     #nn.ReLU(),
                                     nn.Dropout(p=self.dropout),
                                     nn.Linear(self.fc_size, vocab_size)) 
        
        self.f2h=nn.Sequential(nn.Linear(feat_size, self.hidden_size),
                               #nn.ReLU(),
                               nn.Dropout(p=self.dropout),
                               nn.Linear(self.hidden_size, hidden_size)) 
        '''
        '''
        self.classfier1=nn.Sequential(#nn.Linear(hidden_size, self.hidden_size),
                                     #nn.ReLU(),
                                     nn.Dropout(p=self.dropout),
                                     nn.Linear(self.hidden_size, 2)) 
        '''
        '''
        self.classfier=nn.Sequential(nn.Linear(hidden_size, self.hidden_size),
                                     #nn.ReLU(),
                                     nn.Dropout(p=self.dropout),
                                     nn.Linear(self.hidden_size, class_num))
        '''
        self.weights_init()
    
    def forward(self, feats):
        self.batch_size = feats.shape[0]
        h1, c1 = self.init_hidden_state(self.batch_size)
        h2, c2 = self.init_hidden_state(self.batch_size)
        mean_feat = feats.mean(1)
        thought_vectors = torch.zeros(self.batch_size, 3, self.hidden_size).to(device)
        alpha_mat = torch.zeros(self.batch_size, 10).to(device)
        for t in range(1):
            h1, c1 = self.LSTM1(torch.cat([mean_feat, h2], dim=1), (h1, c1))
            att_feats,alpha = self.Att(feats, h1)
            alpha_mat=alpha
            h2, c2 = self.LSTM2(torch.cat([att_feats, h1], dim=1), (h2, c2))
            thought_vectors[:, t, :] = h2 # save every h2, we can calculate sum and ...
            # h2 = h2.abs()
        '''
        thought_vectors=torch.zeros(self.batch_size,10,self.hidden_size).to(device)
        for t in range(10):
            if t==0:
                h1, c1 = self.LSTM1(mean_feat, (h1, c1))
                ah=torch.cat([h1,mean_feat],dim=1)
                att_feats = self.Att1(feats, ah)
                thought_vectors[:,t,:]=h1
            else:
                h1, c1 = self.LSTM1(att_feats, (h1, c1))
                ah=torch.cat([h1,att_feats],dim=1)
                att_feats = self.Att1(feats, ah)
                thought_vectors[:, t, :] = h1
        mean_thought=thought_vectors.mean(1)
        for t in range(3):
            if t==0:
                h2, c2 = self.LSTM2(mean_thought, (h2, c2))
                ah = torch.cat([h2, mean_thought], dim=1)
                att_thoughts = self.Att2(thought_vectors, ah)
            else:
                h2, c2 = self.LSTM2(att_thoughts, (h2, c2))
                ah = torch.cat([h2, att_thoughts], dim=1)
                att_thoughts = self.Att2(thought_vectors, ah)

        '''
        #oword1=self.classfier1(h1)
        #oword=self.classfier(h2)

        return h2, h1, alpha_mat
    
    def init_hidden_state(self,batch_size):
        h = torch.zeros(batch_size,self.hidden_size).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.hidden_size).to(device)
        return h, c

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data, a=self.minval, b=self.maxval)
                m.bias.data.fill_(0)




