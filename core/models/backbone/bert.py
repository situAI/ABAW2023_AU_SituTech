import torch
import torch.nn as nn
import numpy as np
from einops import repeat
import random

from core.utils import MODEL_REGISTRY

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)
        

class TransformerEncoder(nn.Module):
    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inc, 
            nhead=nheads, 
            dim_feedforward=feedforward_dim, 
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        out = self.transformer_encoder(x)

        return out


def Regressor(inc_dim, out_dim, dims_list=[512, 256], dropout=0.3, act=nn.GELU(), has_tanh=True):
        module_list = list()
        module_list.append(nn.Linear(inc_dim, dims_list[0]))
        module_list.append(act)
        if dropout != None:
            module_list.append(nn.Dropout(dropout))
        for i in range(len(dims_list) - 1):
            module_list.append(nn.Linear(dims_list[i], dims_list[i + 1]))
            module_list.append(act)
            if dropout != None:
                module_list.append(nn.Dropout(dropout))

        module_list.append(nn.Linear(dims_list[-1], out_dim))
        if has_tanh:
            module_list.append(nn.Tanh())
        module = nn.Sequential(*module_list)

        return module


@MODEL_REGISTRY.register()
class BERT(nn.Module):
    def __init__(self, 
                 input_dim, 
                 feedforward_dim, 
                 affine_dim, 
                 nheads, 
                 nlayers, 
                 dropout, 
                 head_dropout, 
                 head_dims, 
                 out_dim,
                 use_pe=False, 
                 seq_len=0):

        super().__init__()
        self.__dict__.update(locals())

        if self.affine_dim != None:
            self.affine = nn.Linear(input_dim, affine_dim)
            self.inc = affine_dim
        else:
            self.inc = input_dim

        if use_pe and seq_len:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, self.inc), freeze=True)
        else:
            self.use_pe = False
        self.transformer_encoder = TransformerEncoder(inc=self.inc, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers, dropout=dropout)

        self.head = Regressor(inc_dim=self.inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings

        out = self.transformer_encoder(x)
        # print('0', out.shape)
        out = self.head(out)
        # print('1', out.shape)

        # 0 torch.Size([128, 32, 1024])
        # 1 torch.Size([128, 32, 12])

        return out
