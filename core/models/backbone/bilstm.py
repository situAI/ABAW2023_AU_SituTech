
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
class BiLSTM(nn.Module):
    
    def __init__(self, 
                 input_dim,
                 out_dim,
                 seq_len,
                 dropout,
                 hidden_dim, 
                 nlayers=4,
                 head_dims=[512, 256],  
                 head_dropout=0.1, 
                 affine_dim=None,  
                 use_pe=False):

        super(BiLSTM, self).__init__()
        self.__dict__.update(locals())

        inc = input_dim
        self.affine_dim = affine_dim
        if self.affine_dim != None:
            self.affine = nn.Linear(input_dim, affine_dim)
            inc = affine_dim

                
        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc), freeze=True)
        
        # bilstm
        self.bilstm = nn.LSTM(inc, self.hidden_dim // 2, num_layers=self.nlayers, dropout=dropout, bidirectional=True, bias=False)
        
        # linear
        # self.head = nn.Linear(self.hidden_dim, out_dim)
        self.head = Regressor(self.hidden_dim, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    
    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings
        # print(0, x.shape)
        lstm_out, _ = self.bilstm(x)
        lstm_out = F.tanh(lstm_out)
        # print(1, lstm_out.shape)
        out = self.head(lstm_out)
        # print(2, out.shape)

        return out
    

if __name__ == '__main__':

    tcn = BiLSTM(1024, 12, 128, dropout=0.3, hidden_dim=1024)
    input = torch.randn(128, 32, 1024)
    seq_len, bs, _ = input.shape
    out = tcn(input)
    print(out.shape)