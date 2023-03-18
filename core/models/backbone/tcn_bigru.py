import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
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

    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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
class TCN_BiGRU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_channels, 
                 kernel_size, 
                 hidden_dim, 
                 nheads, 
                 nlayers, 
                 dropout, 
                 seq_len, 
                 out_dim,
                 head_dims=[512],  
                 head_dropout=0.1, 
                 affine_dim=None,  
                 use_pe=False):

        super(TCN_BiGRU, self).__init__()
        self.__dict__.update(locals())


        if self.affine_dim != None:
            self.affine = nn.Linear(input_dim, affine_dim)
            inc = affine_dim
        else:
            inc = input_dim

        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc), freeze=True)
        

        self.tcn = TemporalConvNet(inc, num_channels, kernel_size=kernel_size, dropout=dropout)
       
        self.bigru = nn.GRU(inc, self.hidden_dim // 2, dropout=dropout, num_layers=nlayers, bidirectional=True)
        
        self.head = Regressor(inc_dim=self.hidden_dim + num_channels[-1], out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings
        
        x0 = x.permute(1, 2, 0)   # input should have dimension (N, C, L)
        y0 = self.tcn(x0)
        y0 = y0.permute(2, 0, 1)
        y1, _ = self.bigru(x)
        # print(1, y0.shape, y1.shape)
        y = torch.cat([y0, y1], dim=2)
        out = self.head(y)
        # print(2, out.shape)

        return out


@MODEL_REGISTRY.register()
class TCN_BiGRUV2(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_channels, 
                 kernel_size, 
                 hidden_dim, 
                 nheads, 
                 nlayers, 
                 dropout, 
                 seq_len, 
                 out_dim,
                 head_dims=[512],  
                 head_dropout=0.1, 
                 affine_dim=None,  
                 use_pe=False):

        super(TCN_BiGRUV2, self).__init__()
        self.__dict__.update(locals())


        if self.affine_dim != None:
            self.affine = nn.Linear(input_dim, affine_dim)
            inc = affine_dim
        else:
            inc = input_dim

        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc), freeze=True)
        

        self.tcn = TemporalConvNet(inc, num_channels, kernel_size=kernel_size, dropout=dropout)
       
        self.bigru = nn.GRU(num_channels[-1], self.hidden_dim // 2, dropout=dropout, num_layers=nlayers, bidirectional=True)
        
        self.head = Regressor(inc_dim=self.hidden_dim, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings
        
        x0 = x.permute(1, 2, 0)   # input should have dimension (N, C, L)
        y0 = self.tcn(x0)
        y0 = y0.permute(2, 0, 1)
        y, _ = self.bigru(y0)
        # print(1, y0.shape, y1.shape)
        out = self.head(y)
        # print(2, out.shape)

        return out
    
    
if __name__ == '__main__':

    tcn = TCN_BiGRU(1024, 1024, 4, 4, 0.3, 128, 12, hidden_dim=512)
    input = torch.randn(128, 32, 1024)
    seq_len, bs, _ = input.shape
    out = tcn(input)
    print(out.shape)