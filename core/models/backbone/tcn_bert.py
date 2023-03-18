import torch
import torch.nn as nn
import numpy as np


from .tcn import TemporalConvNet,  Regressor
from .bert import BERT

from core.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class TCN_BERT(BERT):
    def __init__(self, num_channels, kernel_size, **args):

        super(TCN_BERT, self).__init__(**args)

        self.tcn = TemporalConvNet(self.inc, num_channels, kernel_size=kernel_size, dropout=self.dropout)
        self.head = Regressor(inc_dim=num_channels[-1]*2, out_dim=self.out_dim, dims_list=self.head_dims, dropout=self.head_dropout, act=nn.ReLU(), has_tanh=False)
        # self.t_head = nn.Linear(num_channels[-1], self.out_dim)

    def forward(self, x):
        seq_len, bs, _ = x.shape
        x = self.affine(x)
        # print(0, x.shape)
        in0 = x.permute(1, 2, 0)
        out0 = self.tcn(in0)
        out0 = out0.permute(2, 0, 1)
        # print(1, out0.shape)
        out1 = self.transformer_encoder(x)
        out = torch.cat([out0, out1], dim=2)
        # print(2, out0.shape)
        out = self.head(out)
        # print(3, out.shape)
        return out


@MODEL_REGISTRY.register()
class TCN_BERTV2(BERT):
    def __init__(self, num_channels, kernel_size, **args):

        super(TCN_BERTV2, self).__init__(**args)

        self.tcn = TemporalConvNet(self.inc, num_channels, kernel_size=kernel_size, dropout=self.dropout)
        self.head = Regressor(inc_dim=num_channels[-1], out_dim=self.out_dim, dims_list=self.head_dims, dropout=self.head_dropout, act=nn.ReLU(), has_tanh=False)
        # self.t_head = nn.Linear(num_channels[-1], self.out_dim)

    def forward(self, x):
        seq_len, bs, _ = x.shape
        x = self.affine(x)
        # print(0, x.shape)
        y = self.transformer_encoder(x)
        in0 = y.permute(1, 2, 0)
        out = self.tcn(in0).permute(2, 0, 1)
        out = self.head(out)
        # print(3, out.shape)
        return out

if __name__ == '__main__':

    tcn = TCN_BERT(1024, 1024, 1024, 4, 4, kernel_size=16, dropout=0.3, seq_len=128, head_dropout=0.1, head_dims=[512, 256], out_dim=12)
    input = torch.randn(128, 32, 1024)
    out = tcn(input)
    print(out.shape)