import torch
import torch.nn as nn
import numpy as np


from .tcn import TemporalConvNet,  Regressor, TCN


from core.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class GTCN(TCN):
    def __init__(self, t_num_channels,  t_kernel_size, **args):

        super(GTCN, self).__init__( **args)

        self.t_tcn = TemporalConvNet(self.seq_len, t_num_channels, kernel_size=t_kernel_size, dropout=self.dropout)
        self.t_head = nn.Linear(t_num_channels[-1], self.seq_len)
        self.head = Regressor(inc_dim=t_num_channels[-1]*2, out_dim=self.out_dim, dims_list=self.head_dims, dropout=self.head_dropout, act=nn.ReLU(), has_tanh=False)
        

    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        # print(0, x.shape)
        x0 = x.permute(1, 0, 2)
        y0 = self.t_tcn(x0)
        y0 = self.t_head(y0).permute(2, 0, 1)
        # y0 = y0.unsqueeze(1).repeat(1,seq_len,1).permute(1, 0, 2)
        # print(1, y0.shape)
        x1 = x.permute(1, 2, 0)
        y1 = self.tcn(x1)
        y1 = y1.permute(2, 0, 1)
        y = torch.cat([y0, y1], dim=2)
        out = self.head(y)
        return out

if __name__ == '__main__':

    tcn = GTCN(1024, 1024, 1024, 4, 4, kernel_size=16, dropout=0.3, seq_len=128, head_dropout=0.1, head_dims=[512, 256], out_dim=12)
    input = torch.randn(128, 32, 1024)
    out = tcn(input)
    print(out.shape)