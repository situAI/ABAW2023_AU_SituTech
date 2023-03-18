import torch
import torch.nn as nn
from .tcn import TemporalConvNet
from .bert import get_sinusoid_encoding_table, TransformerEncoder, Regressor

from core.utils import MODEL_REGISTRY


class CrossFormerModule(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=nheads,
                                          dropout=dropout,
                                          bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def _sa_block(self, q, k, v):
        x = self.attn(query=q, key=k, value=v, need_weights=False)[0]

        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)

    def forward(self, src, q, k, v):
        x = src
        x = self.norm1(x + self._sa_block(q, k, v))
        x = self.norm2(x + self._ff_block(x))

        return x


class CrossTBERT(nn.Module):
    def __init__(self, inc1, inc2, feedforward_dim, num_channels, nheads=[4, 4], nlayers=[4, 4], kernel_size=[3, 3], dropout=0.3):
        
        super().__init__()
        self.__dict__.update(locals())

        self.f1_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc1, nheads=nheads[0], dim_feedforward=feedforward_dim[0], dropout=dropout) for _ in range(nlayers[0])
        ])

        self.f2_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc2, nheads=nheads[1], dim_feedforward=feedforward_dim[1], dropout=dropout) for _ in range(nlayers[1])
        ])

        self.f1_tcn = TemporalConvNet(inc1, num_channels[0], kernel_size=kernel_size[0], dropout=dropout)
        self.f2_tcn = TemporalConvNet(inc2, num_channels[1], kernel_size=kernel_size[1], dropout=dropout)

        self.f1_transformer = TransformerEncoder(inc1, nheads[0], feedforward_dim[0], nlayers[0], dropout)
        self.f2_transformer = TransformerEncoder(inc2, nheads[1], feedforward_dim[1], nlayers[1], dropout)

        self.f1_fuse_linear = nn.Linear(inc1, num_channels[0][-1])
        self.f2_fuse_linear = nn.Linear(inc2, num_channels[1][-1])

        out_dim = min(num_channels[0])

        self.f1_linear = nn.Linear(num_channels[0][-1]*2, out_dim)
        self.f2_linear = nn.Linear(num_channels[1][-1]*2, out_dim)

        
    def forward(self, f1, f2):

        in1 = f1.permute(1, 2, 0)
        in2 = f2.permute(1, 2, 0) 

        o1 = self.f1_tcn(in1).permute(2, 0, 1)
        o2 = self.f2_tcn(in2).permute(2, 0, 1)

        # for i in range(self.nlayers[0]):
        #     f1 = self.f1_branch[i](src=f1, q=f2, k=f1, v=f2)
        # for i in range(self.nlayers[1]):
        #     f2 = self.f2_branch[i](src=f2, q=f1, k=f2, v=f1)
        
        o3 = self.f1_transformer(f1)
        o3 = self.f1_fuse_linear(o3)
        
        o4 = self.f2_transformer(f2)
        o4 = self.f2_fuse_linear(o4)

        out0 = torch.cat([o1, o3], dim=2)
        out1 = torch.cat([o2, o4], dim=2)
        
        out0 = self.f1_linear(out0)
        out1 = self.f2_linear(out1)
        
        out = torch.cat([out0, out1], dim=2)

        return out


@MODEL_REGISTRY.register()
class Multi_TBERT(nn.Module):
    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 num_channels,
                 kernel_size,
                 nheads,
                 nlayers,
                 dropout,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim,
                 feat_split_list=[],
                 affine_dim_list=[], 
                 use_pe=True):

        super().__init__()
        self.__dict__.update(locals())

        if feat_split_list:
            assert input_dim == sum(feat_split_list), '输入维度和feat分离长度设置不匹配'
        
        self.split_idxs = [[sum(feat_split_list[:i]),sum(feat_split_list[:i+1])] if i > 0 else [0, l] for i, l in enumerate(feat_split_list)]

        if self.affine_dim_list != None:
            self.f1_affine = nn.Linear(feat_split_list[0], affine_dim_list[0], bias=False)
            self.f2_affine = nn.Linear(feat_split_list[1], affine_dim_list[1], bias=False)
            inc1 = affine_dim_list[0]
            inc2 = affine_dim_list[1]
        else:
            inc1, inc2 = feat_split_list[0], feat_split_list[1]
        
        if use_pe:
            self.pe1 = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc1), freeze=True)
            self.pe2 = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc2), freeze=True)

        self.crossformer = CrossTBERT(inc1=inc1, inc2=inc2, feedforward_dim=feedforward_dim, num_channels=num_channels, nheads=nheads, nlayers=nlayers, kernel_size=kernel_size, dropout=dropout)

        self.head = Regressor(inc_dim=num_channels[0][-1]*2, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = [x[:, :, ls[0]:ls[1]] for ls in self.split_idxs]
        f1_feature = multi_feature[0]
        f2_feature = multi_feature[1]

        if self.affine_dim_list:
            f1_feature = self.f1_affine(f1_feature)
            f2_feature = self.f2_affine(f2_feature)

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings1 = self.pe1(position_ids)
            position_embeddings2 = self.pe2(position_ids)
            f1_feature = f1_feature + position_embeddings1
            f2_feature = f2_feature + position_embeddings2
            

        out = self.crossformer(f1=f1_feature, f2=f2_feature)

        out = self.head(out)

        return out
