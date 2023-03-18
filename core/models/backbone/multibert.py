import torch
import torch.nn as nn

from core.utils import MODEL_REGISTRY
from .bert import get_sinusoid_encoding_table, TransformerEncoder, Regressor

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

class CrossFormer(nn.Module):
    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        super().__init__()
        self.nlayers = nlayers
        self.visual_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.audio_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

    def forward(self, visual, audio):
        for i in range(self.nlayers):
            visual = self.visual_branch[i](src=visual, q=audio, k=visual, v=visual)
            audio = self.audio_branch[i](src=audio, q=visual, k=audio, v=audio)

        out = torch.cat([visual, audio], dim=-1)

        return out


class CrossFormerV2(nn.Module):
    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        super().__init__()
        self.nlayers = nlayers
        self.visual_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.audio_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.visual_transformer = TransformerEncoder(inc, nheads, feedforward_dim, nlayers, dropout)
        self.audio_transformer = TransformerEncoder(inc, nheads, feedforward_dim, nlayers, dropout)

    def forward(self, visual, audio):
        for i in range(self.nlayers):
            visual = self.visual_branch[i](src=visual, q=audio, k=visual, v=visual)
            audio = self.audio_branch[i](src=audio, q=visual, k=audio, v=audio)

        visual = self.visual_transformer(visual)
        audio = self.audio_transformer(audio)

        out = torch.cat([visual, audio], dim=-1)

        return out


@MODEL_REGISTRY.register()
class MultiBERT(nn.Module):
    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim):

        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim

        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc // 2), freeze=True)

        self.crossformer = CrossFormer(inc=inc // 2, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)
        self.transformer = TransformerEncoder(inc=inc, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)

        self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    
    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = x.split(x.shape[-1] // 2, -1)
        visual_feature = multi_feature[0]
        audio_feature = multi_feature[1]

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            visual_feature = visual_feature + position_embeddings
            audio_feature = audio_feature + position_embeddings

        out = self.crossformer(visual=visual_feature, audio=audio_feature)
        out = self.transformer(out)

    
        out = self.head(out)

        return out

@MODEL_REGISTRY.register()
class MultiBERTV2(nn.Module):
    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim):

        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        
        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc // 2), freeze=True)

        self.crossformer = CrossFormerV2(inc=inc // 2, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)

        self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = x.split(x.shape[-1] // 2, -1)
        visual_feature = multi_feature[0]
        audio_feature = multi_feature[1]

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            visual_feature = visual_feature + position_embeddings
            audio_feature = audio_feature + position_embeddings

        out = self.crossformer(visual=visual_feature, audio=audio_feature)

        out = self.head(out)

        return out


@MODEL_REGISTRY.register()
class MultiBERTV3(nn.Module):
    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim):

        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        
        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc // 2), freeze=True)
            
        self.visual_transformer = TransformerEncoder(inc//2, nheads, feedforward_dim, nlayers, dropout)

        self.audio_transformer = TransformerEncoder(inc//2, nheads, feedforward_dim, nlayers, dropout)

        self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)


    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = x.split(x.shape[-1] // 2, -1)
        visual_feature = multi_feature[0]
        audio_feature = multi_feature[1]

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            visual_feature = visual_feature + position_embeddings
            audio_feature = audio_feature + position_embeddings
        out0 = self.visual_transformer(visual_feature)
        # print(1, out0.shape)
        out1 = self.audio_transformer(audio_feature)
        # print(2, out1.shape)
        out = torch.cat([out0, out1], dim=2)
        # print(3, out.shape)
        out = self.head(out)

        return out