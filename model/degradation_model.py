import torch
import torch.nn as nn
from model.embedding import CNNEmbeddingLayer
from torchinfo import summary


class Degradation_Transformer(nn.Module):
    def __init__(self, in_dim, d_model, ffn_hidden, n_head, n_layers, drop_prob, emb=True):
        super(Degradation_Transformer, self).__init__()
        self.emb = emb
        if emb:
            self.emb = CNNEmbeddingLayer(in_dim, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, 1)
        
    def forward(self, x):
        if self.emb:
            x = self.emb(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return x
    
    def get_att_map(self, x):
        att_list = []
        if self.emb:
            x = self.emb(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        for layer in self.layers:
            x, att = layer(x, att_map=True)
            att_list.append(att)
        return att_list


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, att_map=False):
        # 1. compute self attention
        _x = x
        x, att = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        if att_map:
            return x, att
        else:
            return x
    

class PositionwiseFeedForward(nn.Module):
    # applied along the last dimension
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class Capacity_CNN(nn.Module):
    # for capacity prediction on Stanford Dataset
    def __init__(self, in_dim, d_model):
        super(Capacity_CNN, self).__init__()
        self.emb = CNNEmbeddingLayer(in_dim, d_model)
        self.linear_out = nn.Sequential(nn.ReLU(),
                                        nn.Linear(d_model, 1))
    def forward(self, x):
        x = self.emb(x)
        x = x.reshape(x.shape[0], -1)
        out = self.linear_out(x)
        return out
    

class Monotone_Loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Monotone_Loss, self).__init__()
        self.tau = tau

    def forward(self, seq):
        exp_diff = torch.exp(-torch.diff(seq/self.tau, dim=1))
        dif = torch.mean(-torch.log(exp_diff/(exp_diff+1)), dim=1)
        return dif
    