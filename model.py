import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class GraphConv(nn.Module):
    '''
    Simple GCN layer for node embedding, similar to https://arxiv.org/abs/1609.02907.
    Modified from https://github.com/jianhuupenn/SpaGCN/blob/master/SpaGCN_package/SpaGCN/layers.py.
    '''
    def __init__(self, in_dim, out_dim, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if exists(self.bias):
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, inputs, adj):
        x = torch.mm(inputs, self.weight) # input: [N, n_Bins] -> [N, D]
        x = torch.spmm(adj, x) # adj: [N, N], x: [N, D]
        if exists(self.bias):
            return x + self.bias
        else:
            return x
        
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.conv_1 = GraphConv(in_dim, hidden_dim)
        self.conv_2 = GraphConv(hidden_dim, out_dim)
        self.mid = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, inputs, adj):
        x = self.conv_1(inputs, adj)
        x = self.mid(x)
        x = self.conv_2(x, adj)
        return x
    
class MultiHeadAttention(nn.Module):
    '''Attention with graph bias.'''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, bias=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if exists(bias):
            dots += bias # bias shape: [1, 1, N, N] or broadcastable

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn.mean(dim=1)
    
class BidirectionalCrossAttention(nn.Module):
    '''
    Bidirectional Cross Attention layer.
    Modified from https://github.com/lucidrains/bidirectional-cross-attention/blob/main/bidirectional_cross_attention/bidirectional_cross_attention.py
    '''
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        self.context_to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        return_attn = False
    ):
        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h = self.heads), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('h i d, h j d -> h i j', qk, context_qk) * self.scale

        # get attention along both sequence length and context length dimensions
        # shared similarity matrix

        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('h i j, h j d -> h i d', attn, context_v)
        context_out = einsum('h j i, h j d -> h i d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(t, 'h n d -> n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out
    
class Model(nn.Module):
    def __init__(self, 
                in_dim,
                hidden_dim, 
                out_dim, 
                X1_dim, 
                X2_dim, 
                dropout=0., 
                n_features_1=None, 
                n_features_2=None,
                gene2vec=False,
                unpaired=False,
                ):
        super().__init__()
        self.X1_dim = X1_dim
        self.X2_dim = X2_dim
        self.n_features_1 = n_features_1
        self.n_features_2 = n_features_2
        self.gene2vec = gene2vec
        self.unpaired = unpaired

        if not gene2vec and exists(n_features_1):
            self.emb1 = nn.Embedding(n_features_1, in_dim)

        if exists(n_features_2):
            self.emb2 = nn.Embedding(n_features_2, in_dim)

        self.gcn_1 = GCN(in_dim, hidden_dim, out_dim, dropout)
        self.gcn_2 = GCN(in_dim, hidden_dim, out_dim, dropout)

        self.cross_attention = BidirectionalCrossAttention(
            dim=out_dim,
            context_dim=out_dim,
            heads=8,
            dim_head=64,
            dropout=dropout
        )
        if self.unpaired:
            self.attention = MultiHeadAttention(
                out_dim, 
                heads=8, 
                dim_head=64, 
                dropout=dropout
            )
        else:
            self.alpha = nn.Parameter(torch.FloatTensor(out_dim, out_dim))

        self.decoder1 = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, X1_dim * 2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, X2_dim * 2)
        )

    def forward(self, 
                X1, 
                X2, 
                input_1, 
                adj_1, 
                input_2, 
                adj_2, 
                lap_pe1, 
                lap_pe2,
                attn_bias=None
                ):
        assert X1.shape[1] == self.X1_dim
        assert X2.shape[1] == self.X2_dim

        if not self.gene2vec and exists(self.n_features_1):
            input_1 = self.emb1(input_1)

        if exists(self.n_features_2):
            input_2 = self.emb2(input_2)
            
        feat1 = self.gcn_1(input_1, adj_1)
        feat2 = self.gcn_2(input_2, adj_2)

        feat1 = feat1 + lap_pe1
        feat2 = feat2 + lap_pe2

        # cross attention
        feat1, feat2 = self.cross_attention(feat1, feat2)

        emb1 = torch.mm(X1, feat1)
        emb2 = torch.mm(X2, feat2)

        if self.unpaired:
            emb = []
            emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
            emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
            x = torch.cat(emb, dim=1)

            emb_combined, alpha = self.attention(x, bias=attn_bias)
            emb_combined = emb_combined.reshape(emb_combined.shape[0], -1)
            alpha = alpha.mean(dim=-2)
        else:
            emb_combined = self.alpha * emb1 + (1 - self.alpha) * emb2
            alpha = self.alpha
        
        x_recon1 = self.decoder1(emb_combined)
        x_recon1 = x_recon1.split((self.X1_dim, self.X1_dim), dim=-1)

        x_recon2 = self.decoder2(emb_combined)
        x_recon2 = x_recon2.split((self.X2_dim, self.X2_dim), dim=-1)

        loss = nb_loss(X1, x_recon1[0], x_recon1[1]) + nb_loss(X2, x_recon2[0], x_recon2[1])
        return {
            "embeddings": emb_combined,
            "attn_weights": alpha,
            "loss": loss
            }
    
def bernoulli_loss(y_true, logits, eps=1e-8):
    y_pred = torch.sigmoid(logits)
    y_pred = torch.clamp(y_pred, min=eps, max=1-eps)
    loss = -torch.mean(torch.sum(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred), 
        dim=-1
    ))
    return loss

def nb_loss(y_true, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    t1 = torch.lgamma(y_true + theta) - torch.lgamma(y_true + 1) - torch.lgamma(theta)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu))
    t3 = y_true * (torch.log(mu) - torch.log(theta + mu))
    log_nb = t1 + t2 + t3

    return -torch.mean(torch.sum(log_nb, dim=-1))