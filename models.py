import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_transformer import FMoETransformerMLP
from gates import CustomNaiveGate_Balance_SMoE, MHMoEGate

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

def _skew(X, pad_value):
    """shift every row one step to the right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value = pad_value) # B x M x (M + L + 1)
    X = X.view(B, -1) # B x (ML + MM + M)
    X = X[:, :-M] # B x (ML + MM)
    X = X.view(B, M, M + L) # B x M x (L + M)
    return X

def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x (M + L)
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x (ML + MM)
    X = F.pad(X, (0, M))  # B x (ML + MM + M)
    X = X.view(B, M, M + L + 1)  # B x M x (L + M + 1)
    X = X[:, :, :L]  # B x M x L
    return X

class SeqAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        attn_span,
        dropout
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_span = attn_span
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, pos_encoding):
        # query size = B x M x H
        # key, value sizes = B x (M + L) x H

        # B x M (q) x (M + L) (k)
        attn_ctx = torch.matmul(query, key.transpose(-1, -2))
        attn_ctx = _unskew(attn_ctx) # B x M x L

        attn_pos = torch.matmul(query, pos_encoding) # B x M x L_pos
        attn = attn_ctx + attn_pos

        attn = attn / math.sqrt(self.hidden_size) # B x M x L_pos
        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)

        attn_ctx = _skew(attn, 0) # B x M x (L + M)
        out = torch.matmul(attn_ctx, value) # B x M x H
        return out

    def get_cache_size(self):
        return self.attn_span

class MultiHeadSeqAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        attn_span,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn = SeqAttention(
            hidden_size = hidden_size,
            dropout = dropout,
            attn_span = attn_span,
        )
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias = False)

    def head_reshape(self, x):
        K = self.num_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D)) # B x (M + L) x K x D
        x = x.transpose(1, 2).contiguous() # B x K x (M + L) x D
        x = x.view(-1, x.size(-2), x.size(-1)) # B_K x (M + L) x D
        return x

    def forward(self, query, key, value, pos_encoding):
        B = query.size(0)
        K = self.num_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, pos_encoding) # B_K x M x D
        out = out.view(B, K, M, D) # B x K x M x D
        out = out.transpose(1, 2).contiguous() # B x M x K x D
        out = out.view(B, M, -1) # B x M x K_D
        out = self.proj_out(out)
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__()
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2

class MomentumLayer(FMoETransformerMLP):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma,
        mu,
        world_size,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(   
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            gate = gate,
            num_experts = num_experts,
            moe_top_k = moe_top_k,
            mhmoe_num_heads = mhmoe_num_heads,
            mhmoe_beta = mhmoe_beta,
            world_size = world_size,
        )
        self.gamma = gamma
        self.mu = mu
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)

        momentum = self.mu * momentum + self.gamma * moe_out
        # output = self.layer_norm(inp - momentum)
        output = inp - momentum
        return output, momentum

class AdamLayer:
    def __init__():
        NotImplementedError("Under development")

class TransformerSeqLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        num_heads,
        attn_span,
        dropout,
        gate_name,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        beta1,
        beta2,
        world_size,
        s,
        g,
        f,
        layerth,
    ):
        super().__init__()
        if gate_name == "smoe":
            gate = CustomNaiveGate_Balance_SMoE # from SwitchTransformer paper
        elif gate_name == "mhmoe":
            gate = MHMoEGate
        else:
            ValueError("Incorrect gate name")
        
        self.use_attn = s == "s"
        self.attn = (
            MultiHeadSeqAttention(
                hidden_size = hidden_size,
                num_heads = num_heads,
                dropout = dropout,
                attn_span = attn_span,
            )
            if self.use_attn
            else None
        )
        
        self.use_smoe = g == "m" or g == "a"
        self.smoe = (
            MomentumLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma = gamma2,
                mu = mu,
                world_size = world_size,
            )
            if g == "m"
            else
            AdamLayer(

            )
            if g == "a"
            else None
        )

        self.use_ff = f == "f"
        self.ff = (
            FeedForwardLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
            )
            if self.use_ff
            else None
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, h, h_cache, pos_encoding, momentum):
        # h = B x M x H
        # h_cache = B x L x H
        if self.use_attn:
            h_all = torch.cat([h_cache, h], dim = 1) # B x (M + L) x H
            attn_out = self.attn(h, h_all, h_all, pos_encoding)
            h = self.norm1(h + attn_out) # B x M x H
        if self.use_smoe:
            smoe_out, momentum = self.smoe(h, momentum)
            h = self.norm2(h + smoe_out) # B x M x H
        if self.use_ff:
            ff_out = self.ff(h)
            h = self.norm3(h + ff_out) # B x M x H
        return h, momentum

class TransformerSeq(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_hidden_size,
        num_heads,
        num_layers,
        attn_span,
        architecture,
        dropout,
        gate_name,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        beta1,
        beta2,
        world_size,
    ):
        super().__init__()
        self.inp_embed = nn.Embedding(vocab_size, hidden_size)
        self.out_embed = nn.Linear(hidden_size, vocab_size)

        self.pos_encoding = nn.Parameter(torch.randn(1, hidden_size // num_heads, attn_span))
        self.arch = architecture

        self.attn_layer_count = self.arch.count("s")
        self.layers = nn.ModuleList()

        self.layers.extend(
            TransformerSeqLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                num_heads = num_heads,
                attn_span = attn_span,
                dropout = dropout,
                gate_name = gate_name,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma1 = gamma1,
                gamma2 = gamma2,
                mu = mu,
                beta1 = beta1,
                beta2 = beta2,
                world_size = world_size,
                s = self.arch[2 * i],
                g = self.arch[2 * i + 1],
                f = None,
                layerth = i
            )
            for i in range(num_layers)
        )
    
    def forward(self, x, h_cache):
        block_size = x.size(1) # B x M
        h = self.inp_embed(x) # B x M x H
        h_cache_next = []
        if "a" in self.arch:
            momentum = (
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros_like(h),
                )
        else: # in case of no momentum --mu will be set to zero
            momentum = torch.zeros_like(h)
        
        for i, layer in enumerate(self.layers):
            if layer.use_attn:
                cache_size = layer.attn.attn.get_cache_size()
                if cache_size > block_size:
                    h_cache_next_l = torch.cat(
                        [h_cache[i][:, -cache_size + block_size:, :], h],
                        dim = 1
                    ).detach()
                else:
                    h_cache_next_l = h[:, -cache_size:, :].detach()
                h_cache_next.append(h_cache_next_l)
                h, momentum = layer(h, h_cache[i], self.pos_encoding, momentum) # B x M x H
            else:
                # TODO: is this branch even necesarry in our case?
                h = layer(h, [], self.pos_encoding)
        
        out = F.log_softmax(self.out_embed(h), dim = -1)
        return out, h_cache_next