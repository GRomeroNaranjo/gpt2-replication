import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.head_size * config.n_heads, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.head_size
        self.q = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.k = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.v = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        return self.l2(self.gelu(self.l1(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lnf1 = nn.LayerNorm(config.n_embed)
        self.attention = MultiHeadAttention(config)
        self.lnf2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.lnf1(x))
        x = x + self.mlp(self.lnf2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.proj = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape

        tok = self.embed(x)
        pos_idx = torch.arange(T, device=x.device, dtype=torch.long)
        pos = self.pos_embed(pos_idx)

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        logits = self.proj(x)

        if y is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, logits.size(-1)),
                y.reshape(-1)
            )
            return logits, loss

        return logits, None