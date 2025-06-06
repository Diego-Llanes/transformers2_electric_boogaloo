import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class FullSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: TensorType['batch', 'time', 'embedding']) -> TensorType['batch', 'time', 'embedding']:
        B, T, C = x.size()
        # project to Q, K, V
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each is (B, T, heads, head_dim)

        # reshape for multi-head
        q = q.permute(0, 2, 1, 3)  # (B, heads, T, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # compute attention scores (no causal mask)
        att = torch.matmul(q, k.transpose(-2, -1)) * \
            self.scale  # (B, heads, T, T)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # apply attention to values
        out = torch.matmul(att, v)  # (B, heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)


class LevTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = FullSelfAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: TensorType['batch', 'time', 'embedding']) -> TensorType['batch', 'time', 'embedding']:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LevenshteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_layers: int = 6,
        n_heads: int = 8,
        embedding_dim: int = 512,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        mask_token_id: int = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(seq_len, embedding_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            LevTransformerBlock(embedding_dim, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, tokens: TensorType['batch', 'time']) -> TensorType['batch', 'time', 'vocab']:
        B, T = tokens.size()
        tok = self.token_emb(tokens)
        pos = self.pos_emb(torch.arange(T, device=tokens.device))
        x = self.drop(tok + pos.unsqueeze(0))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
