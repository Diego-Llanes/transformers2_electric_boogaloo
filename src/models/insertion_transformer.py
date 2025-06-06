import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from .levenshtein_transformer import FullSelfAttention


class InsTransformerBlock(nn.Module):
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


class InsertionTransformer(nn.Module):
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
            InsTransformerBlock(embedding_dim, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embedding_dim)
        # Two heads: content prediction (vocab) + insertion decision (2-way)
        self.content_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.ins_head = nn.Linear(embedding_dim, 2, bias=False)

    def forward(self, tokens: TensorType['batch', 'time']) -> tuple[TensorType['batch', 'time', 'vocab'], TensorType['batch', 'time', 2]]:
        B, T = tokens.size()
        tok = self.token_emb(tokens)  # (B, T, E)
        pos = self.pos_emb(torch.arange(T, device=tokens.device))  # (T, E)
        x = self.drop(tok + pos.unsqueeze(0))  # (B, T, E)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)  # (B, T, E)
        content_logits = self.content_head(x)  # (B, T, V)
        ins_logits = self.ins_head(x)  # (B, T, 2)
        return content_logits, ins_logits
