import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class SimpleTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: TensorType['batch', 'time', 'embedding']) -> TensorType['batch', 'time', 'embedding']:
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        x = x + self.out_proj(out)
        x = x + self.ff(self.ln2(x))
        return x


class CorrectionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        cls_n_layers: int = 2,
        cls_n_heads: int = 2,
        cls_emb_dim: int = 64,
        cls_ff_dim: int = 128,
        rep_n_layers: int = 4,
        rep_n_heads: int = 8,
        rep_emb_dim: int = 256,
        rep_ff_dim: int = 512,
        dropout: float = 0.1,
        mask_token_id: int = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

        # --- Classifier sub-model ---
        self.cls_token_emb = nn.Embedding(vocab_size, cls_emb_dim)
        self.cls_pos_emb = nn.Embedding(seq_len, cls_emb_dim)
        self.cls_blocks = nn.ModuleList([
            SimpleTransformerBlock(
                cls_emb_dim, cls_n_heads, cls_ff_dim, dropout)
            for _ in range(cls_n_layers)
        ])
        self.cls_ln = nn.LayerNorm(cls_emb_dim)
        self.cls_head = nn.Linear(cls_emb_dim, 1)  # per-token logit

        # --- Replacer sub-model ---
        # +1 in pos_emb for the appended mask token at T+1
        self.rep_token_emb = nn.Embedding(vocab_size, rep_emb_dim)
        self.rep_pos_emb = nn.Embedding(seq_len + 1, rep_emb_dim)
        self.rep_blocks = nn.ModuleList([
            SimpleTransformerBlock(
                rep_emb_dim, rep_n_heads, rep_ff_dim, dropout)
            for _ in range(rep_n_layers)
        ])
        self.rep_ln = nn.LayerNorm(rep_emb_dim)
        self.rep_head = nn.Linear(rep_emb_dim, vocab_size)

    def forward(self, tokens: TensorType['batch', 'time']) -> tuple[TensorType['batch', 'time'], TensorType['batch', 'time', 'vocab']]:
        """
        tokens: (B, T) – possibly corrupted sequence
        Returns:
          cls_logits: (B, T) with per-token logit (sigmoid on this for replace vs keep)
          rep_logits: (B, T+1, V) – vocab logits for each position + appended mask
        """
        B, T = tokens.size()

        # --- Classifier forward ---
        # (B, T, cls_emb)
        cls_x = self.cls_token_emb(tokens)
        pos = self.cls_pos_emb(torch.arange(
            T, device=tokens.device))  # (T, cls_emb)
        cls_x = cls_x + pos.unsqueeze(0)
        for block in self.cls_blocks:
            cls_x = block(cls_x)
        # (B, T, cls_emb)
        cls_x = self.cls_ln(cls_x)
        cls_logits = self.cls_head(cls_x).squeeze(-1)  # (B, T)

        # --- Replacer forward ---
        # Append one mask token at end (for the "append" prediction)
        rep_input = torch.cat([
            tokens,
            torch.full((B, 1), self.mask_token_id,
                       dtype=torch.long, device=tokens.device)
        ], dim=1)  # (B, T+1)
        # (B, T+1, rep_emb)
        rep_x = self.rep_token_emb(rep_input)
        rep_pos = self.rep_pos_emb(torch.arange(
            T+1, device=tokens.device))  # (T+1, rep_emb)
        rep_x = rep_x + rep_pos.unsqueeze(0)
        for block in self.rep_blocks:
            rep_x = block(rep_x)
        # (B, T+1, rep_emb)
        rep_x = self.rep_ln(rep_x)
        # (B, T+1, V)
        rep_logits = self.rep_head(rep_x)

        return cls_logits, rep_logits
