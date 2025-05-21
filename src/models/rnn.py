import torch
import torch.nn as nn
from torchtyping import TensorType


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.rnn = nn.RNN(
            input_size=n_embd,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        idx: TensorType["batch", "time"],
    ) -> TensorType["batch", "time", "vocabulary"]:
        x = self.token_emb(idx)  # (B, T, n_embd)
        out, _ = self.rnn(x)      # (B, T, hidden_size)
        logits = self.head(out)   # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    B, T, V, E, H = 2, 5, 100, 16, 32
    model = RNN(vocab_size=V, n_embd=E, hidden_size=H)
    idx = torch.randint(0, V, (B, T))
    logits = model(idx)
    print(f"RNN output shape: {logits.shape}")  # expect (2,5,100)
