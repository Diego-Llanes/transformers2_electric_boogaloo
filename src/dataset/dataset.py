from __future__ import annotations
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pathlib import Path
from typing import Literal


class TinyLanguageDataset(Dataset):
    task_type: Literal['classification', 'regression'] = 'classification'

    def __init__(
        self,
        txt_path: str | Path,
        seq_len: int,
        tokenizer: PreTrainedTokenizerFast | None = None,
        stride: int = 1,
    ) -> None:
        self.seq_len, self.stride = seq_len, stride
        # use the fast tokenizer under the hood
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True
        )

        p = Path(txt_path)
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        texts = [f.read_text() for f in files]

        # batch‐encode all files at once, then flatten
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=999_999,
        )
        self.ids = [tok for seq in enc["input_ids"] for tok in seq]

        max_start = len(self.ids) - seq_len
        self.start_positions = list(range(0, max_start, stride))

    def __getitem__(self, idx) -> tuple[TensorType, TensorType]:
        start = self.start_positions[idx]
        chunk = self.ids[start: start + self.seq_len + 1]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:],  dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.start_positions)

    def split(self, ps: list[float]) -> list[TinyLanguageDataset]:
        total = len(self.start_positions)
        cuts = [0] + [int(sum(ps[:i])*total)
                      for i in range(1, len(ps))] + [total]
        subsets = []
        for i in range(len(ps)):
            sub = object.__new__(TinyLanguageDataset)
            sub.seq_len = self.seq_len
            sub.stride = self.stride
            sub.tokenizer = self.tokenizer
            sub.ids = self.ids
            sub.start_positions = self.start_positions[cuts[i]:cuts[i+1]]
            subsets.append(sub)
        return subsets


def get_ds_stats():
    from pathlib import Path
    import pandas as pd

    data_path = Path(__file__).parent.parent.parent / \
        "data" / "gutenberg_top100_cleaned"
    seq_len = 128

    ds = TinyLanguageDataset(data_path, seq_len=seq_len)
    train_ds, dev_ds, test_ds = ds.split([0.8, 0.1, 0.1])

    total = len(ds.ids)
    cut1 = int(0.8 * total)
    cut2 = int(0.9 * total)

    splits = {
        "Train": (ds.ids[:cut1], train_ds),
        "Dev":   (ds.ids[cut1:cut2], dev_ds),
        "Test":  (ds.ids[cut2:], test_ds),
    }

    rows = []
    for name, (tokens, subset) in splits.items():
        rows.append({
            "Split": name,
            "Num_Tokens": len(tokens),
            "Num_Sequences": len(subset),
            "Unique_Tokens": len(set(tokens)),
        })

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))

def get_miou():
    from pathlib import Path
    import pandas as pd

    data_path = Path(__file__).parent.parent.parent / "data" / "gutenberg_top100_cleaned"
    seq_len = 128

    ds = TinyLanguageDataset(data_path, seq_len=seq_len)
    train_ds, dev_ds, test_ds = ds.split([0.8, 0.1, 0.1])

    total = len(ds.ids)
    cut1 = int(0.8 * total)
    cut2 = int(0.9 * total)

    splits = {
        "Train": ds.ids[:cut1],
        "Dev":   ds.ids[cut1:cut2],
        "Test":  ds.ids[cut2:],
    }

    unique_tokens = {name: set(tokens) for name, tokens in splits.items()}

    pairs = [("Train", "Dev"), ("Train", "Test"), ("Dev", "Test")]
    rows = []
    for a, b in pairs:
        inter = len(unique_tokens[a] & unique_tokens[b])
        union = len(unique_tokens[a] | unique_tokens[b])
        rows.append({
            "Pair": f"{a}-{b}",
            "Intersection": inter,
            "Union": union,
            "IoU": inter / union,
        })

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))

    print(f"\nMean IoU: {df['IoU'].mean():.4f}")

if __name__ == "__main__":
    import sys

    # get_ds_stats()
    get_miou()
    sys.exit(0)

    import random

    frankenstien_path = Path(
        __file__
    ).parent.parent.parent / "data" / "frankenstien.txt"

    gutenberg_top_100_path = Path(
        __file__
    ).parent.parent.parent / "data" / "gutenberg_top100_cleaned"

    def test_dataset(path, seq_len=15):
        ds = TinyLanguageDataset(
            path,
            seq_len=seq_len,
        )
        tokenizer: BertTokenizer = ds.tokenizer
        print(f"getting first 10 lines from 'f{str(path)}'")
        for i in range(10):
            tokens = ds[i][0]
            print(f"{tokens=}\n{tokenizer.decode(tokens)}")

        print(f"getting random 10 lines from 'f{str(path)}'")
        for i in range(10):
            tokens = ds[random.randint(0, len(ds) - 1)][0]
            print(f"{tokens=}\n{tokenizer.decode(tokens)}")

    print('testing single file')
    test_dataset(frankenstien_path)

    print('testing many files')
    test_dataset(gutenberg_top_100_path)
