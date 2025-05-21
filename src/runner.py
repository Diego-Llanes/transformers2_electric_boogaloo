import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchtyping import TensorType
from typing import Callable
import torch.nn as nn

from tqdm import tqdm


class LanguageRunner:
    def __init__(
        self,
        model: nn.Module,
        objective_fn: Callable[[TensorType, TensorType], TensorType[1]],
        dataloader: DataLoader,
        optimizer: Optimizer | None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.objective_fn = objective_fn
        self.loader = dataloader
        self.device = torch.device(device)
        self.optimizer = optimizer

    def run_epoch(self) -> float:
        # Train if optimizer is provided
        is_train = self.optimizer is not None
        self.model.train() if is_train else self.model.eval()


        desc = "loss: _"
        with tqdm(total=len(self.loader), desc=desc) as pbar:
            cum_loss, batches = 0.0, 0
            losses = []
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()
                preds = self.model(x)
                loss = self.objective_fn(preds, y)
                if is_train:
                    loss.backward()
                    self.optimizer.step()

                cum_loss += loss.item()
                losses.append(loss)
                batches += 1

                desc = f"loss: {cum_loss / batches:0.4f}"
                pbar.desc = desc
                pbar.update()

        return cum_loss / batches, losses

    def __repr__(self) -> str:
        return f"Runner: (\n\t{',\n\t'.join(" % s: % s" % item for item in vars(self).items())}\n)"
