import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torchtyping import TensorType
import torch.nn as nn
from tqdm import tqdm

import random
from typing import Callable


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

                # Detach the loss to avoid memory leaks
                loss = float(loss.item())
                cum_loss += loss
                losses.append(loss)
                batches += 1

                desc = f"loss: {cum_loss / batches:0.4f}"
                pbar.desc = desc
                pbar.update()

        return cum_loss / batches, losses

    def __repr__(self) -> str:
        return f"Runner: (\n\t{',\n\t'.join(" % s: % s" % item for item in vars(self).items())}\n)"


class LevenshteinRunner:
    def __init__(
        self,
        model: nn.Module,
        objective_fn: Callable,
        dataloader: DataLoader,
        optimizer: Optimizer | None,
        device: torch.device | str = 'cpu',
        mask_prob: float = 0.15,
    ) -> None:
        self.model = model.to(device)
        # not used directly for LevT, but kept for interface
        self.objective_fn = objective_fn
        self.loader = dataloader
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.mask_prob = mask_prob

        # obtain mask token id from model
        self.mask_token_id = getattr(model, 'mask_token_id', None)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(self) -> tuple[float, list[float]]:
        is_train = self.optimizer is not None
        self.model.train() if is_train else self.model.eval()

        desc = "loss: _"
        with tqdm(total=len(self.loader), desc=desc) as pbar:
            cum_loss, batches = 0.0, 0
            losses = []

            for x, _ in self.loader:
                # x holds the original token IDs (we ignore y)
                x = x.to(self.device)
                # 1) Create a copy to mask
                inputs = x.clone()
                # 2) Randomly choose tokens to mask
                mask = torch.rand_like(
                    inputs, dtype=torch.float) < self.mask_prob
                inputs[mask] = self.mask_token_id

                # 3) Prepare labels: masked positions = original ID, others = -100
                labels = x.clone()
                labels[~mask] = -100

                if is_train:
                    self.optimizer.zero_grad()

                logits = self.model(inputs)  # (B, T, V)
                B, T, V = logits.size()
                loss = self.criterion(logits.view(B*T, V), labels.view(B*T))

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                loss_val = float(loss.item())
                cum_loss += loss_val
                losses.append(loss_val)
                batches += 1

                desc = f"loss: {cum_loss / batches:0.4f}"
                pbar.desc = desc
                pbar.update()

        return cum_loss / batches, losses


class InsertionRunner:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer | None,
        mask_token_id: int,
        device: torch.device | str = 'cpu',
    ) -> None:
        self.model = model.to(device)
        self.loader = dataloader
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.mask_token_id = mask_token_id
        self.content_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.ins_criterion = nn.CrossEntropyLoss()

    def run_epoch(self) -> tuple[float, list[float]]:
        is_train = self.optimizer is not None
        self.model.train() if is_train else self.model.eval()

        desc = "loss: _"
        with tqdm(total=len(self.loader), desc=desc) as pbar:
            cum_loss, batches = 0.0, 0
            losses = []

            for x, _ in self.loader:
                x = x.to(self.device)  # Full target sequence, shape (B, T)
                B, T = x.size()

                # 1) Randomly reveal 50% of tokens
                reveal_mask = (torch.rand((B, T), device=self.device) < 0.5)
                inputs = x.clone()
                inputs[~reveal_mask] = self.mask_token_id  # Mask out 50%

                # 2) Content labels: only masked positions are valid, others = -100
                content_labels = x.clone()
                content_labels[reveal_mask] = -100

                # 3) Insertion labels: 0 = no-insert (revealed), 1 = insert (masked)
                ins_labels = (~reveal_mask).long()

                if is_train:
                    self.optimizer.zero_grad()

                content_logits, ins_logits = self.model(
                    inputs)  # (B, T, V), (B, T, 2)

                # Flatten for loss
                content_loss = self.content_criterion(
                    content_logits.view(B * T, -1),
                    content_labels.view(B * T)
                )
                ins_loss = self.ins_criterion(
                    ins_logits.view(B * T, 2),
                    ins_labels.view(B * T)
                )
                loss = content_loss + ins_loss

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                loss_val = float(loss.item())
                cum_loss += loss_val
                losses.append(loss_val)
                batches += 1

                desc = f"loss: {cum_loss / batches:0.4f}"
                pbar.desc = desc
                pbar.update()

        return cum_loss / batches, losses



class CorrectionRunner:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer | None,
        seq_len: int,
        corruption_prob: float,
        mask_token_id: int,
        device: torch.device | str = 'cpu',
    ) -> None:
        self.model = model.to(device)
        self.loader = dataloader
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.seq_len = seq_len
        self.corruption_prob = corruption_prob
        self.mask_token_id = mask_token_id
        self.bce_loss = BCEWithLogitsLoss()
        # Ignore unmasked tokens when computing replacement loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(self) -> tuple[float, list[float]]:
        is_train = self.optimizer is not None
        self.model.train() if is_train else self.model.eval()

        cum_loss, batches = 0.0, 0
        losses = []

        desc = "loss: _"
        with tqdm(total=len(self.loader), desc=desc) as pbar:
            for x, _ in self.loader:
                x = x.to(self.device)  # (B, T) original clean sequences
                B, T = x.size()

                # 1) Randomly corrupt some tokens
                corrupt_mask = (torch.rand((B, T), device=self.device) < self.corruption_prob)
                rand_tokens = torch.randint(0, self.model.vocab_size, (B, T), device=self.device)
                corrupted = x.clone()
                corrupted[corrupt_mask] = rand_tokens[corrupt_mask]

                # 2) Classifier target: 1.0 where corrupted, 0.0 otherwise
                cls_labels = corrupt_mask.float()  # (B, T)

                # 3) Prepare replacer input: mask out corrupted tokens
                rep_input = corrupted.clone()
                rep_input[corrupt_mask] = self.mask_token_id

                # 4) Replacer target: original x at corrupted positions, others = -100
                rep_labels = x.clone()
                rep_labels[~corrupt_mask] = -100
                # Append a dummy class for the appended token → ignore index
                rep_labels = torch.cat([rep_labels, torch.full((B,1), -100, device=self.device, dtype=torch.long)], dim=1)  # (B, T+1)

                if is_train:
                    self.optimizer.zero_grad()

                cls_logits, rep_logits = self.model(rep_input)  # cls: (B,T), rep: (B, T+1, V)

                # Classification loss
                loss_cls = self.bce_loss(cls_logits, cls_labels)

                # Replacement loss – flatten
                Bp, Tp1, V = rep_logits.size()
                loss_rep = self.ce_loss(rep_logits.view(Bp*Tp1, V), rep_labels.view(Bp*Tp1))

                loss = loss_cls + loss_rep
                if is_train:
                    loss.backward()
                    self.optimizer.step()

                loss_val = float(loss.item())
                cum_loss += loss_val
                losses.append(loss_val)
                batches += 1

                desc = f"loss: {cum_loss / batches:0.4f}"
                pbar.desc = desc
                pbar.update()

        return cum_loss / batches, losses
