import torch
from torch.utils.data import DataLoader, Dataset
import skeletonkey as sk
import torch.nn as nn

from pathlib import Path
from typing import Literal
import math
import os

from runner import LanguageRunner
from utils import TASK_TO_OBJECTIVE_FN
from viz import generate
from logger import LoggerProtocol, get_logger


@sk.unlock(str(Path(__file__).parent.parent / "configs" / "config.yaml"))
def main(cfg: sk.Config):

    logger: LoggerProtocol = get_logger(cfg)
    logger.log_params(cfg.to_dict())

    dataset: Dataset = sk.instantiate(cfg.dataset)
    task_type: Literal['classification', 'regression'] = dataset.task_type
    objective_fn: callable = TASK_TO_OBJECTIVE_FN[task_type]

    model: nn.Module = sk.instantiate(
        cfg.model,
        vocab_size=dataset.tokenizer.vocab_size
    )

    # make the dataset small if we are debugging
    if sum(cfg.split_percentages) != 1.0:

        def softmax(x: list[float]) -> list[float]:
            # Shift by max for numerical stability
            m = max(x)
            exps = [math.exp(i - m) for i in x]
            total = sum(exps)
            return [e / total for e in exps]

        logger.warn("Split percentages do not sum to one, softmaxing them, which may lead to unintended results...")
        cfg.split_percentages = softmax(cfg.split_percentages)

    if cfg.debug:
        logger.warn("In debug mode so, the datasets sizes are smaller")
        cfg.split_percentages = [x * 0.01 for x in cfg.split_percentages]

    train_ds, dev_ds, test_ds = dataset.split(ps=cfg.split_percentages)
    train_dl: DataLoader = DataLoader(
        train_ds, shuffle=True, batch_size=cfg.bs
    )
    dev_dl: DataLoader = DataLoader(
        dev_ds, shuffle=False, batch_size=cfg.bs
    )

    optimizer = sk.instantiate(cfg.optimizer, params=model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")
    train_runner = LanguageRunner(
        model=model,
        optimizer=optimizer,
        objective_fn=objective_fn,
        dataloader=train_dl,
        device=device,
    )
    dev_runner = LanguageRunner(
        model=model,
        optimizer=None,
        objective_fn=objective_fn,
        dataloader=dev_dl,
        device=device,
    )

    best_mean_dev_loss: float = float('inf')
    train_losses, dev_losses = [], []
    train_losses: list[float]
    dev_losses: list[float]

    for epoch in range(1, cfg.epochs):
        terminal_width = os.get_terminal_size().columns
        print(f"{f' Epoch {epoch} ':-^{terminal_width}}")

        mean_train_loss, _train_losses = train_runner.run_epoch()
        mean_train_loss: float
        _train_losses: list[float]
        train_losses.extend(_train_losses)
        logger.info(f"{mean_train_loss=}")

        mean_dev_loss, _dev_losses = dev_runner.run_epoch()
        mean_dev_loss: float
        _dev_losses: list[float]
        dev_losses.extend(_dev_losses)
        logger.info(f"{mean_dev_loss=}")


        logger.log_metrics({
            "mean_train_loss": mean_train_loss,
            "mean_dev_loss": mean_dev_loss,
        })

        if mean_dev_loss < best_mean_dev_loss:
            logger.info(f"New best dev loss!!! ({best_mean_dev_loss} -> {mean_dev_loss})")
            best_mean_dev_loss = mean_dev_loss

        sample = generate(
            model,
            dataset.tokenizer,
            device,
            prompt="Once upon a time,",
            max_length=100,
            temperature=cfg.sample_temperature,
        )
        print(f"Sample generation: {sample}")
        print(f"{f'':-^{terminal_width}}\n")


if __name__ == "__main__":
    main()
