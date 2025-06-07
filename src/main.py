import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import skeletonkey as sk
import wandb

from pathlib import Path
from typing import Literal
import math
import os
import json

from runner import LanguageRunner, LevenshteinRunner, InsertionRunner, CorrectionRunner
from utils import TASK_TO_OBJECTIVE_FN, compute_token_accuracy_and_ppl, compute_bleu_and_rouge
from viz import generate
from logger import LoggerProtocol, get_logger

import sys

# Make python print unbuffered
STD_OUT = sys.stdout
sys.stdout = sys.stderr


@sk.unlock(str(Path(__file__).parent.parent / "configs" / "config.yaml"))
def main(cfg: sk.Config):

    logger: LoggerProtocol = get_logger(cfg)
    logger.log_params(cfg.to_dict())

    dataset: Dataset = sk.instantiate(cfg.dataset)
    task_type: Literal['classification', 'regression'] = dataset.task_type
    objective_fn: callable = TASK_TO_OBJECTIVE_FN[task_type]

    # Instantiate model: pass mask_token_id for LevenshteinTransformer
    model_kwargs = {'vocab_size': dataset.tokenizer.vocab_size}
    if 'LevenshteinTransformer' in cfg.model._target_:
        model_kwargs['mask_token_id'] = dataset.tokenizer.mask_token_id
    model: nn.Module = sk.instantiate(
        cfg.model,
        **model_kwargs
    )

    # make the dataset small if we are debugging
    if sum(cfg.split_percentages) != 1.0:
        def softmax(x: list[float]) -> list[float]:
            # Shift by max for numerical stability
            m = max(x)
            exps = [math.exp(i - m) for i in x]
            total = sum(exps)
            return [e / total for e in exps]

        logger.warning(
            "Split percentages do not sum to one, softmaxing them, which may lead to unintended results...")
        cfg.split_percentages = softmax(cfg.split_percentages)

    if cfg.debug:
        logger.warning("In debug mode so, the datasets sizes are smaller")
        cfg.split_percentages = [x * 0.01 for x in cfg.split_percentages]

    train_ds, dev_ds, test_ds = dataset.split(ps=cfg.split_percentages)
    train_dl: DataLoader = DataLoader(
        train_ds, shuffle=True, batch_size=cfg.bs
    )
    dev_dl: DataLoader = DataLoader(
        dev_ds, shuffle=False, batch_size=cfg.bs
    )

    optimizer = sk.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    device = torch.device("cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")

    if 'LevenshteinTransformer' in cfg.model._target_:
        train_runner = LevenshteinRunner(
            model=model,
            objective_fn=None,
            dataloader=train_dl,
            optimizer=optimizer,
            device=device,
        )
        dev_runner = LevenshteinRunner(
            model=model,
            objective_fn=None,
            dataloader=dev_dl,
            optimizer=None,
            device=device,
        )
    elif 'InsertionTransformer' in cfg.model._target_:
        mask_id = dataset.tokenizer.mask_token_id
        train_runner = InsertionRunner(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            mask_token_id=mask_id,
            device=device,
        )
        dev_runner = InsertionRunner(
            model=model,
            dataloader=dev_dl,
            optimizer=None,
            mask_token_id=mask_id,
            device=device,
        )
    elif 'CorrectionTransformer' in cfg.model._target_:
        mask_id = dataset.tokenizer.mask_token_id
        corr_prob = cfg.corruption_prob
        train_runner = CorrectionRunner(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            seq_len=dataset.seq_len,
            corruption_prob=corr_prob,
            mask_token_id=mask_id,
            device=device,
        )
        dev_runner = CorrectionRunner(
            model=model,
            dataloader=dev_dl,
            optimizer=None,
            seq_len=dataset.seq_len,
            corruption_prob=corr_prob,
            mask_token_id=mask_id,
            device=device,
        )
    else:
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
    bad_epochs = 0

    for epoch in range(1, cfg.epochs):
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80

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
        scheduler.step(mean_dev_loss)

        # dev_acc, dev_ppl = compute_token_accuracy_and_ppl(
        #     model, dev_dl, device)
        # dev_bleu, dev_rouge = compute_bleu_and_rouge(
        #     model, dev_dl, dataset.tokenizer, device,
        #     num_samples=cfg.eval_samples,
        #     max_len=20,
        #     temperature=cfg.sample_temperature
        # )

        metrics = {
            "mean_train_loss": mean_train_loss,
            "mean_dev_loss": mean_dev_loss,

            # "dev_token_accuracy": dev_acc,
            # "dev_perplexity": dev_ppl,
            # "dev_bleu": dev_bleu,
            # "dev_rougeL": dev_rouge,
        }
        logger.log_metrics(metrics)

        if mean_dev_loss < best_mean_dev_loss:
            logger.info(
                f"New best dev loss!!! ({best_mean_dev_loss} -> {mean_dev_loss})")
            best_mean_dev_loss = mean_dev_loss
            # Save weights on new best
            best_path = Path(logger.log_dir) / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            logger.log_artifact(str(best_path), save_name="best_model.pth")
            bad_epochs = 0
        else:
            bad_epochs += 1

        model = model.to('cpu')

        sample = generate(
            model,
            dataset.tokenizer,
            device,
            prompt="Down in the depths of",
            max_length=100,
            temperature=cfg.sample_temperature,
        )
        print(f"Sample generation: {sample}")
        logger.log_artifact(
            f"[EPOCH {epoch}]:\n{sample}\n",
            save_name=f"samples.txt"
        )

        print(f"{f'':-^{terminal_width}}\n")

        if bad_epochs >= cfg.patience:
            logger.warning(
                f"Bad epochs exceeded your patience of {cfg.patience}, exiting.")
            break
        model = model.to(device)

    logger.log_artifact(json.dumps(
        {
            'train_losses': train_losses,
            'dev_losses': dev_losses,
        },
    ),
        "dev_losses.json",
    )
    logger.clean_up()


if __name__ == "__main__":
    main()
