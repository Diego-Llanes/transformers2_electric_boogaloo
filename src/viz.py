import torch
from torchtyping import TensorType
from typing import List
from pathlib import Path

from models.levenshtein_transformer import LevenshteinTransformer
from models.insertion_transformer import InsertionTransformer
from models.correction_transformer import CorrectionTransformer


def generate_causal(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device | str,
    prompt: str = "",
    max_length: int = 50,
    temperature: float = 1.0,
) -> List[int]:
    model.to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]
            token_logits = logits[:, -1] / temperature
            probs = torch.softmax(token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
            generated = torch.cat([generated, next_id], dim=1)
    return generated[0].tolist()


def generate_edit(
    model: LevenshteinTransformer,
    tokenizer,
    device: torch.device | str,
    prompt: str = "",
    max_iters: int = 50,
) -> List[int]:
    model.to(device).eval()
    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        for _ in range(max_iters):
            logits = model(seq)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            if torch.equal(preds, seq):
                break
            seq = preds
    return seq[0].tolist()


def generate_insertion(
    model: InsertionTransformer,
    tokenizer,
    device: torch.device | str,
    max_iters: int = 50,
    ins_threshold: float = 0.5,
) -> List[int]:
    model.to(device).eval()
    mask_id = model.mask_token_id if model.mask_token_id is not None else tokenizer.mask_token_id
    seq = torch.full((1, 1), mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_iters):
            content_logits, ins_logits = model(seq)
            content_preds = content_logits.argmax(dim=-1)
            ins_probs = torch.softmax(ins_logits, dim=-1)[..., 1]
            insert_mask = ins_probs > ins_threshold

            new_seq = []
            for i in range(seq.size(1)):
                new_seq.append(content_preds[0, i].item())
                if insert_mask[0, i].item():
                    new_seq.append(mask_id)

            new_seq_tensor = torch.tensor(
                [new_seq], dtype=torch.long, device=device)
            if new_seq_tensor.size(1) == seq.size(1):
                seq = new_seq_tensor
                break
            seq = new_seq_tensor

    return seq[0].tolist()


def generate_correction(
    model: CorrectionTransformer,
    tokenizer,
    device: torch.device | str,
    corr_threshold: float = 0.5,
    prompt: str = '',
) -> List[int]:
    model.to(device).eval()
    input_ids = tokenizer(
        prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        cls_logits, rep_logits = model(input_ids)
        rep_preds = rep_logits.argmax(dim=-1)
    return rep_preds[0].tolist()


def generate(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device | str,
    prompt: str = "",
    max_length: int = 50,
    temperature: float = 1.0,
) -> str:
    cls_name = model.__class__.__name__

    if cls_name == "LevenshteinTransformer":
        ids = generate_edit(model, tokenizer, device, prompt=prompt)
    elif cls_name == "InsertionTransformer":
        ids = generate_insertion(
            model, tokenizer, device, max_iters=10, ins_threshold=0.5)
    elif cls_name == "CorrectionTransformer":
        ids = generate_correction(model, tokenizer, device, prompt)
    else:
        ids = generate_causal(model, tokenizer, device,
                              prompt, max_length, temperature)

    return tokenizer.decode(ids, skip_special_tokens=True)
