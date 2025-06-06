import torch.nn.functional as F
import math
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torchtyping import TensorType


def _classification_loss(preds, y):
    B, T, V = preds.shape
    return F.cross_entropy(preds.view(B*T, V), y.view(B*T))


def _regression_loss(preds, y):
    return F.mse_loss(preds, y)


TASK_TO_OBJECTIVE_FN = {
    'classification': _classification_loss,
    'regression': _regression_loss,
}


smooth_fn = SmoothingFunction().method1


def compute_token_accuracy_and_ppl(model, dataloader, device):
    """
    Runs through the entire dataloader to compute token‐level accuracy
    and dev perplexity = exp(mean_dev_loss).
    Returns (accuracy: float, perplexity: float).
    """
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    cum_loss = 0.0
    batches = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)  # (B, T)
            y_batch = y_batch.to(device)  # (B, T)
            out = model(x_batch)
            # if tuple, pick the first (B, T, V)
            if isinstance(out, tuple):
                for elem in out:
                    if elem.ndim == 3 and elem.size(-1) > 2:
                        logits = elem
                        break
            else:
                logits = out # (B, T, V)

            # token‐accuracy
            preds = logits.argmax(dim=-1)  # (B, T)
            correct_tokens += (preds == y_batch).sum().item()
            total_tokens += y_batch.numel()

            # cross‐entropy loss to accumulate for PPL
            B, T, V = logits.size()
            loss = criterion(logits.view(B*T, V), y_batch.view(B*T))
            cum_loss += float(loss.item())
            batches += 1

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    mean_loss = cum_loss / batches if batches > 0 else float('inf')
    perplexity = math.exp(mean_loss)
    return accuracy, perplexity


def compute_bleu_and_rouge(model, dataloader, tokenizer, device, num_samples=5, max_len=50, temperature=1.0):
    """
    Samples up to `num_samples` examples from dataloader, generates text from each prompt,
    and computes sentence BLEU + ROUGE-L against the true next token(s).
    Returns (avg_bleu: float, avg_rouge_l: float).
    """
    model.eval()
    bleu_scores = []
    rouge_l_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    count = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)   # (B, T)
            y_batch = y_batch.to(device)   # (B, T)
            B, T = x_batch.size()

            # We’ll just take the first example in this batch
            prompt_ids = x_batch[0].unsqueeze(0)   # (1, T)
            reference_ids = torch.cat([
                x_batch[0],
                y_batch[0][-1].unsqueeze(0)
            ], dim=0)  # (T+1,)

            # decode reference text
            ref_tokens = tokenizer.decode(
                reference_ids.tolist(), skip_special_tokens=True
            ).split()

            # generate hypothesis
            # NOTE: we reuse the same generate() from viz.py
            from viz import generate
            prompt_text = tokenizer.decode(
                prompt_ids[0].tolist(), skip_special_tokens=True
            )
            hyp_text = generate(
                model, tokenizer, device,
                prompt=prompt_text,
                max_length=max_len,
                temperature=temperature
            )
            hyp_tokens = hyp_text.split()

            # BLEU
            bleu = sentence_bleu([ref_tokens], hyp_tokens,
                                 smoothing_function=smooth_fn)
            bleu_scores.append(bleu)

            # ROUGE-L
            rouge = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))[
                'rougeL'].fmeasure
            rouge_l_scores.append(rouge)

            count += 1
            if count >= num_samples:
                break

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_l_scores) / \
        len(rouge_l_scores) if rouge_l_scores else 0.0
    return avg_bleu, avg_rouge
