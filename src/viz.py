import torch
from torchtyping import TensorType


def generate(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device | str,
    prompt: str = "",
    max_length: int = 50,
    temperature: float = 1.0,
) -> str:
    """
    Generates up to max_length tokens, but will not exceed model.pos_emb.num_embeddings.
    Returns the decoded string.
    """
    model.to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids  # shape: (1, T0)

    # Determine the absolute maximum sequence length from pos_emb
    # (this assumes every Transformer‐based model you use has a .pos_emb)
    seq_len_cap = None
    if hasattr(model, "pos_emb"):
        seq_len_cap = model.pos_emb.num_embeddings

    with torch.no_grad():
        for _ in range(max_length):
            T_current = generated.size(1)
            # If the model has a fixed positional‐embedding size, don't exceed it
            if seq_len_cap is not None and T_current >= seq_len_cap:
                break

            out = model(generated)
            # If model returns a tuple (e.g. (logits, other_logits)), pick the tensor of shape (B, T, V)
            if isinstance(out, tuple):
                for elem in out:
                    if elem.ndim == 3 and elem.size(-1) > 2:
                        logits = elem
                        break
            else:
                logits = out  # assume shape (B, T, V)

            token_logits = logits[:, -1] / temperature  # (B, V)
            probs = torch.softmax(token_logits, dim=-1)  # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1) or (B,)

            # Ensure shape is (B, 1)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)

            generated = torch.cat([generated, next_id], dim=1)  # (B, T+1)

    # Decode only the sequence we have
    result = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return result
