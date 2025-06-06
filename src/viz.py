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
    model.to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            out = model(generated)

            # If model returns a tuple, pick the tensor of shape (B, T, V)
            if isinstance(out, tuple):
                for elem in out:
                    if elem.ndim == 3 and elem.size(-1) > 2:
                        logits = elem
                        break
            else:
                logits = out

            token_logits = logits[:, -1] / temperature  # (B, V)
            probs = torch.softmax(token_logits, dim=-1) # (B, V)
            next_id = torch.multinomial(
                probs, num_samples=1)  # (B, 1) or (B,)

            # ensure shape is (B, 1)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)

            generated = torch.cat([generated, next_id], dim=1)     # (B, T+1)

    result = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return result
