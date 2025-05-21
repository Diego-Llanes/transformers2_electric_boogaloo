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
            logits: TensorType[1, "time", "vocab"] = model(generated)
            token_logits = logits[:, -1] / temperature
            probs = torch.softmax(token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
