"""Token-weighted language model evaluation."""
import math
import torch
from model.runtime import load_model, load_data, batch

@torch.inference_mode()
def evaluate(model_path, data, device="cpu"):
    torch.set_num_threads(2)
    model = load_model(model_path, device).eval()
    rows = load_data(data, model.config.vocab_size, model.config.max_position_embeddings)
    total, targets = 0., 0
    for row in rows:
        ids, labels, mask = batch([row], device)
        n = int((labels[:, 1:] != -100).sum())
        loss = model(ids, labels=labels, attention_mask=mask).loss.item()
        total += loss * n
        targets += n
    nll = total / targets
    return {"nll": nll, "perplexity": math.exp(nll), "targets": targets, "rows": len(rows)}
