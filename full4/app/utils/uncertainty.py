from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F


def softmax_entropy(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # prob: [N, C, H, W] softmax probabilities
    logp = torch.log(prob.clamp(min=eps))
    ent = -torch.sum(prob * logp, dim=1)  # [N, H, W]
    return ent


def compute_uncertainty(model, x: torch.Tensor, method: Literal['mc_dropout', 'softmax_entropy'], mc_iters: int = 8):
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    x = x.to(device)
    if method == 'softmax_entropy':
        with torch.no_grad():
            logits = model.forward_logits(x)
            prob = F.softmax(logits, dim=1)
            ent = softmax_entropy(prob)[0]
            return ent.cpu().numpy()
    elif method == 'mc_dropout':
        model.train()  # enable dropout
        probs = []
        with torch.no_grad():
            for _ in range(max(1, mc_iters)):
                logits = model.forward_logits(x)
                prob = F.softmax(logits, dim=1)
                probs.append(prob[0].cpu().numpy())
        probs = np.stack(probs, axis=0)  # [T, C, H, W]
        mean_prob = probs.mean(axis=0)
        # predictive entropy
        mean_prob_t = torch.from_numpy(mean_prob)
        pe = softmax_entropy(mean_prob_t.unsqueeze(0))[0].numpy()
        return pe
    else:
        return None

