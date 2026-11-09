"""Published GSPO-token surrogate with explicit per-token gradient routing."""
from __future__ import annotations


def gspo_token_loss(logps, old_logps, advantages, mask, *, epsilon_low=0.0003,
                    epsilon_high=0.0004, ref_logps=None, beta=0.0,
                    inference_correction=None, accumulation_steps=1):
    """Sequence-mean then batch-mean loss, including optional reference KL.

    The sequence ratio is detached. p_t / stopgrad(p_t) is numerically one but
    retains each token's own gradient. This distinction is essential when
    advantages differ within a response. Inference correction is a separate,
    detached weight supplied by the upstream vLLM generation path.
    """
    import torch

    if logps.ndim != 2 or any(x.shape != logps.shape for x in (old_logps, advantages, mask)):
        raise ValueError("logps, old_logps, advantages and mask must all have shape (B,T)")
    if accumulation_steps < 1 or epsilon_low < 0 or epsilon_high < 0 or beta < 0:
        raise ValueError("Invalid objective hyperparameters")
    active = mask.bool()
    for name, value in (("logps", logps), ("old_logps", old_logps), ("advantages", advantages)):
        if not torch.isfinite(value[active]).all():
            raise ValueError(f"Nonfinite active {name}")
    # Clear masked positions before arithmetic; NaN * 0 is still NaN.
    p = torch.where(active, logps.float(), 0.0)
    old = torch.where(active, old_logps.detach().float(), 0.0)
    adv = torch.where(active, advantages.detach().float(), 0.0)
    weights = active.to(p.dtype)
    lengths = weights.sum(-1, keepdim=True).clamp_min(1)
    sequence_ratio = torch.exp(((p - old) * weights).sum(-1, keepdim=True) / lengths)
    if not torch.isfinite(sequence_ratio).all():
        raise ValueError("Nonfinite GSPO sequence ratio")
    token_ratio = sequence_ratio.detach() * torch.exp(p - p.detach())
    clipped = token_ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    policy_terms = -torch.minimum(token_ratio * adv, clipped * adv)
    if inference_correction is not None:
        correction = inference_correction.detach().float()
        if correction.shape not in (p.shape, (p.shape[0], 1)):
            raise ValueError("Inference correction must have shape (B,T) or (B,1)")
        correction = torch.broadcast_to(correction, p.shape)
        if not torch.isfinite(correction[active]).all() or (correction[active] < 0).any():
            raise ValueError("Invalid inference correction")
        policy_terms = policy_terms * torch.where(active, correction, 0.0)
    kl = torch.zeros_like(p)
    if beta:
        if ref_logps is None or ref_logps.shape != p.shape or not torch.isfinite(ref_logps[active]).all():
            raise ValueError("Finite reference log probabilities required when beta > 0")
        delta = torch.where(active, ref_logps.detach().float(), 0.0) - p
        kl = torch.exp(delta) - delta - 1
        if not torch.isfinite(kl).all():
            raise ValueError("Nonfinite reference KL")
    per_token_loss = policy_terms + beta * kl
    loss = ((per_token_loss * weights).sum(-1) / lengths.squeeze(-1)).mean() / accumulation_steps
    low = (sequence_ratio < 1 - epsilon_low) & (adv < 0) & active
    high = (sequence_ratio > 1 + epsilon_high) & (adv > 0) & active
    return loss, {
        "sequence_ratio": sequence_ratio.detach(), "kl": kl.detach(),
        "low_clipped": low, "high_clipped": high,
    }
