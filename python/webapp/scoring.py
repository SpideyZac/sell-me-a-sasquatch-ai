"""Ranks currently-legal action indices by a model's preference, for
surfacing a "recommended" move without depending on policy internals."""

from __future__ import annotations

import numpy as np


def rank_actions(model, obs: dict, mask: np.ndarray, samples: int = 40) -> list[tuple[int, float]]:
    """Ranks currently-legal action indices by how often the model (sampled
    stochastically) picks them, with its single deterministic best heavily
    weighted in so it always surfaces at rank 0. Uses only the public
    `.predict()` API (no reliance on policy-internals like raw logits)."""
    legal = np.flatnonzero(mask)
    if model is None:
        return [(int(i), 1.0 / len(legal)) for i in legal] if legal.size else []
    counts: dict[int, int] = {int(i): 0 for i in legal}  # every legal option ranked, even ones never sampled
    for _ in range(samples):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        counts[int(action)] = counts.get(int(action), 0) + 1
    best_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    counts[int(best_action)] = counts.get(int(best_action), 0) + samples
    total = sum(counts.values()) or 1
    return sorted(((idx, cnt / total) for idx, cnt in counts.items()), key=lambda kv: -kv[1])
