"""Ranks currently-legal action indices by a model's preference, for
surfacing a "recommended" move without depending on policy internals."""

from __future__ import annotations

import numpy as np  # type: ignore


def _exact_probabilities(model, obs: dict, mask: np.ndarray, legal: np.ndarray):
    """The masked action distribution itself, when the policy will hand it
    over. Exact where sampling is noisy, and one forward pass instead of
    dozens; falls back to `None` for anything that cannot supply it."""
    try:
        import torch as th  # type: ignore  # pylint: disable=import-outside-toplevel

        tensor_obs, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            distribution = model.policy.get_distribution(tensor_obs, action_masks=mask)
            probs = distribution.distribution.probs[0].cpu().numpy()
    except (
        Exception
    ):  # pylint: disable=broad-except  # noqa: BLE001 - any policy that cannot, falls back to sampling
        return None
    return sorted(((int(i), float(probs[i])) for i in legal), key=lambda kv: -kv[1])


def rank_actions(
    model, obs: dict, mask: np.ndarray, samples: int = 40
) -> list[tuple[int, float]]:
    """Ranks currently-legal action indices by how often the model (sampled
    stochastically) picks them, with its single deterministic best heavily
    weighted in so it always surfaces at rank 0. Uses only the public
    `.predict()` API (no reliance on policy-internals like raw logits)."""
    legal = np.flatnonzero(mask)
    if model is None:
        return [(int(i), 1.0 / len(legal)) for i in legal] if legal.size else []

    exact = _exact_probabilities(model, obs, mask, legal)
    if exact is not None:
        return exact

    counts: dict[int, int] = {
        int(i): 0 for i in legal
    }  # every legal option ranked, even ones never sampled
    for _ in range(samples):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        counts[int(action)] = counts.get(int(action), 0) + 1
    best_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    counts[int(best_action)] = counts.get(int(best_action), 0) + samples
    total = sum(counts.values()) or 1
    return sorted(
        ((idx, cnt / total) for idx, cnt in counts.items()), key=lambda kv: -kv[1]
    )
