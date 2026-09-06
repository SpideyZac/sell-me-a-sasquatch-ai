"""Loading/caching trained models (or the "random" policy), shared by the
Watch/Play game sessions and the Live tracker's advisor scoring."""

from __future__ import annotations

import glob
import os

from sell_me_a_sasquatch import spaces as sasquatch_spaces
from sell_me_a_sasquatch.selfplay_env import random_masked_policy

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_MODEL_CACHE: dict[str, object] = {}


def _resolve_model_path(spec: str) -> str:
    candidate = os.path.join(MODELS_DIR, spec)
    return candidate if os.path.exists(candidate) else spec


def get_model(spec: str):
    """Returns a loaded `MaskablePPO`, or `None` for the "random" policy."""
    spec = (spec or "random").strip()
    if spec.lower() == "random":
        return None
    if spec not in _MODEL_CACHE:
        from sb3_contrib import MaskablePPO  # lazy: only needed for real models

        model = MaskablePPO.load(_resolve_model_path(spec), device="cpu")
        _check_observation_format(spec, model)
        _MODEL_CACHE[spec] = model
    return _MODEL_CACHE[spec]


def _check_observation_format(spec: str, model) -> None:
    """Rejects checkpoints trained against a different observation layout.

    Without this the mismatch only surfaces mid-game, as a tensor-shape
    error from deep inside the policy - which reads like a crash rather than
    what it is: an old checkpoint that needs retraining."""
    space = getattr(model, "observation_space", None)
    state = getattr(space, "spaces", {}).get("state") if space is not None else None
    if state is None or state.shape != (sasquatch_spaces.OBS_LEN,):
        raise ValueError(
            f"'{spec}' was trained against a different observation format "
            f"(this build expects a Dict space with a {sasquatch_spaces.OBS_LEN}-float 'state'). "
            "Retrain it with scripts/train.py."
        )


def observation_width(model) -> int | None:
    """Action-space width the checkpoint was trained with, or `None` for
    the random policy.

    A checkpoint trained across every table size has a wider head than any
    single table needs, so its observations must be built at *its* width,
    not the current game's."""
    return None if model is None else int(model.action_space.n)


def get_policy(spec: str):
    """`(obs, mask, legal_count) -> action_index` callable, matching every
    other policy in this codebase (see `selfplay_env.py`).

    The returned callable carries a `max_actions` attribute so callers know
    how wide to encode observations for it."""
    model = get_model(spec)
    if model is None:
        return random_masked_policy

    def _policy(obs, mask, legal_count):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    _policy.max_actions = observation_width(model)
    return _policy


def list_available_models() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    paths = glob.glob(os.path.join(MODELS_DIR, "**", "*.zip"), recursive=True)
    return sorted(os.path.relpath(p, MODELS_DIR).replace(os.sep, "/") for p in paths)
