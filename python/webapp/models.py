"""Loading/caching trained models (or the "random" policy), shared by the
Watch/Play game sessions and the Live tracker's advisor scoring."""

from __future__ import annotations

import glob
import os

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

        _MODEL_CACHE[spec] = MaskablePPO.load(_resolve_model_path(spec), device="cpu")
    return _MODEL_CACHE[spec]


def get_policy(spec: str):
    """`(obs, mask) -> action_index` callable, matching every other policy
    in this codebase (see `selfplay_env.py`)."""
    model = get_model(spec)
    if model is None:
        return random_masked_policy

    def _policy(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return _policy


def list_available_models() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    paths = glob.glob(os.path.join(MODELS_DIR, "**", "*.zip"), recursive=True)
    return sorted(os.path.relpath(p, MODELS_DIR).replace(os.sep, "/") for p in paths)
