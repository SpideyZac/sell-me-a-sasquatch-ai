"""§5 Python-side smoke test: N random-policy episodes through the
PettingZoo env, asserting no exceptions, valid space conformance (via
PettingZoo's own `api_test`), and that `action_mask` is never empty for a
live (non-terminated) acting agent."""

import os
import random

import numpy as np
import pytest
from pettingzoo.test import api_test

from sell_me_a_sasquatch.env import env as make_env

DECK_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml"))


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_pettingzoo_api_conformance(num_players):
    e = make_env(num_players=num_players, deck_config_path=DECK_PATH)
    api_test(e, num_cycles=800, verbose_progress=False)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_policy_episode_runs_to_completion_without_deadlock(num_players, seed):
    e = make_env(num_players=num_players, deck_config_path=DECK_PATH)
    e.reset(seed=seed)
    rng = random.Random(seed)

    steps = 0
    for agent in e.agent_iter(max_iter=5000):
        obs, reward, termination, truncation, info = e.last()
        if termination or truncation:
            action = None
        else:
            mask = obs["action_mask"]
            legal_idx = np.flatnonzero(mask)
            assert legal_idx.size > 0, f"empty action_mask for a live agent at phase={obs['phase']}"
            action = int(rng.choice(legal_idx))
        e.step(action)
        steps += 1

    assert steps > 0
    assert e.agents == []
    e.close()
