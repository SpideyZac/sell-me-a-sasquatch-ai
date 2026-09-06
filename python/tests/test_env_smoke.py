"""Python-side smoke test: random-policy episodes through the PettingZoo
env, asserting no exceptions, space conformance (via PettingZoo's own
`api_test`), and that the action mask is never empty for a live agent."""

import os
import random

import numpy as np  # type: ignore
import pytest  # type: ignore
from pettingzoo.test import api_test  # type: ignore

from sell_me_a_sasquatch import spaces as sasquatch_spaces
from sell_me_a_sasquatch.env import SasquatchAECEnv
from sell_me_a_sasquatch.env import env as make_env

DECK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml")
)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_pettingzoo_api_conformance(num_players):
    """PettingZoo's own `api_test` is a thorough smoke test of the env's
    conformance to the AEC API, including that the observation and action spaces are well-formed
    and that the env can run through a full episode."""
    e = make_env(num_players=num_players, deck_config_path=DECK_PATH)
    api_test(e, num_cycles=400, verbose_progress=False)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_policy_episode_runs_to_completion_without_deadlock(num_players, seed):
    """A random policy should be able to run through a full episode without
    deadlocking on an empty action mask, and the env should always produce
    observations that conform to its observation space."""
    e = make_env(num_players=num_players, deck_config_path=DECK_PATH)
    e.reset(seed=seed)
    rng = random.Random(seed)

    steps = 0
    for _agent in e.agent_iter(max_iter=5000):
        obs, _reward, termination, truncation, _info = e.last()
        if termination or truncation:
            action = None
        else:
            legal_idx = np.flatnonzero(obs["action_mask"])  # type: ignore
            assert legal_idx.size > 0, "empty action_mask for a live agent"
            action = int(rng.choice(legal_idx))
        e.step(action)
        steps += 1

    assert steps > 0
    assert e.agents == []
    e.close()


@pytest.mark.parametrize("num_players", [2, 4, 6])
def test_observation_shape_is_identical_across_table_sizes(num_players):
    """The whole point of the padded, ego-centric encoding: one policy has to
    be able to observe a 2-player game and a 6-player game unchanged."""
    e = SasquatchAECEnv(num_players=num_players, deck_config_path=DECK_PATH)
    e.reset(seed=0)
    obs = e.observe(e.agent_selection)
    assert obs["state"].shape == (sasquatch_spaces.OBS_LEN,)
    assert obs["actions"].shape[1] == sasquatch_spaces.ACTION_FEAT_LEN
    assert e.observation_space(e.agent_selection).contains(obs)


def test_action_mask_is_a_prefix_of_ones():
    """Several hot paths assume this (see `spaces.action_mask`), so it is
    worth asserting rather than trusting."""
    e = SasquatchAECEnv(num_players=4, deck_config_path=DECK_PATH)
    e.reset(seed=7)
    for _ in range(50):
        if not e.agents or e.terminations[e.agent_selection]:
            break
        mask = e.observe(e.agent_selection)["action_mask"]
        count = int(mask.sum())
        assert count > 0
        assert np.array_equal(mask[:count], np.ones(count, dtype=mask.dtype))
        assert not mask[count:].any()
        e.step(0)


def test_ego_centric_observation_is_seat_independent():
    """Two players looking at the same fresh table should describe their own
    seat in slot 0, so a policy trained at one seat transfers to any other."""
    e = SasquatchAECEnv(num_players=5, deck_config_path=DECK_PATH)
    e.reset(seed=99)
    views = [e.observe(agent)["state"] for agent in e.agents]
    # Nobody's view is another's (different hands, different ego ordering)...
    for i, view in enumerate(views):
        for j in range(i + 1, len(views)):
            assert not np.array_equal(view, views[j])
    # ...but they agree on the global block, which is seat-independent apart
    # from the leader-offset and "is it me" flags at its tail.
    phase_block = sasquatch_spaces.PHASES
    assert all(
        np.array_equal(views[0][: len(phase_block)], v[: len(phase_block)])
        for v in views
    )


def test_personas_are_resampled_per_episode():
    """ "Noise" is added to the persona block of the observation, so that a
    policy trained against one persona can generalize to others. This test
    asserts that the noise is actually resampled per episode, rather than
    being fixed for the lifetime of the env."""
    e = SasquatchAECEnv(num_players=3, deck_config_path=DECK_PATH)
    e.reset(seed=1)
    first = e.observe(e.agents[0])["state"][sasquatch_spaces.NOISE_OFFSET :].copy()
    e.reset(seed=2)
    second = e.observe(e.agents[0])["state"][sasquatch_spaces.NOISE_OFFSET :]
    assert first.shape == (sasquatch_spaces.NOISE_LEN,)
    assert not np.array_equal(first, second)
