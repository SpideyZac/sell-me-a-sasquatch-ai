"""Smoke tests for the single-agent self-play wrapper used for training."""

import os

import numpy as np
import pytest

from sell_me_a_sasquatch.selfplay_env import OpponentPool, SasquatchSelfPlayEnv, random_masked_policy

DECK_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml"))


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_random_vs_random_episode_runs_to_completion(num_players):
    env = SasquatchSelfPlayEnv(num_players=num_players, deck_config_path=DECK_PATH, opponent_policy=random_masked_policy)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)

    rng = np.random.default_rng(1)  # seeded, not global np.random - keeps this test reproducible
    terminated = False
    reward = 0.0
    steps = 0
    while not terminated and steps < 5000:
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        assert legal.size > 0, "action mask must never be empty for the learner's own turn"
        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert not truncated
        steps += 1

    assert terminated
    assert "winner" in info
    assert 0 <= info["winner"] < num_players
    # Total reward carries the terminal +-1 (§3.4) plus an accumulated
    # potential-based shaping term (see `default_reward_fn` in env.py), so
    # it won't be exactly +-1 - just check it's a sane finite number.
    assert np.isfinite(reward)


def test_learner_seat_can_be_fixed():
    env = SasquatchSelfPlayEnv(num_players=4, deck_config_path=DECK_PATH, opponent_policy=random_masked_policy, learner_seat=2)
    env.reset(seed=5)
    assert env._learner_agent == "player_2"


def test_action_mask_matches_observation_action_mask():
    env = SasquatchSelfPlayEnv(num_players=4, deck_config_path=DECK_PATH, opponent_policy=random_masked_policy)
    obs, _ = env.reset(seed=2)
    assert np.array_equal(env.action_masks(), obs["action_mask"])


def test_masked_out_action_index_is_handled_defensively():
    env = SasquatchSelfPlayEnv(num_players=4, deck_config_path=DECK_PATH, opponent_policy=random_masked_policy)
    obs, _ = env.reset(seed=3)
    # Deliberately pass an action index guaranteed to be illegal.
    from sell_me_a_sasquatch import spaces as sasquatch_spaces

    illegal_action = sasquatch_spaces.MAX_ACTIONS - 1
    assert obs["action_mask"][illegal_action] == 0
    obs2, reward, terminated, truncated, info = env.step(illegal_action)
    assert env.observation_space.contains(obs2)


class _StubModel:
    """Duck-typed stand-in for an SB3 model's `.predict()` - avoids pulling
    the (optional, `[train]`-extra-only) sb3-contrib dependency into the
    base test suite just to test pool bookkeeping."""

    def __init__(self, tag):
        self.tag = tag

    def predict(self, obs, action_masks, deterministic=False):
        legal = np.flatnonzero(action_masks)
        return int(legal[0]) if legal.size else 0, None


def test_opponent_pool_falls_back_to_random_with_no_model_or_snapshots():
    pool = OpponentPool()
    pool.new_episode()
    mask = np.zeros(8, dtype=np.int8)
    mask[3] = 1
    action = pool(obs={}, mask=mask)
    assert action == 3  # random_masked_policy over a single legal action is deterministic


def test_opponent_pool_add_snapshot_caps_at_max_snapshots():
    pool = OpponentPool(max_snapshots=3)
    for i in range(5):
        pool.add_snapshot(_StubModel(i))
    assert len(pool.snapshots) == 3
    assert [m.tag for m in pool.snapshots] == [2, 3, 4]  # oldest evicted first


def test_opponent_pool_always_uses_live_model_when_no_snapshots_exist_yet():
    pool = OpponentPool(current_prob=0.0)  # would always prefer a snapshot if any existed
    pool.model = _StubModel("live")
    pool.new_episode()
    assert pool._active is pool.model


def test_opponent_pool_can_select_an_older_snapshot():
    pool = OpponentPool(current_prob=0.0, max_snapshots=5)  # never prefer the live model once snapshots exist
    pool.model = _StubModel("live")
    pool.add_snapshot(_StubModel("old"))
    pool.new_episode()
    assert pool._active is pool.snapshots[0]
