"""Smoke tests for the single-agent self-play wrapper used for training."""

import os

import numpy as np  # type: ignore
import pytest  # type: ignore

from sell_me_a_sasquatch.selfplay_env import (
    OpponentPool,
    SasquatchSelfPlayEnv,
    random_masked_policy,
)

DECK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml")
)


def make_env(**kwargs):
    kwargs.setdefault("deck_config_path", DECK_PATH)
    kwargs.setdefault("opponent_policy", random_masked_policy)
    return SasquatchSelfPlayEnv(**kwargs)


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6])
def test_random_vs_random_episode_runs_to_completion(players):
    env = make_env(players=players)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)

    rng = np.random.default_rng(1)  # seeded, not global - keeps this test reproducible
    terminated = False
    reward = 0.0
    steps = 0
    while not terminated and steps < 5000:
        legal = np.flatnonzero(env.action_masks())
        assert legal.size > 0, "the learner's own turn must always have a legal action"
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        assert env.observation_space.contains(obs)
        assert not truncated
        steps += 1

    assert terminated
    assert 0 <= info["winner"] < players
    assert info["num_players"] == players
    # the total carries the terminal +-1 plus accumulated potential-based
    # shaping, so it will not be exactly +-1, just finite
    assert np.isfinite(reward)


def test_one_env_plays_every_table_size():
    """The single general model's premise: 2- through 6-player games share
    one observation and action space, so one env can serve all of them."""
    env = make_env(players=(2, 3, 4, 5, 6))
    seen = set()
    for seed in range(40):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        seen.add(env.num_players)
    assert len(seen) >= 4, f"expected a spread of table sizes, saw {sorted(seen)}"


def test_action_space_width_covers_every_table_size():
    env = make_env(players=(2, 3, 4, 5, 6))
    for n in (2, 3, 4, 5, 6):
        assert env.action_space.n >= env.deck.max_legal_actions(n)  # type: ignore


def test_learner_seat_can_be_fixed():
    env = make_env(players=4, learner_seat=2)
    env.reset(seed=5)
    assert env.learner_seat == 2


def test_masked_out_action_index_is_handled_defensively():
    env = make_env(players=4)
    env.reset(seed=3)
    illegal_action = env.action_space.n - 1  # type: ignore
    assert env.action_masks()[illegal_action] == 0
    obs, reward, terminated, truncated, info = env.step(illegal_action)
    assert env.observation_space.contains(obs)


def test_personas_differ_between_seats_within_an_episode():
    env = make_env(players=5)
    env.reset(seed=11)
    personas = env._personas[: env.num_players]
    assert len({tuple(p) for p in personas}) == env.num_players


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
    mask[0] = 1
    assert (
        pool(obs={}, mask=mask, legal_count=1) == 0
    )  # one legal action -> deterministic


def test_opponent_pool_add_snapshot_caps_at_max_snapshots():
    pool = OpponentPool(max_snapshots=3)
    for i in range(5):
        pool.add_snapshot(_StubModel(i))
    assert [m.tag for m in pool.snapshots] == [2, 3, 4]  # oldest evicted first


def test_opponent_pool_always_uses_live_model_when_no_snapshots_exist_yet():
    pool = OpponentPool(
        current_prob=0.0
    )  # would always prefer a snapshot if any existed
    pool.model = _StubModel("live")  # type: ignore
    pool.new_episode()
    assert pool._active is pool.model


def test_opponent_pool_can_select_an_older_snapshot():
    pool = OpponentPool(
        current_prob=0.0, max_snapshots=5
    )  # never prefer the live model once snapshots exist
    pool.model = _StubModel("live")  # type: ignore
    pool.add_snapshot(_StubModel("old"))
    pool.new_episode()
    assert pool._active is pool.snapshots[0]


def test_add_opponent_snapshot_is_a_no_op_without_a_pool():
    """`VecEnv.env_method` fans the call out to every worker, including ones
    training against the random baseline - it must not blow up there."""
    env = make_env(players=4)
    env.add_opponent_snapshot("does/not/exist.zip")
