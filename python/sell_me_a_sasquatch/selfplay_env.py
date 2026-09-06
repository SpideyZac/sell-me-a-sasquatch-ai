"""Single-agent self-play wrapper around `SasquatchAECEnv`.

Training libraries like Stable-Baselines3 expect a single-agent
`gymnasium.Env`. Since every seat in Sell Me a Sasquatch is symmetric (same
action/observation shape, same rules), the standard way to train one via
self-play is: pick one seat as "the learner," and for every other seat's
turn, act using an `opponent_policy` callback - by default a uniformly
random legal move, but pointed at the learner's own (in-training) policy
this becomes genuine self-play with no separate multi-agent training loop
needed.

This bypasses PettingZoo's `agent_iter()`/`last()` protocol entirely (that
machinery exists so *external* consumers can drive one agent at a time; here
we own the whole rollout ourselves, so it's simpler to read
`SasquatchAECEnv.rewards`/`.terminations` directly and to check
`is_game_over()` on the underlying engine instead of driving the formal
`_was_dead_step` dead-agent flush).
"""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from .env import DEFAULT_DECK_PATH, RewardFn, SasquatchAECEnv, default_reward_fn
from . import spaces as sasquatch_spaces

OpponentPolicy = Callable[[dict, np.ndarray], int]


def random_masked_policy(obs: dict, mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    return int(np.random.choice(legal)) if legal.size else 0


class OpponentPool:
    """Self-play opponent that mixes the live in-training model with a pool
    of its own older, frozen snapshots ("fictitious self-play" / a tiny
    league).

    Always playing against an exact, always-current copy of itself (the
    simplest possible self-play) can cycle or over-fit to beating a mirror
    of itself rather than learning something robust - the policy and its
    "opponent" are, after all, the same weights at every single step. Mixing
    in older snapshots gives the learner a more stable, more diverse curriculum
    of opponents and is the standard fix for that failure mode.

 - `current_prob`: chance the opponent for a given episode is the live
      model rather than a random older snapshot (only used once a snapshot
      exists at all - until then, always the live model).
 - The opponent identity is picked once per episode (`new_episode`), not
      re-rolled every micro-turn, so "playing against snapshot #3" means the
      whole game, not a different opponent every action.
    """

    def __init__(self, current_prob: float = 0.5, max_snapshots: int = 10):
        self.model = None  # set externally once the live model exists (chicken-and-egg at construction time)
        self.snapshots: list = []
        self.current_prob = current_prob
        self.max_snapshots = max_snapshots
        self._active = None

    def add_snapshot(self, model) -> None:
        """`model` is an already-loaded, frozen model object (its own
        predict() only - never trained further). Call from a training
        callback; see `scripts/train.py`'s `SnapshotCallback`."""
        self.snapshots.append(model)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def new_episode(self) -> None:
        if self.snapshots and (self.model is None or np.random.random() >= self.current_prob):
            self._active = self.snapshots[np.random.randint(len(self.snapshots))]
        else:
            self._active = self.model

    def __call__(self, obs: dict, mask: np.ndarray) -> int:
        model = self._active if self._active is not None else self.model
        if model is None:
            return random_masked_policy(obs, mask)
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)


class SasquatchSelfPlayEnv(gym.Env):
    """A single "hero" seat plays against `opponent_policy` for every other
    seat. `learner_seat="random"` (default) reassigns the hero to a
    different seat each episode so the learned policy isn't seat-biased."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 4,
        deck_config_path: str = DEFAULT_DECK_PATH,
        opponent_policy: Optional[OpponentPolicy] = None,
        learner_seat: "int | str" = "random",
        reward_fn: Optional[RewardFn] = None,
    ):
        super().__init__()
        self.num_players = num_players
        self.opponent_policy = opponent_policy or random_masked_policy
        self.learner_seat_mode = learner_seat
        self.observation_space = sasquatch_spaces.observation_space(num_players)
        self.action_space = sasquatch_spaces.action_space()

        self._aec = SasquatchAECEnv(num_players=num_players, deck_config_path=deck_config_path, reward_fn=reward_fn or default_reward_fn)
        self._learner_agent: str | None = None
        self._current_mask: np.ndarray = np.zeros(sasquatch_spaces.MAX_ACTIONS, dtype=np.int8)

    def action_masks(self) -> np.ndarray:
        """`sb3_contrib.common.wrappers.ActionMasker` convention."""
        return self._current_mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._aec.reset(seed=seed)
        seat = self.np_random.integers(0, self.num_players) if self.learner_seat_mode == "random" else int(self.learner_seat_mode)
        self._learner_agent = self._aec.possible_agents[seat]

        if hasattr(self.opponent_policy, "new_episode"):
            self.opponent_policy.new_episode()

        self._play_opponent_turns()
        obs = self._observe_learner()
        return obs, {}

    def _observe_learner(self) -> dict:
        obs = self._aec.observe(self._learner_agent)
        self._current_mask = obs["action_mask"]
        return obs

    def _play_opponent_turns(self) -> float:
        """Advances the underlying AEC env through every non-learner turn
        until it's the learner's turn again or the game has ended, summing
        the learner's reward across *each* of those steps - with 3+ players
        several opponent micro-turns (e.g. a whole Thingamabob window) can
        happen in a row before control returns, and under a dense reward_fn
        any of them individually could affect the learner (e.g. an opponent
        stealing a point token from the learner mid-window)."""
        game = self._aec._game
        total = 0.0
        while not game.is_game_over() and self._aec.agent_selection != self._learner_agent:
            agent = self._aec.agent_selection
            obs = self._aec.observe(agent)
            action = self.opponent_policy(obs, obs["action_mask"])
            self._aec.step(int(action))
            total += float(self._aec.rewards.get(self._learner_agent, 0.0))
        return total

    def step(self, action: int):
        mask = self._current_mask
        if action < 0 or action >= len(mask) or mask[action] == 0:
            legal = np.flatnonzero(mask)
            action = int(legal[0]) if legal.size else 0

        self._aec.step(int(action))
        total_reward = float(self._aec.rewards.get(self._learner_agent, 0.0))
        game = self._aec._game

        if not game.is_game_over():
            total_reward += self._play_opponent_turns()

        terminated = game.is_game_over()
        obs = self._observe_learner()
        info = {"winner": game.winner()} if terminated else {}
        return obs, total_reward, terminated, False, info

    def render(self):
        return None
