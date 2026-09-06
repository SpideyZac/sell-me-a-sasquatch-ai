"""Single-agent self-play environment used for training.

Training libraries like Stable-Baselines3 expect a single-agent
`gymnasium.Env`. Every seat in Sell Me a Sasquatch is symmetric (same
observation and action shape, same rules), so the standard recipe is: pick
one seat as "the learner", and play every other seat with an
`opponent_policy` - by default a uniformly random legal move, but pointed at
the learner's own in-training policy this becomes genuine self-play with no
separate multi-agent training loop.

Two things differ from the obvious implementation:

* **It drives `native.Game` directly** rather than going through
  `SasquatchAECEnv`. The AEC protocol exists so *external* consumers can
  take one agent at a time; here we own the whole rollout, and its per-agent
  reward/termination bookkeeping is pure overhead on the hot path.

* **Table size is resampled every episode.** The observation is padded and
  ego-centric (see `spaces.py`), so 2- through 6-player games share one
  observation and action space. Training across all of them yields a single
  general model instead of one checkpoint per table size - and the variety
  acts as regularization, since a policy cannot overfit to one table's
  particular dynamics.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from . import _native as native
from . import spaces as sasquatch_spaces
from .env import DEFAULT_DECK_PATH, RewardFn, default_reward_fn, load_deck

# `(observation, action_mask, legal_count) -> action index`
#
# `legal_count` is redundant with `mask` - the ordinal action space makes
# every mask a prefix of ones - but it is passed explicitly because the
# environment already knows it, and recovering it by summing a few hundred
# bytes is a real cost at hundreds of thousands of micro-steps per second.
OpponentPolicy = Callable[[dict, np.ndarray, int], int]


def random_masked_policy(obs: dict, mask: np.ndarray, legal_count: int) -> int:
    """Uniform over legal actions - the baseline opponent."""
    return int(np.random.randint(legal_count)) if legal_count else 0


class OpponentPool:
    """Self-play opponent that mixes the live in-training model with a pool
    of its own older, frozen snapshots ("fictitious self-play" / a tiny
    league).

    Always playing an exact, always-current copy of itself - the simplest
    possible self-play - can cycle, or overfit to beating a mirror rather
    than learning something robust: the policy and its opponent are, after
    all, the same weights at every step. Mixing in older snapshots gives a
    more stable and more diverse curriculum, and is the standard fix.

    - `current_prob`: chance that a given episode's opponent is the live
      model rather than a random older snapshot (only meaningful once a
      snapshot exists; until then, always the live model).
    - The opponent identity is drawn once per episode (`new_episode`), not
      re-rolled every micro-turn, so "playing snapshot #3" means the whole
      game rather than a different opponent every action.
    """

    def __init__(self, current_prob: float = 0.5, max_snapshots: int = 10, rng: np.random.Generator | None = None):
        self.model = None  # set externally once the live model exists (chicken-and-egg at construction)
        self.snapshots: list = []
        self.current_prob = current_prob
        self.max_snapshots = max_snapshots
        self.rng = rng or np.random.default_rng()
        self._active = None

    def add_snapshot(self, model) -> None:
        """Freezes `model` into the league. Never trained further - only
        asked for moves. Called from a training callback; see
        `scripts/train.py`'s `SnapshotCallback`.

        Where possible the snapshot is converted to its numpy equivalent
        first. Opponent moves outnumber the learner's by (table size - 1)
        to one, and a torch forward pass on a single observation is mostly
        framework overhead, so this is one of the larger wins available on
        the rollout path."""
        try:
            from .policy import NumpyPointerPolicy

            model = NumpyPointerPolicy.from_model(model, rng=self.rng)
        except (ImportError, TypeError, AttributeError):
            pass  # not a pointer policy (or torch is absent) - use it as-is
        self.snapshots.append(model)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def new_episode(self) -> None:
        if self.snapshots and (self.model is None or self.rng.random() >= self.current_prob):
            self._active = self.snapshots[self.rng.integers(len(self.snapshots))]
        else:
            self._active = self.model

    def __call__(self, obs: dict, mask: np.ndarray, legal_count: int) -> int:
        model = self._active if self._active is not None else self.model
        if model is None:
            return random_masked_policy(obs, mask, legal_count)
        if not hasattr(model, "predict"):
            return int(model(obs, mask, legal_count))
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)


class SasquatchSelfPlayEnv(gym.Env):
    """One "hero" seat plays against `opponent_policy` at every other seat.

    `players` is the pool of table sizes to sample from each episode; pass a
    single value (e.g. `players=4`) to pin the table size.

    The hero's seat is redrawn each episode too. The ego-centric encoding
    already makes seats interchangeable to the policy, but a seat's
    *position relative to the turn leader* is a real difference - the seat
    that offers first sees a different game from the one that offers last.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        players: "int | Sequence[int]" = sasquatch_spaces.ALL_PLAYER_COUNTS,
        deck_config_path: "str | native.Deck" = DEFAULT_DECK_PATH,
        opponent_policy: Optional[OpponentPolicy] = None,
        learner_seat: "int | str" = "random",
        reward_fn: Optional[RewardFn] = None,
        max_actions: int | None = None,
    ):
        super().__init__()
        self.players = (players,) if isinstance(players, int) else tuple(players)
        if not self.players:
            raise ValueError("players must name at least one table size")
        for n in self.players:
            if not sasquatch_spaces.MIN_PLAYERS <= n <= sasquatch_spaces.MAX_PLAYERS:
                raise ValueError(f"table size must be 2..=6, got {n}")

        self.deck = load_deck(deck_config_path)
        self.opponent_policy = opponent_policy or random_masked_policy
        self.learner_seat_mode = learner_seat
        self.reward_fn = reward_fn or default_reward_fn
        self.max_actions = max_actions or sasquatch_spaces.max_legal_actions(self.deck, self.players)

        self.observation_space = sasquatch_spaces.observation_space(self.max_actions)
        self.action_space = sasquatch_spaces.action_space(self.max_actions)

        self._game: native.Game | None = None
        self._num_players = self.players[0]
        self._learner: int = 0
        self._personas = np.zeros((sasquatch_spaces.MAX_PLAYERS, sasquatch_spaces.NOISE_LEN), dtype=np.float32)
        self._prev_tokens: Sequence[int] = [0] * self._num_players
        self._mask = np.zeros(self.max_actions, dtype=np.int8)
        # Mirrors `_mask`. The encoder already returns the legal count, so
        # nothing on the hot path should be summing 300 int8s to recover it.
        self._legal_count = 0
        self._carried_reward = 0.0
        # Scratch buffers reused across micro-steps. `observe` copies out of
        # them before handing anything to a caller, so nothing outlives a
        # step; this just keeps a ~40 KB allocation off the hot path.
        self._scratch = sasquatch_spaces.empty_observation(self.max_actions)

    # gymnasium API

    @property
    def num_players(self) -> int:
        """Table size drawn for the current episode."""
        return self._num_players

    @property
    def learner_seat(self) -> int:
        """Which seat the learner is playing this episode."""
        return self._learner

    @property
    def legal_count(self) -> int:
        """How many actions are legal right now - the width of the mask's
        leading run of ones."""
        return self._legal_count

    def action_masks(self) -> np.ndarray:
        """`sb3_contrib.common.wrappers.ActionMasker` convention."""
        return self._mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._num_players = int(self.np_random.choice(self.players))
        self._learner = (
            int(self.np_random.integers(0, self._num_players)) if self.learner_seat_mode == "random" else int(self.learner_seat_mode)
        )
        self._learner %= self._num_players
        game_seed = int(self.np_random.integers(0, 2**63 - 1))
        self._game = native.Game(self._num_players, self.deck, game_seed)
        self._personas[: self._num_players] = sasquatch_spaces.sample_personas(self.np_random, self._num_players)
        self._prev_tokens = [0] * self._num_players

        if hasattr(self.opponent_policy, "new_episode"):
            self.opponent_policy.new_episode()

        reward = self._play_opponent_turns()
        obs = self._observe(self._learner)
        # A reward earned before the learner's first action has nowhere to
        # go in the gymnasium API; it is only ever shaping, and only when an
        # opponent moved first, so fold it into the first step instead.
        self._carried_reward = reward
        return obs, {}

    def step(self, action: int):
        idx = int(action)
        if not 0 <= idx < self._legal_count:
            # Masked-out index: a policy bug rather than a rules violation.
            # §3.4 says mask, don't raise - fail soft onto a legal move.
            idx = 0
        done, winner = self._game.step_index(self._learner, idx)
        reward = self._carried_reward + self._reward(winner if done else None)
        self._carried_reward = 0.0

        if not done:
            reward += self._play_opponent_turns()
            done = self._game.is_game_over()
            winner = self._game.winner()

        obs = self._observe(self._learner)
        info = {"winner": winner, "num_players": self._num_players, "learner_seat": self._learner} if done else {}
        return obs, reward, done, False, info

    def render(self):
        return None

    def add_opponent_snapshot(self, path: str) -> None:
        """Loads a saved checkpoint into this env's opponent league.

        Reached through `VecEnv.env_method` so the training callback can
        refresh the league inside every worker process, which cannot share
        the live model object (see `scripts/train.py`)."""
        if not isinstance(self.opponent_policy, OpponentPool):
            return
        from sb3_contrib import MaskablePPO

        self.opponent_policy.add_snapshot(MaskablePPO.load(path, device="cpu"))

    # internals

    def _observe(self, player: int, copy: bool = True) -> dict:
        """Encodes `player`'s view and refreshes `self._mask` to match.

        `copy=False` hands back the scratch buffers themselves, valid only
        until the next encode - fine for an opponent policy, which consumes
        the observation immediately, but never for anything returned to the
        training loop, which keeps it in a rollout buffer.
        """
        state, actions = self._scratch["state"], self._scratch["actions"]
        legal_count = self._game.encode(player, state, actions)
        state[sasquatch_spaces.NOISE_OFFSET :] = self._personas[player]
        if legal_count != self._legal_count:
            sasquatch_spaces.action_mask(legal_count, self.max_actions, out=self._mask)
            self._legal_count = legal_count
        if copy:
            return {"state": state.copy(), "actions": actions.copy()}
        return {"state": state, "actions": actions}

    def _reward(self, winner: Optional[int]) -> float:
        tokens = self._game.point_tokens()
        reward = self.reward_fn(self._prev_tokens, tokens, self._learner, winner)
        self._prev_tokens = tokens
        return reward

    def _play_opponent_turns(self) -> float:
        """Runs every non-learner micro-turn until it is the learner's move
        again (or the game ends), accumulating the learner's reward across
        *each* of them.

        With 3+ players several opponent micro-turns - a whole Thingamabob
        window, say - can pass before control returns, and under a dense
        reward any one of them can move the learner's standing (an opponent
        stealing one of their Point Tokens mid-window, for instance).
        """
        game = self._game
        total = 0.0
        while True:
            active = game.active_player()
            if active is None or active == self._learner:
                break
            obs = self._observe(active, copy=False)
            action = int(self.opponent_policy(obs, self._mask, self._legal_count))
            game.step_index(active, action if 0 <= action < self._legal_count else 0)
            total += self._reward(game.winner() if game.is_game_over() else None)
        return total
