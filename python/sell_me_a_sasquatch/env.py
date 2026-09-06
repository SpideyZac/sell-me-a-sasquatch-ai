"""PettingZoo AECEnv wrapper (§3.4, Option A) around the Rust engine.

Each engine "turn" (§2.3) is decomposed into many micro-steps - one Seller's
deal-offer submit, that Seller's reveal, the Buyer's peek, each Thingamabob -window pass/play, the Buyer's final commit, each Nasty-penalty resolution -so that at any tick exactly one agent is `self.agent_selection`, matching
`GameState::active_players()` from the Rust layer. Option B (a true
`ParallelEnv` for the simultaneous deal-offer phase) is a documented
alternative not implemented here - see PROMPT.md §3.4.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from . import _native as native
from . import spaces as sasquatch_spaces

DEFAULT_DECK_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml"))

RewardFn = Callable[["SasquatchAECEnv", int, object, Optional[int]], float]


def _lead_margin(tokens: list[int], player_id: int) -> int:
    """Own Point Tokens minus the *best* of everyone else's. This is the
    quantity that actually has to go positive to win (§2.6: strictly more
    tokens than every other player at the threshold) - unlike raw own-token
    count, which rewards hoarding tokens even while an opponent races ahead
    faster."""
    others = [t for i, t in enumerate(tokens) if i != player_id]
    return tokens[player_id] - (max(others) if others else 0)


_SHAPING_COEF = 0.3
_SHAPING_GAMMA = 0.99  # matches MaskablePPO's default discount factor


def default_reward_fn(env: "SasquatchAECEnv", player_id: int, step_result, winner: Optional[int]) -> float:
    """+1 to the winner / -1 to everyone else at game end (§3.4's sparse
    ground-truth objective) plus a potential-based dense shaping term
    tracking each step's change in *lead margin* (`_lead_margin` above).

    Potential-based shaping (Ng, Harada & Russell 1999): adding
    `gamma * phi(s') - phi(s)` to the reward for any potential function
    `phi` providably leaves the optimal policy unchanged, unlike an
    arbitrary dense bonus. That means this gives PPO a much denser signal
    across a ~29-step sparse-terminal episode without distorting what
    "best play" actually means - a real risk with naive shaping (e.g.
    rewarding raw token gains regardless of opponents' standing).
    """
    tokens_now = env._game.observation(0).point_tokens  # Point Tokens are public info regardless of observer
    phi_before = _lead_margin(env._last_point_tokens, player_id)
    phi_after = _lead_margin(tokens_now, player_id)
    shaped = _SHAPING_COEF * (_SHAPING_GAMMA * phi_after - phi_before)

    terminal = 0.0 if winner is None else (1.0 if player_id == winner else -1.0)
    return shaped + terminal


class SasquatchAECEnv(AECEnv):
    """`Sell Me a Sasquatch` as a PettingZoo `AECEnv`.

    3-6 players use Buyer mode (§2.3); exactly 2 players automatically use
    the materially different 2-player variant (§2.7) - this is decided by
    the Rust engine itself based on `num_players`, not by this wrapper.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "sell_me_a_sasquatch_v0"}

    def __init__(
        self,
        num_players: int = 4,
        deck_config_path: str = DEFAULT_DECK_PATH,
        render_mode: str | None = None,
        reward_fn: RewardFn | None = None,
    ):
        super().__init__()
        if not (2 <= num_players <= 6):
            raise ValueError(f"num_players must be 2..=6, got {num_players}")
        self.num_players = num_players
        self.deck_config_path = deck_config_path
        self.render_mode = render_mode
        self.reward_fn = reward_fn or default_reward_fn

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        self._game: native.Game | None = None
        self._last_point_tokens: list[int] = [0] * num_players
        self._last_legal_actions: dict[str, list] = {}

    # Gymnasium/PettingZoo space plumbing

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return sasquatch_spaces.observation_space(self.num_players)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return sasquatch_spaces.action_space()

    # lifecycle

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**63 - 1))
        self._game = native.Game(self.num_players, self.deck_config_path, int(seed))

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._last_point_tokens = [0] * self.num_players
        self._last_legal_actions = {}

        self._update_agent_selection()

    def _update_agent_selection(self):
        if not self.agents:
            self.agent_selection = None
            return
        if self._game.is_game_over():
            # Every remaining agent is now terminated simultaneously (the
            # game ends for the whole table at once, not one agent at a
            # time). Per PettingZoo's `_was_dead_step` convention, each one
            # still needs exactly one more agent_selection turn so its
            # terminal reward is delivered via `last()` before being pruned;
            # `_was_dead_step` itself cascades through the rest from here.
            self.agent_selection = self.agents[0]
            return
        active = self._game.active_players()
        self.agent_selection = self.possible_agents[active[0]]

    def step(self, action):
        if not self.agents:
            return
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_id = self.agent_name_mapping[agent]
        self._cumulative_rewards[agent] = 0.0

        legal = self._last_legal_actions.get(agent) or self._game.legal_actions(player_id)
        idx = int(action)
        if idx < 0 or idx >= len(legal):
            # An out-of-range (masked-out) index is a policy bug, not a game
            # rule violation - §3.4 says illegal actions default to masking,
            # not raising, so fail soft onto the first legal action.
            idx = 0
        chosen = legal[idx]

        result = self._game.step(player_id, chosen)
        self.infos[agent] = {"events": result.events()}

        winner = result.winner if result.done else None
        for a in self.agents:
            pid = self.agent_name_mapping[a]
            self.rewards[a] = self.reward_fn(self, pid, result, winner)
        if not result.done:
            self._last_point_tokens = list(self._game.observation(0).point_tokens)
        else:
            for a in self.agents:
                self.terminations[a] = True

        self._update_agent_selection()
        self._accumulate_rewards()
        self._last_legal_actions = {}
        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        player_id = self.agent_name_mapping[agent]
        obs = self._game.observation(player_id)
        legal = self._game.legal_actions(player_id)
        self._last_legal_actions[agent] = legal
        return sasquatch_spaces.vectorize_observation(self._game, obs, len(legal), self.num_players)

    def render(self):
        from . import render as render_module

        if self.render_mode is None:
            return None
        text = render_module.render_state(self._game)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        self._game = None


def env(num_players: int = 4, deck_config_path: str = DEFAULT_DECK_PATH, render_mode: str | None = None, reward_fn: RewardFn | None = None):
    """Standard PettingZoo entry point: wraps the raw env with the usual
    order-enforcing / out-of-bounds wrappers."""
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = SasquatchAECEnv(num_players=num_players, deck_config_path=deck_config_path, render_mode=internal_render_mode, reward_fn=reward_fn)
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e
