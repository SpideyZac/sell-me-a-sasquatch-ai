"""PettingZoo AECEnv wrapper around the Rust engine.

Each engine "turn" is decomposed into many micro-steps: one seller's
deal-offer submit, that seller's reveal, the buyer's peek, each thingamabob
window pass or play, the buyer's final commit, each nasty-penalty
resolution, so that at any tick exactly one agent is `self.agent_selection`,
matching `GameState::active_player()` from the Rust layer. A true
`ParallelEnv` for the simultaneous deal-offer phase would also be possible,
but isn't implemented here.

This is the interface env: PettingZoo-conformant, one agent at a time,
suitable for external consumers, the web app, and evaluation. Training does
not go through it; `selfplay_env.py` drives the engine directly, since the
AEC protocol's per-agent bookkeeping is pure overhead when one process owns
the whole rollout anyway.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional, Sequence

import numpy as np  # type: ignore
from pettingzoo import AECEnv  # type: ignore
from pettingzoo.utils import wrappers  # type: ignore

from . import _native as native  # type: ignore
from . import spaces as sasquatch_spaces

DEFAULT_DECK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml")
)
"""Path to the confirmed real deck config, used when no other deck is given."""

_DECK_CACHE: dict[str, "native.Deck"] = {}
"""Parsed decks keyed by normalized path, so repeated loads reuse one instance."""


def load_deck(deck: "str | native.Deck" = DEFAULT_DECK_PATH) -> "native.Deck":
    """Parses `deck.toml` once per path and reuses it for every game.

    A fresh parse per `reset()` used to cost a file read plus a full TOML
    parse, which, once the engine itself got fast, was a real share of an
    episode's total cost. Passing an already-loaded `Deck` through unchanged
    keeps callers that manage their own deck honest.
    """
    if isinstance(deck, native.Deck):  # pylint: disable=c-extension-no-member
        return deck
    path = os.path.normpath(deck)
    cached = _DECK_CACHE.get(path)
    if cached is None:
        cached = native.Deck(path)  # pylint: disable=c-extension-no-member
        _DECK_CACHE[path] = cached
    return cached


RewardFn = Callable[[Sequence[int], Sequence[int], int, Optional[int]], float]
"""`(previous_tokens, current_tokens, player_id, winner) -> reward`."""

_SHAPING_COEF = 0.3
"""Weight of the dense potential-based shaping term relative to the terminal reward."""
_SHAPING_GAMMA = 0.99
"""Discount factor used in the shaping term, matches MaskablePPO's default."""


def lead_margin(tokens: Sequence[int], player_id: int) -> int:
    """One player's point tokens minus the best of everyone else's.

    This is the quantity that actually has to go positive to win (strictly
    more tokens than every other player at the threshold), unlike a raw
    token count, which rewards hoarding even while an opponent races ahead
    faster.

    Deliberately a plain Python loop over a list rather than numpy: it runs
    several times per micro-step on a table of at most six seats, where
    numpy's per-call overhead costs more than the arithmetic it saves.
    """
    best_other = 0
    for seat, count in enumerate(tokens):
        if seat != player_id and count > best_other:
            best_other = count
    return tokens[player_id] - best_other


def default_reward_fn(
    prev_tokens: Sequence[int],
    tokens: Sequence[int],
    player_id: int,
    winner: Optional[int],
) -> float:
    """+1 to the winner and -1 to everyone else at game end (the sparse
    ground-truth objective), plus a potential-based dense shaping term
    tracking each step's change in lead margin.

    Potential-based shaping (Ng, Harada and Russell 1999): adding
    `gamma * phi(s') - phi(s)` for any potential function `phi` provably
    leaves the optimal policy unchanged, unlike an arbitrary dense bonus.
    That gives PPO a much denser signal across a long sparse-terminal
    episode without distorting what "best play" means, a real risk with
    naive shaping (e.g. rewarding raw token gains regardless of standing).
    """
    phi_before = lead_margin(prev_tokens, player_id)
    phi_after = lead_margin(tokens, player_id)
    shaped = _SHAPING_COEF * (_SHAPING_GAMMA * phi_after - phi_before)
    terminal = 0.0 if winner is None else (1.0 if player_id == winner else -1.0)
    return float(shaped + terminal)


class SasquatchAECEnv(AECEnv):  # pylint: disable=abstract-method
    """`Sell Me a Sasquatch` as a PettingZoo `AECEnv`.

    Three to six players use buyer mode; exactly two players automatically
    use the materially different two-player variant, decided by the Rust
    engine from `num_players`, not by this wrapper.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "sell_me_a_sasquatch_v0"}
    """PettingZoo metadata: supported render modes and the env's registered name."""

    def __init__(
        self,
        num_players: int = 4,
        deck_config_path: "str | native.Deck" = DEFAULT_DECK_PATH,
        render_mode: str | None = None,
        reward_fn: RewardFn | None = None,
        max_actions: int | None = None,
    ):
        super().__init__()
        if not (
            sasquatch_spaces.MIN_PLAYERS <= num_players <= sasquatch_spaces.MAX_PLAYERS
        ):
            raise ValueError(f"num_players must be 2..=6, got {num_players}")
        self.num_players = num_players
        self.deck = load_deck(deck_config_path)
        self.render_mode = render_mode
        self.reward_fn = reward_fn or default_reward_fn
        # default to the width every table size needs, so a checkpoint
        # trained on mixed table sizes drops straight into any of them
        self.max_actions = max_actions or sasquatch_spaces.max_legal_actions(self.deck)

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        self._game: native.Game | None = None
        self._prev_tokens: Sequence[int] = [0] * num_players
        self._personas = np.zeros(
            (num_players, sasquatch_spaces.NOISE_LEN), dtype=np.float32
        )

    # Gymnasium/PettingZoo space plumbing

    @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def observation_space(self, agent):  # type: ignore
        """This agent's observation space, identical for every agent."""
        return sasquatch_spaces.observation_space(
            self.max_actions, with_action_mask=True
        )

    @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def action_space(self, agent):  # type: ignore
        """This agent's action space, identical for every agent."""
        return sasquatch_spaces.action_space(self.max_actions)

    # lifecycle

    def reset(self, seed=None, options=None):
        """Starts a new episode."""
        rng = np.random.default_rng(seed)
        if seed is None:
            seed = int(rng.integers(0, 2**63 - 1))
        self._game = native.Game(
            self.num_players, self.deck, int(seed)
        )  # pylint: disable=c-extension-no-member

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._prev_tokens = [0] * self.num_players
        self._personas = sasquatch_spaces.sample_personas(rng, self.num_players)

        self._update_agent_selection()

    def _update_agent_selection(self):
        """Sets `agent_selection` to whichever agent the engine says is active."""
        if not self.agents:
            self.agent_selection = None
            return
        active = self._game.active_player()  # type: ignore
        if active is None:
            # the game ends for the whole table at once, not one agent at a
            # time. per pettingzoo's _was_dead_step convention each agent
            # still needs one more agent_selection turn so its terminal
            # reward is delivered via last() before being pruned;
            # _was_dead_step cascades through the rest from here
            self.agent_selection = self.agents[0]
            return
        self.agent_selection = self.possible_agents[active]

    def step(self, action):
        """Applies one action for the currently selected agent."""
        if not self.agents:
            return
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_id = self.agent_name_mapping[agent]
        self._cumulative_rewards[agent] = 0.0

        legal_count = self._game.legal_action_count(player_id)  # type: ignore
        idx = int(action)
        if not 0 <= idx < legal_count:
            # an out-of-range (masked-out) index is a policy bug, not a
            # rules violation. illegal actions default to masking, not
            # raising, so fail soft onto the first legal action
            idx = 0
        done, winner = self._game.step_index(player_id, idx)  # type: ignore

        tokens = self._game.point_tokens()  # type: ignore
        for a in self.agents:
            self.rewards[a] = self.reward_fn(
                self._prev_tokens,
                tokens,
                self.agent_name_mapping[a],
                winner if done else None,
            )
        self._prev_tokens = tokens
        if done:
            for a in self.agents:
                self.terminations[a] = True

        self._update_agent_selection()
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        """This agent's current observation, including its action mask."""
        player_id = self.agent_name_mapping[agent]
        obs = sasquatch_spaces.empty_observation(self.max_actions)
        legal_count = self._game.encode(player_id, obs["state"], obs["actions"])  # type: ignore
        obs["state"][sasquatch_spaces.NOISE_OFFSET :] = self._personas[player_id]
        obs["action_mask"] = sasquatch_spaces.action_mask(legal_count, self.max_actions)
        return obs

    def render(self):
        """Renders the current game state as text, or prints it in human mode."""
        from . import render as render_module  # pylint: disable=import-outside-toplevel

        if self.render_mode is None:
            return None
        text = render_module.render_state(self._game)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        """Releases the underlying game."""
        self._game = None


def env(
    num_players: int = 4,
    deck_config_path: "str | native.Deck" = DEFAULT_DECK_PATH,
    render_mode: str | None = None,
    reward_fn: RewardFn | None = None,
):
    """Standard PettingZoo entry point: wraps the raw env with the usual
    order-enforcing / out-of-bounds wrappers."""
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = SasquatchAECEnv(
        num_players=num_players,
        deck_config_path=deck_config_path,
        render_mode=internal_render_mode,
        reward_fn=reward_fn,
    )
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e
