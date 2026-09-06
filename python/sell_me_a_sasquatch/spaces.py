"""Fixed-shape observation/action space definitions (§3.4).

Every number here is *read from the Rust encoder* rather than declared
again on this side. `engine/src/encode.rs` owns the observation layout, and
`GameState::max_legal_actions` owns the action-space width, so the two can
never silently drift apart.

Two conventions are worth knowing before reading anything else:

- **The observation is one flat float32 vector plus one action-feature
  matrix**, not a dict of per-card slots. Hands and Collections are
  unordered sets, so they are encoded as per-class counts (permutation
  invariant, and far smaller than a padded slot per card). Everything is
  padded to `MAX_PLAYERS`/`MAX_DEALS` and expressed relative to the
  observer's own seat, which is what lets a single policy play 2- through
  6-player games.

- **The action space is `Discrete(max_actions)` with an ordinal encoding**:
  index `i` means "the i-th entry of `legal_actions()` right now", not a
  fixed absolute action. That sidesteps a combinatorially complete encoding
  of the nested `Action` type (deal triples, removal subsets, ...) while
  keeping a fixed-shape space with an explicit mask, per §3.4. Because the
  meaning of `i` changes state to state, the observation carries an
  `actions` matrix describing what each candidate index actually does - see
  `policy.py`, which scores candidates from those descriptions instead of
  having to memorize the engine's enumeration order.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from gymnasium import spaces

from . import _native as native

# Observation layout, straight from the Rust encoder.
OBS_LEN: int = native.OBS_LEN
ACTION_FEAT_LEN: int = native.ACTION_FEAT_LEN
NOISE_OFFSET: int = native.NOISE_OFFSET
NOISE_LEN: int = native.NOISE_LEN
MAX_PLAYERS: int = native.MAX_PLAYERS
MAX_DEALS: int = native.MAX_DEALS
MIN_PLAYERS: int = 2

ALL_PLAYER_COUNTS: tuple[int, ...] = tuple(range(MIN_PLAYERS, MAX_PLAYERS + 1))

CARD_CLASSES: list[str] = list(native.CARD_CLASS_NAMES)

# Must match `Phase::name()` in engine/src/phase.rs exactly.
PHASES: list[str] = [
    "deal_offer_submit",
    "deal_offer_reveal",
    "buyer_peek",
    "thingamabob_window",
    "buyer_chooses_deal",
    "respond_to_deal",
    "nasty_resolution",
    "game_over",
]

# Features are all ratios/one-hots/squashed counts, but a lead margin can go
# mildly negative and a Collection can briefly overshoot its normalizer, so
# the declared bounds leave room rather than clipping real values.
_FEATURE_LIMIT = 4.0


def max_legal_actions(deck: "native.Deck", player_counts: Iterable[int] = ALL_PLAYER_COUNTS) -> int:
    """Widest action space any of `player_counts` needs with this deck.

    Sizing the space from the deck (rather than a hand-audited constant)
    means a custom `deck.toml` or a narrower table range resizes the policy
    head automatically instead of silently truncating legal actions.
    """
    counts = tuple(player_counts)
    if not counts:
        raise ValueError("player_counts must not be empty")
    return max(deck.max_legal_actions(n) for n in counts)


def observation_space(max_actions: int, with_action_mask: bool = False) -> spaces.Dict:
    """`state` is the tableau; `actions` describes each candidate action.

    `with_action_mask` adds the mask to the observation itself, which is the
    PettingZoo convention for `AECEnv`. Single-agent training passes masks
    out of band instead (sb3-contrib's `ActionMasker`), so the mask stays out
    of the policy's input there.
    """
    fields = {
        "state": spaces.Box(low=-_FEATURE_LIMIT, high=_FEATURE_LIMIT, shape=(OBS_LEN,), dtype=np.float32),
        "actions": spaces.Box(low=-_FEATURE_LIMIT, high=_FEATURE_LIMIT, shape=(max_actions, ACTION_FEAT_LEN), dtype=np.float32),
    }
    if with_action_mask:
        fields["action_mask"] = spaces.MultiBinary(max_actions)
    return spaces.Dict(fields)


def action_space(max_actions: int) -> spaces.Discrete:
    return spaces.Discrete(max_actions)


def action_mask(legal_count: int, max_actions: int, out: np.ndarray | None = None) -> np.ndarray:
    """The ordinal encoding makes every mask a prefix of ones, so this is a
    fill rather than a per-action test."""
    if out is None:
        out = np.zeros(max_actions, dtype=np.int8)
    else:
        out.fill(0)
    out[: min(legal_count, max_actions)] = 1
    return out


def sample_personas(rng: np.random.Generator, num_players: int) -> np.ndarray:
    """One random "persona" vector per seat, resampled each episode.

    The last `NOISE_LEN` slots of the state vector are left empty by the
    engine for exactly this. A policy that is deterministic given the state
    plays the same opening from the same deal every time - easy to read and
    a poor explorer. Conditioning on a latent that is *constant within an
    episode but resampled across episodes* lets one set of weights express a
    family of coherent strategies and commit to one per game, instead of
    re-rolling its personality on every micro-turn (which is all that
    sampling from the action distribution gives you).
    """
    return rng.standard_normal((num_players, NOISE_LEN), dtype=np.float32)


def empty_observation(max_actions: int) -> dict[str, np.ndarray]:
    """Zeroed buffers of the right dtype/shape for `Game.encode` to fill."""
    return {
        "state": np.zeros(OBS_LEN, dtype=np.float32),
        "actions": np.zeros((max_actions, ACTION_FEAT_LEN), dtype=np.float32),
    }


def encode_for_player(game, player: int, max_actions: int, persona: np.ndarray | None = None):
    """One-call observation for `player`, for callers outside the training
    loop (the web app, evaluation, ad-hoc analysis).

    Returns `(observation, action_mask, legal_count)`. `persona` fills the
    per-episode noise slots; left out, they stay zero, which is a
    perfectly valid - just maximally bland - persona."""
    obs = empty_observation(max_actions)
    legal_count = game.encode(player, obs["state"], obs["actions"])
    if persona is not None:
        obs["state"][NOISE_OFFSET:] = persona
    return obs, action_mask(legal_count, max_actions), legal_count


def describe_layout() -> Sequence[str]:
    """Human-readable summary, for `scripts/bench.py` and debugging."""
    return (
        f"state: {OBS_LEN} floats (last {NOISE_LEN} = per-episode persona, offset {NOISE_OFFSET})",
        f"actions: (n, {ACTION_FEAT_LEN}) floats, one row per candidate action",
        f"card classes: {len(CARD_CLASSES)}",
        f"seats/deals padded to: {MAX_PLAYERS}/{MAX_DEALS}",
    )
