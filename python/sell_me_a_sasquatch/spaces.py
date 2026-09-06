"""Fixed-shape observation/action space definitions (§3.4).

Hand sizes, deal contents, and the legal-action count all vary turn to turn,
but Gymnasium/PettingZoo spaces must be fixed-shape. This module defines the
padding/masking conventions used throughout:

- Cards are never observed by raw `CardId` (an arbitrary, per-episode
  integer with no meaning to a policy). They're bucketed into a small fixed
  vocabulary of *card classes* (tier / Nasty kind / Thingamabob kind), with
  class 0 reserved as the empty-slot sentinel.
- The action space is `Discrete(MAX_ACTIONS)`: at any step, action index `i`
  means "the i-th entry of `legal_actions()` right now" (an *ordinal*
  encoding), not a fixed absolute action meaning. This sidesteps needing a
  combinatorially complete fixed encoding of the full nested `Action` space
  (SubmitDeal card triples, removal-target subsets, etc.) while still giving
  a fixed-shape `Discrete` action space with an explicit `action_mask`, per
  §3.4's requirement. The policy conditions on the observation (which fully
  describes hands/collections/deals) to pick a meaningful ordinal index.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

MAX_HAND = 8
MAX_COLLECTION = 40
MAX_DEALS = 6
MAX_DEAL_CARDS = 4  # a deal starts at 3 but can grow via Cryptozooptic Expander
MAX_ACTIONS = 256

CARD_CLASSES: list[str] = [
    "Creature:Giant",
    "Creature:Big",
    "Creature:Medium",
    "Creature:Tiny",
    "Nasty:Poison Pill Bug",
    "Nasty:Loan Shark",
    "Nasty:Trojan Horse",
    "Thingamabob:Platonic Isolator",
    "Thingamabob:Detrital Repositioner",
    "Thingamabob:Super Detrital Repositioner",
    "Thingamabob:Cryptozooptic Expander",
    "Thingamabob:Spectroelectric Optimeter",
]
EMPTY_CARD_CLASS = 0
NUM_CARD_CLASSES = 1 + len(CARD_CLASSES)
_CARD_CLASS_INDEX = {name: i + 1 for i, name in enumerate(CARD_CLASSES)}

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
_PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}


def card_class_id(kind_str: str | None) -> int:
    """Maps a `Game.card_kind()` string (e.g. `"Creature:Giant"`) to its
    fixed class id, or `EMPTY_CARD_CLASS` for `None` / an unknown string."""
    if kind_str is None:
        return EMPTY_CARD_CLASS
    return _CARD_CLASS_INDEX.get(kind_str, EMPTY_CARD_CLASS)


def phase_id(phase: str) -> int:
    return _PHASE_INDEX.get(phase, len(PHASES) - 1)


def observation_space(num_players: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "own_hand": spaces.MultiDiscrete(np.full(MAX_HAND, NUM_CARD_CLASSES, dtype=np.int64)),
            "own_hand_len": spaces.Discrete(MAX_HAND + 1),
            # Flattened (not (num_players, MAX_COLLECTION)) - SB3's obs
            # flattening/preprocessing only supports 1-D MultiDiscrete nvec.
            "collections": spaces.MultiDiscrete(np.full(num_players * MAX_COLLECTION, NUM_CARD_CLASSES, dtype=np.int64)),
            "collection_lens": spaces.Box(low=0, high=MAX_COLLECTION, shape=(num_players,), dtype=np.int32),
            "point_tokens": spaces.Box(low=0, high=99, shape=(num_players,), dtype=np.int32),
            "deal_present": spaces.MultiBinary(MAX_DEALS),
            "deal_seller": spaces.MultiDiscrete(np.full(MAX_DEALS, num_players + 1, dtype=np.int64)),
            "deal_revealed_cards": spaces.MultiDiscrete(np.full(MAX_DEALS * MAX_DEAL_CARDS, NUM_CARD_CLASSES, dtype=np.int64)),
            "deal_num_hidden": spaces.Box(low=0, high=MAX_DEAL_CARDS, shape=(MAX_DEALS,), dtype=np.int32),
            "phase": spaces.Discrete(len(PHASES)),
            "turn_leader": spaces.Discrete(num_players),
            "draw_pile_len": spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
            "discard_pile_len": spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
            "action_mask": spaces.MultiBinary(MAX_ACTIONS),
        }
    )


def action_space() -> spaces.Discrete:
    return spaces.Discrete(MAX_ACTIONS)


def _pad_card_classes(card_kinds: list[str | None], width: int) -> np.ndarray:
    arr = np.full(width, EMPTY_CARD_CLASS, dtype=np.int64)
    for i, k in enumerate(card_kinds[:width]):
        arr[i] = card_class_id(k)
    return arr


def vectorize_observation(game, observation, legal_action_count: int, num_players: int) -> dict:
    """Builds the fixed-shape observation dict for one player from the
    engine's `Observation` object (see `bindings/src/lib.rs::PyObservation`)
    plus a `card_kind()` lookup on `game` for each raw `CardId`."""
    own_hand_kinds = [game.card_kind(c) for c in observation.own_hand]
    own_hand = _pad_card_classes(own_hand_kinds, MAX_HAND)

    collections = np.full((num_players, MAX_COLLECTION), EMPTY_CARD_CLASS, dtype=np.int64)
    collection_lens = np.zeros(num_players, dtype=np.int32)
    for p, cards in enumerate(observation.collections):
        kinds = [game.card_kind(c) for c in cards]
        collections[p] = _pad_card_classes(kinds, MAX_COLLECTION)
        collection_lens[p] = min(len(cards), MAX_COLLECTION)
    collections = collections.reshape(-1)  # flattened to match the 1-D space

    point_tokens = np.array(observation.point_tokens, dtype=np.int32)

    deal_present = np.zeros(MAX_DEALS, dtype=np.int8)
    deal_seller = np.full(MAX_DEALS, num_players, dtype=np.int64)  # sentinel = "no deal"
    deal_revealed = np.full((MAX_DEALS, MAX_DEAL_CARDS), EMPTY_CARD_CLASS, dtype=np.int64)
    deal_num_hidden = np.zeros(MAX_DEALS, dtype=np.int32)
    for i, deal in enumerate(observation.deals[:MAX_DEALS]):
        deal_present[i] = 1
        deal_seller[i] = deal.seller
        deal_revealed[i] = _pad_card_classes([game.card_kind(c) for c in deal.revealed_cards], MAX_DEAL_CARDS)
        deal_num_hidden[i] = deal.num_hidden
    deal_revealed = deal_revealed.reshape(-1)  # flattened to match the 1-D space

    action_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    action_mask[: min(legal_action_count, MAX_ACTIONS)] = 1

    return {
        "own_hand": own_hand,
        "own_hand_len": min(len(observation.own_hand), MAX_HAND),
        "collections": collections,
        "collection_lens": collection_lens,
        "point_tokens": point_tokens,
        "deal_present": deal_present,
        "deal_seller": deal_seller,
        "deal_revealed_cards": deal_revealed,
        "deal_num_hidden": deal_num_hidden,
        "phase": phase_id(observation.phase),
        "turn_leader": observation.turn_leader,
        "draw_pile_len": np.array([observation.draw_pile_len], dtype=np.int32),
        "discard_pile_len": np.array([observation.discard_pile_len], dtype=np.int32),
        "action_mask": action_mask,
    }
