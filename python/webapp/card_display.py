"""Shared card-label helpers for the webapp (watch/play and the live tracker).

Creatures show their tier only ("Tiny Creature") rather than the engine's
flavor names (see `engine/src/card.rs::creature_flavor_name`); those are
non-mechanical fluff and just noise in this UI.
"""

from __future__ import annotations


def display_name(game, card_id: int) -> str:
    """A card's UI-facing name: tier only for creatures, the real name otherwise."""
    kind = game.card_kind(card_id)
    if kind and kind.startswith("Creature:"):
        return f"{kind.split(':', 1)[1]} Creature"
    return game.card_name(card_id) or f"card#{card_id}"


def card_dict(game, card_id: int) -> dict:
    """A card's id, display name, and mechanical kind, as a plain dict."""
    return {
        "id": card_id,
        "name": display_name(game, card_id),
        "kind": game.card_kind(card_id),
    }


def human_card_class_label(card_class: str) -> str:
    """Tier-only label for a `sasquatch_spaces.CARD_CLASSES` entry, e.g.
    `"Creature:Tiny"` becomes `"Tiny Creature"`; other kinds keep their real name."""
    kind, _, name = card_class.partition(":")
    return f"{name} Creature" if kind == "Creature" else name


def describe_action(game, action) -> str:
    """Full-information action label (assumes every referenced card's kind
    is meaningful to show), used by watch/play, where the engine deals a
    real (if simulated) deck. The live tracker uses its own pin-aware
    variant instead (`live_game.LiveSession._live_action_label`), since most
    of a live game's cards start out as meaningless placeholders."""
    d = action.to_dict()
    t = d["type"]

    def label(cid):
        return display_name(game, cid)

    if t == "submit_deal":
        return f"Offer deal: {', '.join(label(c) for c in d['cards'])}"
    if t == "two_player_submit_deal":
        own = ", ".join(label(c) for c in d["own_pile"]) or "nothing"
        other = ", ".join(label(c) for c in d["other_pile"]) or "nothing"
        return f"Split: keep [{own}], offer [{other}]"
    if t == "reveal_card":
        return f"Reveal {label(d['card'])}"
    if t == "buyer_peek":
        return f"Peek into player_{d['target_seller']}'s deal"
    if t == "play_thingamabob":
        name = label(d["card"])
        effect = d["effect"]
        if effect == "platonic_isolator":
            return f"Play {name}: steal a token from player_{d['target_player']}"
        if effect == "remove_from_deals":
            removals = d["removals"]
            if not removals:
                return f"Play {name} (discard, no removals)"
            parts = "; ".join(
                f"remove {label(c)} from player_{s}'s deal" for s, c in removals
            )
            return f"Play {name}: {parts}"
        if effect == "cryptozootic_expander":
            return f"Play {name}: add {label(d['hand_card'])} to player_{d['target_deal']}'s deal (face down)"  # pylint: disable=line-too-long
        if effect == "spectroelectric_optimeter":
            return f"Play {name}: reveal {label(d['target_card'])} in player_{d['target_deal']}'s deal"  # pylint: disable=line-too-long
        return f"Play {name}"
    if t == "pass_thingamabob_window":
        return "Pass"
    if t == "choose_deal":
        return f"Choose player_{d['seller']}'s deal"
    if t == "respond_to_deal":
        return (
            "Reverse the deal (swap piles)"
            if d["reverse"]
            else "Accept the deal (keep your own pile)"
        )
    if t == "resolve_nasty_penalty":
        taken = d["taken_cards"]
        return (
            "Take nothing (decline)"
            if not taken
            else f"Take: {', '.join(label(c) for c in taken)}"
        )
    return str(d)
