"""Plain-text renderer for debugging. Not used for training, since the
engine has no concept of a human-facing UI; just for eyeballing a rollout."""

from __future__ import annotations


def _card_label(game, card_id: int) -> str:
    """A short human-readable label for one card."""
    kind = game.card_kind(card_id)
    name = game.card_name(card_id)
    return f"{name}[{kind}]#{card_id}"


def render_state(game, observer: int | None = None) -> str:
    """Renders the full engine state (or, if `observer` is given, that
    player's filtered observation) as a human-readable multi-line string."""
    lines = []
    lines.append(f"phase={game.current_phase()} turn_leader={game.turn_leader()} active={game.active_players()}")
    if game.is_game_over():
        lines.append(f"GAME OVER - winner: player_{game.winner()}")

    if observer is None:
        obs = game.observation(0)
    else:
        obs = game.observation(observer)
        lines.append(f"player_{observer} hand: {', '.join(_card_label(game, c) for c in obs.own_hand)}")

    lines.append("point tokens: " + ", ".join(f"p{p}={t}" for p, t in enumerate(obs.point_tokens)))
    lines.append("collection sizes: " + ", ".join(f"p{p}={len(c)}" for p, c in enumerate(obs.collections)))

    if obs.deals:
        lines.append("active deals:")
        for deal in obs.deals:
            revealed = ", ".join(_card_label(game, c) for c in deal.revealed_cards)
            lines.append(f"  seller=player_{deal.seller} revealed=[{revealed}] hidden={deal.num_hidden}")
    else:
        lines.append("active deals: none")

    lines.append(f"draw_pile={obs.draw_pile_len} discard_pile={obs.discard_pile_len}")
    return "\n".join(lines)
