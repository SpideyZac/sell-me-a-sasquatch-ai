"""Live tracker: drives a real `native.Game` for an actual physical game.

Unlike watch/play, nothing here is randomly dealt. Every card keeps an
arbitrary placeholder kind until the moment it actually becomes known at
the table (your starting hand up front, and any other card exactly when
it's revealed), at which point `Game.pin_kind` (engine/src/game.rs)
overwrites it with the truth. Every other rule (collections, point tokens,
discard and draw piles, nasty and creature set trade-ins, turn order) then
runs on the real engine exactly as in watch/play, so you never track any
of that by hand.

State machine, in one pass over a turn:
  1. If your own hand has any not-yet-pinned card (session start, or right
     after a refill), that's always the first thing asked for.
  2. Otherwise, whoever's turn it is has their legal actions grouped by
     what's actually distinguishable right now (`_group_actions`); options
     that would look identical (e.g. every combo of an opponent's fully
     unknown hand) collapse to one, and get auto-applied with no prompt at
     all when there's nothing left to ask.
  3. A choice that's genuinely open (two or more distinguishable options)
     is shown as a `choose_action` prompt, scored by the advisor model
     when it's your own turn.
  4. Applying some choices needs a card's kind pinned first (an opponent's
     revealed card, a hand card from a Cryptozootic Expander, a peeked
     card) or needs every still-hidden card across all active deals
     pinned before a `ChooseDeal`/`RespondToDeal` can be resolved correctly
     (creature/nasty set completion depends on the truth). Those show up as
     `pin_resolution` / `reveal_kind` prompts instead of auto-applying.
"""

from __future__ import annotations

from sell_me_a_sasquatch import _native as native
from sell_me_a_sasquatch import spaces as sasquatch_spaces

from card_display import card_dict, display_name, human_card_class_label
from models import get_model, observation_width
from scoring import rank_actions

MAX_SETTLE_STEPS = 2000
"""Safety cap on auto-applied steps per settle pass, guards against an infinite loop bug."""


class LiveGameError(Exception):
    """Any invalid live-session request (bad payload, stale prompt, engine
    rejection); callers turn this into a 400 with `str(e)`."""


class LiveSession:
    """One live-tracked physical game in progress."""

    def __init__(self, num_players: int, human_seat: int, deck_path: str, seed: int, advisor_model_spec: str, first_player: int | None = None):
        """Starts a new live session with every card unpinned."""
        self.game = native.Game(num_players, deck_path, seed, first_player)
        self.num_players = num_players
        self.human_seat = human_seat
        self.advisor_model_spec = advisor_model_spec
        self.log: list[str] = []
        self._awaiting_initial_hand = True
        # {"seat", "action", "unpinned": [CardId], "seller_of": {CardId: seller}}
        self.pending_resolution: dict | None = None
        """A choose_deal/respond_to_deal action waiting on every hidden deal card to be pinned."""
        # {"reveal_type": "opponent_deal_reveal" | "spectro" | "buyer_peek_followup", ...}
        self.pending_reveal: dict | None = None
        """A single revealed card waiting on the human to say what it is."""
        self._settle()

    # public API

    def state(self) -> dict:
        """The full JSON state for the live tracker UI: hands, collections, deals, and the current prompt."""
        game = self.game
        obs = game.observation(self.human_seat)
        return {
            "num_players": self.num_players,
            "your_seat": self.human_seat,
            "phase": game.current_phase(),
            "turn_leader": game.turn_leader(),
            "is_game_over": game.is_game_over(),
            "winner": game.winner(),
            "hand": [card_dict(game, c) for c in game.player_hand(self.human_seat)],
            "point_tokens": [game.player_point_tokens(p) for p in range(self.num_players)],
            "collections": [[card_dict(game, c) for c in game.player_collection(p)] for p in range(self.num_players)],
            "deals": [
                {"seller": d.seller, "revealed": [card_dict(game, c) for c in d.revealed_cards], "num_hidden": d.num_hidden} for d in obs.deals
            ],
            "draw_pile_len": obs.draw_pile_len,
            "discard_pile_len": obs.discard_pile_len,
            "log": self.log[-40:],
            "kind_options": self._kind_options(),
            "prompt": self._current_prompt(),
        }

    def respond(self, payload: dict) -> None:
        """Applies the human's answer to whatever the current prompt is
        asking for; raises `LiveGameError` on a bad/stale payload."""
        prompt = self._current_prompt()
        t = prompt["type"]
        if t == "pin_hand":
            self._pin_hand(_require_list(payload, "kinds"))
        elif t == "pin_resolution":
            self._pin_resolution(_require_list(payload, "kinds"))
        elif t == "reveal_kind":
            kind = payload.get("kind")
            if not kind:
                raise LiveGameError("expected 'kind'")
            self._fulfill_reveal(kind)
        elif t == "choose_action":
            index = payload.get("index")
            if index is None:
                raise LiveGameError("expected 'index'")
            self._choose(int(index))
        else:
            raise LiveGameError("the game is already over")

    # prompt construction

    def _current_prompt(self) -> dict:
        """The single prompt the UI should show right now, in priority order."""
        unpinned_hand = self._unpinned_human_hand()
        if unpinned_hand:
            return {
                "type": "pin_hand",
                "count": len(unpinned_hand),
                "context": "starting_hand" if self._awaiting_initial_hand else "hand_refill",
            }
        if self.pending_resolution is not None:
            pr = self.pending_resolution
            seen: dict[int, int] = {}
            cards = []
            for cid in pr["unpinned"]:
                seller = pr["seller_of"][cid]
                seen[seller] = seen.get(seller, 0) + 1
                cards.append({"seller": seller, "position": seen[seller]})
            return {"type": "pin_resolution", "count": len(cards), "cards": cards}
        if self.pending_reveal is not None:
            pr = self.pending_reveal
            rt = pr["reveal_type"]
            if rt == "opponent_deal_reveal":
                label = f"What did player_{pr['seat']} reveal?"
            elif rt == "spectro":
                label = f"What was the hidden card revealed in player_{pr['target_deal']}'s deal?"
            elif rt == "buyer_peek_followup":
                label = f"What did the peek into player_{pr['seller']}'s deal reveal?"
            else:  # pragma: no cover, defensive
                raise LiveGameError(f"unknown reveal_type: {rt}")
            return {"type": "reveal_kind", "label": label}
        if self.game.is_game_over():
            return {"type": "game_over", "winner": self.game.winner()}

        seat = self.game.active_player()
        legal = self.game.legal_actions(seat)
        groups = self._group_actions(seat, legal)
        if seat == self.human_seat:
            self._score_groups(legal, groups)
        return {
            "type": "choose_action",
            "seat": seat,
            "your_turn": seat == self.human_seat,
            "phase": self.game.current_phase(),
            "options": [
                {
                    "index": i,
                    "label": g["label"],
                    **({"score": g["score"]} if "score" in g else {}),
                    **({"recommended": True} if g.get("recommended") else {}),
                }
                for i, g in enumerate(groups)
            ],
        }

    def _kind_options(self) -> list[dict]:
        """Every card class the UI can offer for a pin prompt, with remaining supply."""
        supply = self.game.kind_supply()
        return [{"value": c, "label": human_card_class_label(c), "remaining": supply.get(c, 0)} for c in sasquatch_spaces.CARD_CLASSES]

    # the settle loop: auto-apply everything that isn't a genuine choice

    def _settle(self) -> None:
        """Auto-applies actions until a real choice or a pin prompt is needed."""
        for _ in range(MAX_SETTLE_STEPS):
            if self._unpinned_human_hand():
                return
            if self.pending_resolution is not None or self.pending_reveal is not None:
                return
            if self.game.is_game_over():
                return

            seat = self.game.active_player()
            legal = self.game.legal_actions(seat)
            if not legal:
                return
            groups = self._group_actions(seat, legal)
            if len(groups) > 1:
                return
            if not self._dispatch(seat, groups[0]["action"]):
                return
        raise LiveGameError("live session failed to settle, this is a bug, please report it")

    def _dispatch(self, seat: int, action) -> bool:
        """Applies `action` if it can be applied right now, or defers it via
        `pending_resolution`/`pending_reveal` when a card it touches isn't
        pinned yet. Returns True if the game state actually advanced (so the
        settle loop should keep going), False if it's now waiting on you."""
        ref = self._pin_requirement(action)
        if ref is None:
            self._apply(seat, action)
            return True
        kind, payload = ref
        if kind == "resolution":
            return self._start_resolution(seat, action)
        if kind == "reveal_card":
            self.pending_reveal = {"reveal_type": "opponent_deal_reveal", "seat": seat, "card": payload}
            return False
        if kind == "spectro":
            d = action.to_dict()
            self.pending_reveal = {"reveal_type": "spectro", "action": action, "seat": seat, "card": payload, "target_deal": d["target_deal"]}
            return False
        if kind == "buyer_peek":
            return self._apply_buyer_peek(seat, payload)
        raise LiveGameError(f"unhandled pin requirement: {kind}")  # pragma: no cover, defensive

    def _pin_requirement(self, action) -> "tuple[str, int] | None":
        """What (if anything) needs a kind pinned before/after `action` can
        be meaningfully applied. `None` means it can just be applied now."""
        d = action.to_dict()
        t = d["type"]
        if t in ("choose_deal", "respond_to_deal"):
            return ("resolution", -1)
        if t == "reveal_card" and not self.game.is_pinned(d["card"]):
            return ("reveal_card", d["card"])
        if t == "buyer_peek":
            return ("buyer_peek", d["target_seller"])
        if t == "play_thingamabob" and d.get("effect") == "spectroelectric_optimeter" and not self.game.is_pinned(d["target_card"]):
            return ("spectro", d["target_card"])
        return None

    def _start_resolution(self, seat: int, action) -> bool:
        """`ChooseDeal`/`RespondToDeal` award every active deal's full
        contents into a collection in one atomic engine call, and set
        completion has to see the truth to compute correctly, so every
        still-hidden card across all deals must be pinned before we call
        it, not just the chosen deal's."""
        unpinned = []
        seller_of = {}
        for seller in self._all_active_deal_sellers():
            for cid in self.game.hidden_cards_in_deal(seller):
                if not self.game.is_pinned(cid):
                    unpinned.append(cid)
                    seller_of[cid] = seller
        if not unpinned:
            self._apply(seat, action)
            return True
        self.pending_resolution = {"seat": seat, "action": action, "unpinned": unpinned, "seller_of": seller_of}
        return False

    def _apply_buyer_peek(self, seat: int, target_seller: int) -> bool:
        """Applies a buyer peek and defers to a reveal prompt unless the peeked card is already pinned."""
        action = native.Action.buyer_peek(target_seller)
        who = "You" if seat == self.human_seat else f"player_{seat}"
        self.log.append(f"{who}: peeked into player_{target_seller}'s deal")
        result = self.game.step(seat, action)
        card = next((ev["card"] for ev in result.events() if ev["type"] == "buyer_peeked"), None)
        if result.done:  # pragma: no cover, a peek can't itself end the game, kept for safety
            self.log.append(f"Game over - winner: player_{result.winner}")
        if card is not None and self.game.is_pinned(card):
            # recycled via a discard/draw-pile reshuffle, its kind is
            # already known truth, nothing new to ask about
            self.log.append(f"The peek revealed: {display_name(self.game, card)}")
            return True
        self.pending_reveal = {"reveal_type": "buyer_peek_followup", "card": card, "seller": target_seller}
        return False

    def _apply(self, seat: int, action) -> None:
        """Applies an action that needs no further pinning and logs it."""
        who = "You" if seat == self.human_seat else f"player_{seat}"
        self.log.append(f"{who}: {self._live_action_label(seat, action)}")
        result = self.game.step(seat, action)
        if result.done:
            self.log.append(f"Game over - winner: player_{result.winner}")

    # responding to prompts

    def _pin_hand(self, kinds: list[str]) -> None:
        """Pins every unpinned card in the human's hand from a pin_hand response."""
        unpinned = self._unpinned_human_hand()
        if len(kinds) != len(unpinned):
            raise LiveGameError(f"expected {len(unpinned)} card(s), got {len(kinds)}")
        for cid, kind in zip(unpinned, kinds):
            self._pin(cid, kind)
        was_initial = self._awaiting_initial_hand
        self._awaiting_initial_hand = False
        if was_initial:
            self.log.append(f"Your starting hand: {', '.join(display_name(self.game, c) for c in unpinned)}")
        else:
            self.log.append(f"You drew: {', '.join(display_name(self.game, c) for c in unpinned)}")
        self._settle()

    def _pin_resolution(self, kinds: list[str]) -> None:
        """Pins every card a pending resolution was waiting on, then applies it."""
        if self.pending_resolution is None:
            raise LiveGameError("no pending resolution")
        pr = self.pending_resolution
        ids = pr["unpinned"]
        if len(kinds) != len(ids):
            raise LiveGameError(f"expected {len(ids)} card(s), got {len(kinds)}")
        for cid, kind in zip(ids, kinds):
            self._pin(cid, kind)
        self.pending_resolution = None
        self._apply(pr["seat"], pr["action"])
        self._settle()

    def _fulfill_reveal(self, kind: str) -> None:
        """Pins a pending reveal's card and applies whatever action it was blocking."""
        if self.pending_reveal is None:
            raise LiveGameError("no pending reveal")
        pr = self.pending_reveal
        self._pin(pr["card"], kind)
        self.pending_reveal = None
        if pr["reveal_type"] == "opponent_deal_reveal":
            self._apply(pr["seat"], native.Action.reveal_card(pr["card"]))
        elif pr["reveal_type"] == "spectro":
            self._apply(pr["seat"], pr["action"])
        elif pr["reveal_type"] == "buyer_peek_followup":
            self.log.append(f"The peek revealed: {display_name(self.game, pr['card'])}")
        else:  # pragma: no cover, defensive
            raise LiveGameError(f"unknown reveal_type: {pr['reveal_type']}")
        self._settle()

    def _choose(self, index: int) -> None:
        """Dispatches the group at `index` from the current choose_action prompt."""
        if self.game.is_game_over():
            raise LiveGameError("the game is already over")
        seat = self.game.active_player()
        legal = self.game.legal_actions(seat)
        groups = self._group_actions(seat, legal)
        if not (0 <= index < len(groups)):
            raise LiveGameError("choice index out of range")
        self._dispatch(seat, groups[index]["action"])
        self._settle()

    def _pin(self, card_id: int, kind: str) -> None:
        """Pins one card's kind, raising `LiveGameError` on the engine's own rejection."""
        try:
            self.game.pin_kind(card_id, kind)
        except Exception as e:  # noqa: BLE001, surface the engine's own message
            raise LiveGameError(str(e)) from e

    # small helpers

    def _unpinned_human_hand(self) -> list[int]:
        """Ids of the human's hand cards that still need a kind pinned."""
        return [c for c in self.game.player_hand(self.human_seat) if not self.game.is_pinned(c)]

    def _all_active_deal_sellers(self) -> list[int]:
        """Sellers of every currently active deal."""
        return [d.seller for d in self.game.observation(self.human_seat).deals]

    def _group_actions(self, seat: int, legal: list) -> list[dict]:
        """Collapses legal actions that would currently look identical (most
        commonly: every way to split up an opponent's fully-unknown hand)
        into one option, so a choice is only ever shown when it's real."""
        groups: list[dict] = []
        index_of_label: dict[str, int] = {}
        for i, action in enumerate(legal):
            label = self._live_action_label(seat, action)
            if label in index_of_label:
                groups[index_of_label[label]]["indices"].append(i)
            else:
                index_of_label[label] = len(groups)
                groups.append({"label": label, "indices": [i], "action": action})
        return groups

    def _score_groups(self, legal: list, groups: list[dict]) -> None:
        """Scores each action group with the advisor model and flags the best one as recommended."""
        model = get_model(self.advisor_model_spec)
        width = observation_width(model) or self.game.max_legal_actions()
        obs, mask, _ = sasquatch_spaces.encode_for_player(self.game, self.human_seat, width)
        ranked = dict(rank_actions(model, obs, mask))
        for g in groups:
            g["score"] = round(sum(ranked.get(i, 0.0) for i in g["indices"]), 3)
        if groups:
            max(groups, key=lambda g: g["score"])["recommended"] = True

    def _live_action_label(self, seat: int, action) -> str:
        """Like `card_display.describe_action`, but shows "???" (or, where
        the specific position is itself a real choice, "hidden card #N")
        for anything not yet pinned, never leaks a card's true kind before
        it's actually meant to be known."""
        d = action.to_dict()
        t = d["type"]
        game = self.game

        def label(cid: int) -> str:
            return display_name(game, cid) if game.is_pinned(cid) else "???"

        def positioned(seller: int, cid: int) -> str:
            if game.is_pinned(cid):
                return display_name(game, cid)
            hidden = game.hidden_cards_in_deal(seller)
            pos = hidden.index(cid) + 1 if cid in hidden else 0
            return f"hidden card #{pos}"

        def pile_label(cid: int) -> str:
            """Like `label`, but for a still-hidden 2-player deal card, names
            which pile it's in instead of a bare "???"; in two-player mode
            `reveal_card` can offer cards from either the active player's own
            pile or the pile offered to the other player, and which pile a
            flip came from is a real, physically-visible choice at the table
            even before the card's kind is known (unlike buyer mode's
            single-seller reveal, where every candidate is genuinely
            interchangeable and collapsing them to one option is correct)."""
            if game.is_pinned(cid):
                return display_name(game, cid)
            for seller in self._all_active_deal_sellers():
                if cid in game.hidden_cards_in_deal(seller):
                    whose = "your" if seller == self.human_seat else f"player_{seller}'s"
                    return f"a card from {whose} pile"
            return "???"

        if t == "submit_deal":
            return f"Offer deal: {', '.join(label(c) for c in d['cards'])}"
        if t == "two_player_submit_deal":
            own = ", ".join(label(c) for c in d["own_pile"]) or "nothing"
            other = ", ".join(label(c) for c in d["other_pile"]) or "nothing"
            return f"Split: keep [{own}], offer [{other}]"
        if t == "reveal_card":
            return f"Reveal {pile_label(d['card'])}"
        if t == "buyer_peek":
            return f"Peek into player_{d['target_seller']}'s deal"
        if t == "play_thingamabob":
            name = label(d["card"])  # always from seat's own collection, always already pinned
            effect = d["effect"]
            if effect == "platonic_isolator":
                return f"Play {name}: steal a token from player_{d['target_player']}"
            if effect == "remove_from_deals":
                removals = d["removals"]
                if not removals:
                    return f"Play {name} (discard, no removals)"
                parts = "; ".join(f"remove {positioned(s, c)} from player_{s}'s deal" for s, c in removals)
                return f"Play {name}: {parts}"
            if effect == "cryptozootic_expander":
                return f"Play {name}: add {label(d['hand_card'])} to player_{d['target_deal']}'s deal (face down)"
            if effect == "spectroelectric_optimeter":
                return f"Play {name}: reveal {positioned(d['target_deal'], d['target_card'])} in player_{d['target_deal']}'s deal"
            return f"Play {name}"
        if t == "pass_thingamabob_window":
            return "Pass"
        if t == "choose_deal":
            return f"Choose player_{d['seller']}'s deal"
        if t == "respond_to_deal":
            return "Reverse the deal (swap piles)" if d["reverse"] else "Accept the deal (keep your own pile)"
        if t == "resolve_nasty_penalty":
            taken = d["taken_cards"]
            return "Take nothing (decline)" if not taken else f"Take: {', '.join(label(c) for c in taken)}"
        return str(d)  # pragma: no cover, defensive


def _require_list(payload: dict, key: str) -> list:
    """Extracts `key` from `payload` as a list, raising `LiveGameError` if it's missing or the wrong type."""
    value = payload.get(key)
    if not isinstance(value, list):
        raise LiveGameError(f"expected '{key}': [...]")
    return value
