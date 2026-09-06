"""A very simple local web app for Sell Me a Sasquatch (three modes):

- Watch: a game of loaded AI models (or random policies) playing each other,
  stepped one micro-turn at a time.
- Play: you take one seat, AI models (or random) play the rest.
- Advisor: manually describe your hand (and, for the deal-choice advisor,
  what's been revealed of each seller's deal) and get a model's ranked
  recommendation - no real game session needed. This is an *approximation*:
  it builds a hand-entered snapshot of the same observation format the
  model was trained on rather than routing through a real `GameState`, so
  anything you don't specify defaults to a neutral/empty value.

Run with:
    cd python
    uv pip install -e ".[web,train]"   # train extra optional - only needed to load real models
    uv run python webapp/app.py
Then open http://127.0.0.1:5000/
"""

from __future__ import annotations

import glob
import itertools
import os
import random
import uuid

import numpy as np
from flask import Flask, abort, jsonify, render_template, request

from sell_me_a_sasquatch import _native as native
from sell_me_a_sasquatch import spaces as sasquatch_spaces
from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH
from sell_me_a_sasquatch.selfplay_env import random_masked_policy

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))

app = Flask(__name__)
GAMES: dict[str, "GameSession"] = {}
_MODEL_CACHE: dict[str, object] = {}


# model loading
def _resolve_model_path(spec: str) -> str:
    candidate = os.path.join(MODELS_DIR, spec)
    return candidate if os.path.exists(candidate) else spec


def get_model(spec: str):
    """Returns a loaded `MaskablePPO`, or `None` for the "random" policy."""
    spec = (spec or "random").strip()
    if spec.lower() == "random":
        return None
    if spec not in _MODEL_CACHE:
        from sb3_contrib import MaskablePPO  # lazy: only needed for real models

        _MODEL_CACHE[spec] = MaskablePPO.load(_resolve_model_path(spec), device="cpu")
    return _MODEL_CACHE[spec]


def get_policy(spec: str):
    """`(obs, mask) -> action_index` callable, matching every other policy
    in this codebase (see `selfplay_env.py`)."""
    model = get_model(spec)
    if model is None:
        return random_masked_policy

    def _policy(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return _policy


def list_available_models() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    paths = glob.glob(os.path.join(MODELS_DIR, "**", "*.zip"), recursive=True)
    return sorted(os.path.relpath(p, MODELS_DIR).replace(os.sep, "/") for p in paths)


# game sessions (Watch / Play)
class GameSession:
    def __init__(self, game, num_players: int, mode: str, human_seat: "int | None", seat_specs: list[str], seat_policies: list):
        self.game = game
        self.num_players = num_players
        self.mode = mode  # "watch" | "play"
        self.human_seat = human_seat
        self.seat_specs = seat_specs
        self.seat_policies = seat_policies
        self.log: list[str] = []


def _card_dict(game, card_id: int) -> dict:
    return {"id": card_id, "name": game.card_name(card_id), "kind": game.card_kind(card_id)}


def describe_action(game, action) -> str:
    d = action.to_dict()
    t = d["type"]

    def label(cid):
        return game.card_name(cid) or f"card#{cid}"

    if t == "submit_deal":
        return f"Offer deal: {', '.join(label(c) for c in d['cards'])}"
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
            parts = "; ".join(f"remove {label(c)} from player_{s}'s deal" for s, c in removals)
            return f"Play {name}: {parts}"
        if effect == "cryptozootic_expander":
            return f"Play {name}: add {label(d['hand_card'])} to player_{d['target_deal']}'s deal (face down)"
        if effect == "spectroelectric_optimeter":
            return f"Play {name}: reveal {label(d['target_card'])} in player_{d['target_deal']}'s deal"
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
    return str(d)


def _step_seat(session: GameSession, player: int) -> None:
    game = session.game
    legal = game.legal_actions(player)
    if not legal:
        return
    obs_native = game.observation(player)
    obs_vec = sasquatch_spaces.vectorize_observation(game, obs_native, len(legal), session.num_players)
    idx = session.seat_policies[player](obs_vec, obs_vec["action_mask"])
    idx = idx if isinstance(idx, int) and 0 <= idx < len(legal) else 0
    action = legal[idx]
    who = "You" if player == session.human_seat else f"player_{player}"
    session.log.append(f"{who}: {describe_action(game, action)}")
    result = game.step(player, action)
    if result.done:
        session.log.append(f"Game over - winner: player_{result.winner}")


def _auto_resolve_ai_turns(session: GameSession, max_steps: int = 1000) -> None:
    game = session.game
    for _ in range(max_steps):
        if game.is_game_over():
            return
        active = game.active_players()[0]
        if session.mode == "play" and active == session.human_seat:
            return
        _step_seat(session, active)


def spectator_view(session: GameSession) -> dict:
    game = session.game
    n = session.num_players
    hands, collections, tokens = [], [], []
    for p in range(n):
        obs = game.observation(p)
        hands.append([_card_dict(game, c) for c in obs.own_hand])
        collections.append([_card_dict(game, c) for c in obs.collections[p]])
        tokens.append(obs.point_tokens[p])
    obs0 = game.observation(0)
    deals = [{"seller": d.seller, "revealed": [_card_dict(game, c) for c in d.revealed_cards], "num_hidden": d.num_hidden} for d in obs0.deals]
    return {
        "mode": "watch",
        "num_players": n,
        "seat_specs": session.seat_specs,
        "phase": game.current_phase(),
        "turn_leader": game.turn_leader(),
        "active_players": game.active_players(),
        "is_game_over": game.is_game_over(),
        "winner": game.winner(),
        "hands": hands,
        "collections": collections,
        "point_tokens": tokens,
        "deals": deals,
        "log": session.log[-40:],
    }


def human_view(session: GameSession) -> dict:
    game = session.game
    seat = session.human_seat
    obs = game.observation(seat)
    your_turn = (not game.is_game_over()) and game.active_players() == [seat]
    legal = game.legal_actions(seat) if your_turn else []
    return {
        "mode": "play",
        "num_players": session.num_players,
        "seat_specs": session.seat_specs,
        "your_seat": seat,
        "phase": game.current_phase(),
        "turn_leader": game.turn_leader(),
        "your_turn": your_turn,
        "is_game_over": game.is_game_over(),
        "winner": game.winner(),
        "hand": [_card_dict(game, c) for c in obs.own_hand],
        "point_tokens": list(obs.point_tokens),
        "collections": [[_card_dict(game, c) for c in coll] for coll in obs.collections],
        "deals": [{"seller": d.seller, "revealed": [_card_dict(game, c) for c in d.revealed_cards], "num_hidden": d.num_hidden} for d in obs.deals],
        "legal_actions": [{"index": i, "label": describe_action(game, a)} for i, a in enumerate(legal)],
        "log": session.log[-40:],
    }


def _view_for(session: GameSession) -> dict:
    return spectator_view(session) if session.mode == "watch" else human_view(session)


def _get_session(game_id: str) -> GameSession:
    session = GAMES.get(game_id)
    if session is None:
        abort(404, "unknown game_id")
    return session


# routes: pages
@app.route("/")
def index():
    return render_template("index.html")


# routes: shared JSON API
@app.route("/api/models")
def api_models():
    return jsonify({"models": list_available_models()})


@app.route("/api/card_classes")
def api_card_classes():
    def human_label(cls: str) -> str:
        kind, _, name = cls.partition(":")
        return f"{name} Creature" if kind == "Creature" else name

    return jsonify({"classes": [{"value": c, "label": human_label(c)} for c in sasquatch_spaces.CARD_CLASSES]})


# routes: Watch / Play game sessions
@app.route("/api/games", methods=["POST"])
def api_new_game():
    body = request.get_json(force=True)
    num_players = int(body["num_players"])
    if not (2 <= num_players <= 6):
        return jsonify({"error": "num_players must be 2-6"}), 400
    mode = body.get("mode", "watch")
    if mode not in ("watch", "play"):
        return jsonify({"error": "mode must be 'watch' or 'play'"}), 400
    deck = body.get("deck") or DEFAULT_DECK_PATH
    seed = int(body["seed"]) if body.get("seed") not in (None, "") else random.randint(0, 2**63 - 1)

    seat_specs = body.get("seat_models") or ["random"] * num_players
    seat_specs = (seat_specs + ["random"] * num_players)[:num_players]
    human_seat = int(body["human_seat"]) if mode == "play" and body.get("human_seat") is not None else (0 if mode == "play" else None)

    try:
        game = native.Game(num_players, deck, seed)
    except Exception as e:  # noqa: BLE001 - surface engine setup errors to the UI as-is
        return jsonify({"error": str(e)}), 400

    seat_policies = []
    for i in range(num_players):
        if mode == "play" and i == human_seat:
            seat_policies.append(None)
        else:
            try:
                seat_policies.append(get_policy(seat_specs[i]))
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": f"failed to load model '{seat_specs[i]}': {e}"}), 400

    game_id = uuid.uuid4().hex[:12]
    session = GameSession(game, num_players, mode, human_seat, seat_specs, seat_policies)
    GAMES[game_id] = session

    if mode == "play":
        _auto_resolve_ai_turns(session)

    return jsonify({"game_id": game_id, "state": _view_for(session)})


@app.route("/api/games/<game_id>")
def api_get_game(game_id):
    return jsonify({"state": _view_for(_get_session(game_id))})


@app.route("/api/games/<game_id>/advance", methods=["POST"])
def api_advance(game_id):
    session = _get_session(game_id)
    if session.mode != "watch":
        return jsonify({"error": "advance is only for watch-mode games"}), 400
    if not session.game.is_game_over():
        _step_seat(session, session.game.active_players()[0])
    return jsonify({"state": _view_for(session)})


@app.route("/api/games/<game_id>/act", methods=["POST"])
def api_act(game_id):
    session = _get_session(game_id)
    if session.mode != "play":
        return jsonify({"error": "act is only for play-mode games"}), 400
    game = session.game
    if game.is_game_over():
        return jsonify({"error": "game is already over"}), 400
    if game.active_players() != [session.human_seat]:
        return jsonify({"error": "not your turn"}), 400

    body = request.get_json(force=True)
    legal = game.legal_actions(session.human_seat)
    action_index = int(body["action_index"])
    if not (0 <= action_index < len(legal)):
        return jsonify({"error": "action_index out of range"}), 400

    action = legal[action_index]
    session.log.append(f"You: {describe_action(game, action)}")
    result = game.step(session.human_seat, action)
    if result.done:
        session.log.append(f"Game over - winner: player_{result.winner}")
    else:
        _auto_resolve_ai_turns(session)
    return jsonify({"state": _view_for(session)})


@app.route("/api/games/<game_id>", methods=["DELETE"])
def api_delete_game(game_id):
    GAMES.pop(game_id, None)
    return jsonify({"ok": True})


# routes: move Advisor
def _blank_observation(num_players: int) -> dict:
    return {
        "own_hand": np.zeros(sasquatch_spaces.MAX_HAND, dtype=np.int64),
        "own_hand_len": 0,
        "collections": np.zeros(num_players * sasquatch_spaces.MAX_COLLECTION, dtype=np.int64),
        "collection_lens": np.zeros(num_players, dtype=np.int32),
        "point_tokens": np.zeros(num_players, dtype=np.int32),
        "deal_present": np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int8),
        "deal_seller": np.full(sasquatch_spaces.MAX_DEALS, num_players, dtype=np.int64),
        "deal_revealed_cards": np.zeros(sasquatch_spaces.MAX_DEALS * sasquatch_spaces.MAX_DEAL_CARDS, dtype=np.int64),
        "deal_num_hidden": np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int32),
        "phase": 0,
        "turn_leader": 0,
        "draw_pile_len": np.array([20], dtype=np.int32),
        "discard_pile_len": np.array([0], dtype=np.int32),
        "action_mask": np.zeros(sasquatch_spaces.MAX_ACTIONS, dtype=np.int8),
    }


def _pad_classes(classes: list[str], width: int) -> np.ndarray:
    arr = np.zeros(width, dtype=np.int64)
    for i, c in enumerate(classes[:width]):
        arr[i] = sasquatch_spaces.card_class_id(c)
    return arr


ADVISOR_DISCLAIMER = (
    "Approximate: built from what you entered, not a real game session - "
    "anything you didn't specify (other players' collections, deal history, etc.) "
    "defaults to empty/neutral. Currently covers the deal-offer and choose-deal decisions only."
)


def _rank_actions(model, obs: dict, mask: np.ndarray, samples: int = 40) -> list[tuple[int, float]]:
    """Ranks currently-legal action indices by how often the model (sampled
    stochastically) picks them, with its single deterministic best heavily
    weighted in so it always surfaces at rank 0. Uses only the public
    `.predict()` API (no reliance on policy-internals like raw logits)."""
    legal = np.flatnonzero(mask)
    if model is None:
        return [(int(i), 1.0 / len(legal)) for i in legal] if legal.size else []
    counts: dict[int, int] = {int(i): 0 for i in legal}  # every legal option ranked, even ones never sampled
    for _ in range(samples):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        counts[int(action)] = counts.get(int(action), 0) + 1
    best_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    counts[int(best_action)] = counts.get(int(best_action), 0) + samples
    total = sum(counts.values()) or 1
    return sorted(((idx, cnt / total) for idx, cnt in counts.items()), key=lambda kv: -kv[1])


@app.route("/api/advisor/deal_offer", methods=["POST"])
def api_advisor_deal_offer():
    body = request.get_json(force=True)
    hand_classes = body["hand"]
    num_players = int(body.get("num_players", 4))
    k = len(hand_classes)
    if not (3 <= k <= sasquatch_spaces.MAX_HAND):
        return jsonify({"error": f"hand must have 3 to {sasquatch_spaces.MAX_HAND} cards"}), 400
    point_tokens = (body.get("point_tokens") or [0] * num_players + [0] * num_players)[:num_players]
    model_spec = body.get("model", "random")

    try:
        model = get_model(model_spec)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"failed to load model '{model_spec}': {e}"}), 400

    obs = _blank_observation(num_players)
    obs["own_hand"] = _pad_classes(hand_classes, sasquatch_spaces.MAX_HAND)
    obs["own_hand_len"] = k
    obs["point_tokens"] = np.array(point_tokens, dtype=np.int32)
    obs["phase"] = sasquatch_spaces.phase_id("deal_offer_submit")

    combos = list(itertools.combinations(range(k), 3))
    mask = np.zeros(sasquatch_spaces.MAX_ACTIONS, dtype=np.int8)
    mask[: len(combos)] = 1
    obs["action_mask"] = mask

    ranked = _rank_actions(model, obs, mask)
    recommendations = [{"cards": [hand_classes[i] for i in combos[idx]], "score": round(score, 3)} for idx, score in ranked[:5] if idx < len(combos)]

    reveal_recommendation = None
    if recommendations:
        best_combo = combos[ranked[0][0]]
        remaining = [hand_classes[i] for i in range(k) if i not in best_combo]
        reveal_obs = _blank_observation(num_players)
        reveal_obs["own_hand"] = _pad_classes(remaining, sasquatch_spaces.MAX_HAND)
        reveal_obs["own_hand_len"] = len(remaining)
        reveal_obs["point_tokens"] = obs["point_tokens"]
        reveal_obs["phase"] = sasquatch_spaces.phase_id("deal_offer_reveal")
        deal_present = np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int8)
        deal_present[0] = 1
        reveal_obs["deal_present"] = deal_present
        deal_seller = np.full(sasquatch_spaces.MAX_DEALS, num_players, dtype=np.int64)
        deal_seller[0] = 0
        reveal_obs["deal_seller"] = deal_seller
        deal_hidden = np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int32)
        deal_hidden[0] = 3
        reveal_obs["deal_num_hidden"] = deal_hidden
        reveal_mask = np.zeros(sasquatch_spaces.MAX_ACTIONS, dtype=np.int8)
        reveal_mask[:3] = 1
        reveal_obs["action_mask"] = reveal_mask
        reveal_ranked = _rank_actions(model, reveal_obs, reveal_mask)
        best_combo_cards = [hand_classes[i] for i in best_combo]
        reveal_recommendation = best_combo_cards[reveal_ranked[0][0]] if reveal_ranked else None

    return jsonify({"recommendations": recommendations, "reveal_recommendation": reveal_recommendation, "disclaimer": ADVISOR_DISCLAIMER})


@app.route("/api/advisor/choose_deal", methods=["POST"])
def api_advisor_choose_deal():
    body = request.get_json(force=True)
    num_players = int(body["num_players"])
    sellers = body["sellers"]
    if not sellers:
        return jsonify({"error": "at least one seller's deal is required"}), 400
    point_tokens = (body.get("point_tokens") or [0] * num_players + [0] * num_players)[:num_players]
    model_spec = body.get("model", "random")

    try:
        model = get_model(model_spec)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"failed to load model '{model_spec}': {e}"}), 400

    obs = _blank_observation(num_players)
    obs["point_tokens"] = np.array(point_tokens, dtype=np.int32)
    obs["phase"] = sasquatch_spaces.phase_id("buyer_chooses_deal")

    deal_present = np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int8)
    deal_seller = np.full(sasquatch_spaces.MAX_DEALS, num_players, dtype=np.int64)
    deal_revealed = np.zeros((sasquatch_spaces.MAX_DEALS, sasquatch_spaces.MAX_DEAL_CARDS), dtype=np.int64)
    deal_hidden = np.zeros(sasquatch_spaces.MAX_DEALS, dtype=np.int32)
    for i, s in enumerate(sellers[: sasquatch_spaces.MAX_DEALS]):
        deal_present[i] = 1
        deal_seller[i] = i + 1  # "you" (the buyer) are seat 0; sellers fill the rest
        for j, c in enumerate((s.get("revealed") or [])[: sasquatch_spaces.MAX_DEAL_CARDS]):
            deal_revealed[i, j] = sasquatch_spaces.card_class_id(c)
        deal_hidden[i] = int(s.get("num_hidden", 0))
    obs["deal_present"] = deal_present
    obs["deal_seller"] = deal_seller
    obs["deal_revealed_cards"] = deal_revealed.reshape(-1)
    obs["deal_num_hidden"] = deal_hidden

    mask = np.zeros(sasquatch_spaces.MAX_ACTIONS, dtype=np.int8)
    mask[: len(sellers)] = 1
    obs["action_mask"] = mask

    ranked = _rank_actions(model, obs, mask)
    recommendations = [{"seller_slot": idx, "score": round(score, 3)} for idx, score in ranked if idx < len(sellers)]
    return jsonify({"recommendations": recommendations, "disclaimer": ADVISOR_DISCLAIMER})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
