"""A very simple local web app for Sell Me a Sasquatch (three modes):

- Watch: a game of loaded AI models (or random policies) playing each other,
  stepped one micro-turn at a time.
- Play: you take one seat, AI models (or random) play the rest.
- Advisor: same as Play (a real `GameState` session, so hands, collections,
  deals, and draw/discard pile state are all tracked exactly), except your
  own legal actions are additionally ranked by a model so you can see what
  it would recommend before you pick.

Run with:
    cd python
    uv pip install -e ".[web,train]"   # train extra optional - only needed to load real models
    uv run python webapp/app.py
Then open http://127.0.0.1:5000/
"""

from __future__ import annotations

import glob
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


# game sessions (Watch / Play / Advisor)
class GameSession:
    def __init__(
        self,
        game,
        num_players: int,
        mode: str,
        human_seat: "int | None",
        seat_specs: list[str],
        seat_policies: list,
        advisor_model_spec: "str | None" = None,
    ):
        self.game = game
        self.num_players = num_players
        self.mode = mode  # "watch" | "play" | "advisor"
        self.human_seat = human_seat
        self.seat_specs = seat_specs
        self.seat_policies = seat_policies
        self.advisor_model_spec = advisor_model_spec  # "advisor" mode only
        self.log: list[str] = []


def _display_name(game, card_id: int) -> str:
    """Card label for UI/log purposes: real names for Nasties/Thingamabobs,
    but tier-only for Creatures (their flavor names are non-mechanical
    fluff - see engine/src/card.rs - and just noise here)."""
    kind = game.card_kind(card_id)
    if kind and kind.startswith("Creature:"):
        return f"{kind.split(':', 1)[1]} Creature"
    return game.card_name(card_id) or f"card#{card_id}"


def _card_dict(game, card_id: int) -> dict:
    return {"id": card_id, "name": _display_name(game, card_id), "kind": game.card_kind(card_id)}


def describe_action(game, action) -> str:
    d = action.to_dict()
    t = d["type"]

    def label(cid):
        return _display_name(game, cid)

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
        if session.mode in ("play", "advisor") and active == session.human_seat:
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
        "draw_pile_len": obs0.draw_pile_len,
        "discard_pile_len": obs0.discard_pile_len,
        "log": session.log[-40:],
    }


def human_view(session: GameSession) -> dict:
    game = session.game
    seat = session.human_seat
    obs = game.observation(seat)
    your_turn = (not game.is_game_over()) and game.active_players() == [seat]
    legal = game.legal_actions(seat) if your_turn else []
    legal_actions = [{"index": i, "label": describe_action(game, a)} for i, a in enumerate(legal)]

    if session.mode == "advisor" and your_turn and legal:
        model = get_model(session.advisor_model_spec)
        obs_vec = sasquatch_spaces.vectorize_observation(game, obs, len(legal), session.num_players)
        scores = {idx: score for idx, score in _rank_actions(model, obs_vec, obs_vec["action_mask"])}
        for a in legal_actions:
            a["score"] = round(scores.get(a["index"], 0.0), 3)
        legal_actions.sort(key=lambda a: -a["score"])
        legal_actions[0]["recommended"] = True

    return {
        "mode": session.mode,
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
        "draw_pile_len": obs.draw_pile_len,
        "discard_pile_len": obs.discard_pile_len,
        "legal_actions": legal_actions,
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


# routes: Watch / Play / Advisor game sessions
@app.route("/api/games", methods=["POST"])
def api_new_game():
    body = request.get_json(force=True)
    num_players = int(body["num_players"])
    if not (2 <= num_players <= 6):
        return jsonify({"error": "num_players must be 2-6"}), 400
    mode = body.get("mode", "watch")
    if mode not in ("watch", "play", "advisor"):
        return jsonify({"error": "mode must be 'watch', 'play', or 'advisor'"}), 400
    seated = mode in ("play", "advisor")  # both put a human in one seat, AIs in the rest
    deck = body.get("deck") or DEFAULT_DECK_PATH
    seed = int(body["seed"]) if body.get("seed") not in (None, "") else random.randint(0, 2**63 - 1)

    seat_specs = body.get("seat_models") or ["random"] * num_players
    seat_specs = (seat_specs + ["random"] * num_players)[:num_players]
    human_seat = int(body["human_seat"]) if seated and body.get("human_seat") is not None else (0 if seated else None)
    advisor_model_spec = body.get("advisor_model", "random") if mode == "advisor" else None

    try:
        game = native.Game(num_players, deck, seed)
    except Exception as e:  # noqa: BLE001 - surface engine setup errors to the UI as-is
        return jsonify({"error": str(e)}), 400

    seat_policies = []
    for i in range(num_players):
        if seated and i == human_seat:
            seat_policies.append(None)
        else:
            try:
                seat_policies.append(get_policy(seat_specs[i]))
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": f"failed to load model '{seat_specs[i]}': {e}"}), 400

    if advisor_model_spec is not None:
        try:
            get_model(advisor_model_spec)  # validate/preload before the session is created
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"failed to load advisor model '{advisor_model_spec}': {e}"}), 400

    game_id = uuid.uuid4().hex[:12]
    session = GameSession(game, num_players, mode, human_seat, seat_specs, seat_policies, advisor_model_spec)
    GAMES[game_id] = session

    if seated:
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
    if session.mode not in ("play", "advisor"):
        return jsonify({"error": "act is only for play/advisor-mode games"}), 400
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


# advisor scoring (used by human_view when session.mode == "advisor")
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
