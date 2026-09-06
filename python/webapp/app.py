"""A very simple local web app for Sell Me a Sasquatch (three modes):

- Watch: a game of loaded AI models (or random policies) playing each other,
  stepped one micro-turn at a time.
- Play: you take one seat, AI models (or random) play the rest.
- Live (the "Advisor"): for an actual live (physical) game. It's still the
  real Rust rules engine underneath - so collections, point totals, discard/
  draw piles, set trade-ins, and turn order are all handled automatically
  and correctly - but nothing is randomly dealt. You tell it what's really
  on the table exactly when it becomes visible (your starting hand, each
  card as it's revealed, what a resolved deal's still-hidden cards turn out
  to be), via `Game.pin_kind` (see `engine/src/game.rs`); everything else
  the engine can compute on its own without you tracking a thing. See
  `live_game.py` for the state machine that drives this.

Run with:
    cd python
    uv pip install -e ".[web,train]"   # train extra optional - only needed to load real models
    uv run python webapp/app.py
Then open http://127.0.0.1:5000/
"""

from __future__ import annotations

import random
import uuid

from flask import Flask, abort, jsonify, render_template, request

from sell_me_a_sasquatch import _native as native
from sell_me_a_sasquatch import spaces as sasquatch_spaces
from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH

from card_display import card_dict, describe_action
from live_game import LiveGameError, LiveSession
from models import get_policy, list_available_models

app = Flask(__name__)
GAMES: dict[str, "GameSession"] = {}
LIVE_GAMES: dict[str, LiveSession] = {}


# game sessions (Watch / Play - real `GameState`, AI/random-controlled seats)
class GameSession:
    def __init__(self, game, num_players: int, mode: str, human_seat: "int | None", seat_specs: list[str], seat_policies: list):
        self.game = game
        self.num_players = num_players
        # Width for random-policy seats; model seats use their own (see
        # `models.observation_width`).
        self.max_actions = game.max_legal_actions()
        self.mode = mode  # "watch" | "play"
        self.human_seat = human_seat
        self.seat_specs = seat_specs
        self.seat_policies = seat_policies
        self.log: list[str] = []


def _step_seat(session: GameSession, player: int) -> None:
    game = session.game
    legal = game.legal_actions(player)
    if not legal:
        return
    policy = session.seat_policies[player]
    width = getattr(policy, "max_actions", None) or session.max_actions
    obs, mask, legal_count = sasquatch_spaces.encode_for_player(game, player, width)
    idx = policy(obs, mask, legal_count)
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
        active = game.active_player()
        if session.mode == "play" and active == session.human_seat:
            return
        _step_seat(session, active)


def spectator_view(session: GameSession) -> dict:
    game = session.game
    n = session.num_players
    hands, collections, tokens = [], [], []
    for p in range(n):
        obs = game.observation(p)
        hands.append([card_dict(game, c) for c in obs.own_hand])
        collections.append([card_dict(game, c) for c in obs.collections[p]])
        tokens.append(obs.point_tokens[p])
    obs0 = game.observation(0)
    deals = [{"seller": d.seller, "revealed": [card_dict(game, c) for c in d.revealed_cards], "num_hidden": d.num_hidden} for d in obs0.deals]
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
        "hand": [card_dict(game, c) for c in obs.own_hand],
        "point_tokens": list(obs.point_tokens),
        "collections": [[card_dict(game, c) for c in coll] for coll in obs.collections],
        "deals": [{"seller": d.seller, "revealed": [card_dict(game, c) for c in d.revealed_cards], "num_hidden": d.num_hidden} for d in obs.deals],
        "draw_pile_len": obs.draw_pile_len,
        "discard_pile_len": obs.discard_pile_len,
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


def _get_live_session(live_id: str) -> LiveSession:
    session = LIVE_GAMES.get(live_id)
    if session is None:
        abort(404, "unknown live_id")
    return session


# routes: pages
@app.route("/")
def index():
    return render_template("index.html")


# routes: shared JSON API
@app.route("/api/models")
def api_models():
    return jsonify({"models": list_available_models()})


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
    first_player = int(body["first_player"]) if body.get("first_player") not in (None, "") else None
    if first_player is not None and not (0 <= first_player < num_players):
        return jsonify({"error": "first_player out of range"}), 400

    seat_specs = body.get("seat_models") or ["random"] * num_players
    seat_specs = (seat_specs + ["random"] * num_players)[:num_players]
    human_seat = int(body["human_seat"]) if mode == "play" and body.get("human_seat") is not None else (0 if mode == "play" else None)

    try:
        game = native.Game(num_players, deck, seed, first_player)
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
        _step_seat(session, session.game.active_player())
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


# routes: Live tracker (the "Advisor" tab) - see live_game.py
@app.route("/api/live", methods=["POST"])
def api_new_live_game():
    body = request.get_json(force=True)
    num_players = int(body["num_players"])
    if not (2 <= num_players <= 6):
        return jsonify({"error": "num_players must be 2-6"}), 400
    human_seat = int(body.get("human_seat", 0))
    if not (0 <= human_seat < num_players):
        return jsonify({"error": "human_seat out of range"}), 400
    deck = body.get("deck") or DEFAULT_DECK_PATH
    seed = int(body["seed"]) if body.get("seed") not in (None, "") else random.randint(0, 2**63 - 1)
    advisor_model = body.get("advisor_model", "random")
    first_player = int(body["first_player"]) if body.get("first_player") not in (None, "") else None
    if first_player is not None and not (0 <= first_player < num_players):
        return jsonify({"error": "first_player out of range"}), 400

    try:
        session = LiveSession(num_players, human_seat, deck, seed, advisor_model, first_player)
    except LiveGameError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - surface engine/model setup errors to the UI as-is
        return jsonify({"error": str(e)}), 400

    live_id = uuid.uuid4().hex[:12]
    LIVE_GAMES[live_id] = session
    return jsonify({"live_id": live_id, "state": session.state()})


@app.route("/api/live/<live_id>")
def api_get_live_game(live_id):
    return jsonify({"state": _get_live_session(live_id).state()})


@app.route("/api/live/<live_id>/respond", methods=["POST"])
def api_live_respond(live_id):
    session = _get_live_session(live_id)
    body = request.get_json(force=True)
    try:
        session.respond(body)
    except LiveGameError as e:
        return jsonify({"error": str(e), "state": session.state()}), 400
    return jsonify({"state": session.state()})


@app.route("/api/live/<live_id>", methods=["DELETE"])
def api_delete_live_game(live_id):
    LIVE_GAMES.pop(live_id, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
