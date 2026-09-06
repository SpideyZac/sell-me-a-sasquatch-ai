"""Smoke tests for the local web app (Watch / Play / Advisor), using Flask's
test client (no real server/port needed). Only exercises the "random" policy
path - model-backed paths need the optional `[train]` extra plus an actual
trained `.zip` on disk, neither of which a fresh checkout has.
"""

import importlib.util
import os
import sys

import pytest

pytest.importorskip("flask")


def _load_app_module():
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "webapp", "app.py"))
    spec = importlib.util.spec_from_file_location("sasquatch_webapp_app", path)
    module = importlib.util.module_from_spec(spec)
    # Flask's get_root_path() (used to locate templates/ and static/) looks
    # the module up in sys.modules by name - module_from_spec alone doesn't
    # register it there, only importlib's higher-level import machinery does.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def client():
    module = _load_app_module()
    module.app.testing = True
    module.GAMES.clear()
    with module.app.test_client() as c:
        yield c


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Sell Me a Sasquatch" in resp.data


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_watch_mode_full_game_via_random_policies(client, num_players):
    resp = client.post("/api/games", json={"mode": "watch", "num_players": num_players, "seat_models": ["random"] * num_players, "seed": 11})
    assert resp.status_code == 200
    game_id = resp.get_json()["game_id"]

    state = None
    for _ in range(3000):
        state = client.post(f"/api/games/{game_id}/advance").get_json()["state"]
        if state["is_game_over"]:
            break
    assert state["is_game_over"], "watch-mode game did not finish in time"
    assert 0 <= state["winner"] < num_players
    assert len(state["hands"]) == num_players


def test_play_mode_full_game_via_random_policies(client):
    resp = client.post("/api/games", json={"mode": "play", "num_players": 4, "human_seat": 2, "seat_models": ["random"] * 4, "seed": 22})
    assert resp.status_code == 200
    state = resp.get_json()["state"]
    assert state["your_seat"] == 2

    for _ in range(300):
        if state["is_game_over"]:
            break
        assert state["your_turn"], "human should always be the one needing to act when control returns"
        act_resp = client.post(f"/api/games/{resp.get_json()['game_id']}/act", json={"action_index": 0})
        assert act_resp.status_code == 200
        state = act_resp.get_json()["state"]

    assert state["is_game_over"]
    assert 0 <= state["winner"] < 4


def test_play_mode_rejects_out_of_range_action(client):
    resp = client.post("/api/games", json={"mode": "play", "num_players": 4, "human_seat": 0, "seat_models": ["random"] * 4, "seed": 5})
    game_id = resp.get_json()["game_id"]
    bad = client.post(f"/api/games/{game_id}/act", json={"action_index": 99999})
    assert bad.status_code == 400


def test_advisor_mode_full_game_via_random_policy(client):
    """Advisor mode is a real game session (like Play), plus every legal
    action the human sees is annotated with a model-ranked score and the
    top one is flagged 'recommended'."""
    resp = client.post(
        "/api/games",
        json={"mode": "advisor", "num_players": 4, "human_seat": 2, "seat_models": ["random"] * 4, "advisor_model": "random", "seed": 22},
    )
    assert resp.status_code == 200
    game_id = resp.get_json()["game_id"]
    state = resp.get_json()["state"]
    assert state["your_seat"] == 2
    assert state["mode"] == "advisor"

    for _ in range(300):
        if state["is_game_over"]:
            break
        assert state["your_turn"], "human should always be the one needing to act when control returns"
        assert state["legal_actions"], "advisor turn should always offer at least one legal action"
        assert all("score" in a for a in state["legal_actions"])
        assert state["legal_actions"][0].get("recommended") is True
        assert isinstance(state["draw_pile_len"], int)
        assert isinstance(state["discard_pile_len"], int)
        act_resp = client.post(f"/api/games/{game_id}/act", json={"action_index": 0})
        assert act_resp.status_code == 200
        state = act_resp.get_json()["state"]

    assert state["is_game_over"]
    assert 0 <= state["winner"] < 4


def test_creature_cards_display_tier_only_name(client):
    resp = client.post("/api/games", json={"mode": "watch", "num_players": 4, "seat_models": ["random"] * 4, "seed": 1})
    state = resp.get_json()["state"]
    creature_names = {c["name"] for hand in state["hands"] for c in hand if c["kind"] and c["kind"].startswith("Creature:")}
    for name in creature_names:
        assert name.endswith(" Creature")
        assert name.split(" Creature")[0] in ("Giant", "Big", "Medium", "Tiny")
