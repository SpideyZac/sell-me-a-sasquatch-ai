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


def test_card_classes_endpoint(client):
    resp = client.get("/api/card_classes")
    assert resp.status_code == 200
    classes = resp.get_json()["classes"]
    assert len(classes) == 12
    assert {"value": "Creature:Giant", "label": "Giant Creature"} in classes


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


def test_advisor_deal_offer_with_random_policy(client):
    resp = client.post(
        "/api/advisor/deal_offer",
        json={"hand": ["Creature:Giant", "Creature:Big", "Nasty:Trojan Horse", "Thingamabob:Platonic Isolator", "Creature:Tiny"], "num_players": 4, "model": "random"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["recommendations"]) == 5  # C(5,3) == 10 combos total, top-5 returned
    for r in body["recommendations"]:
        assert len(r["cards"]) == 3


def test_advisor_deal_offer_rejects_too_short_hand(client):
    resp = client.post("/api/advisor/deal_offer", json={"hand": ["Creature:Giant"], "num_players": 4, "model": "random"})
    assert resp.status_code == 400


def test_advisor_choose_deal_with_random_policy(client):
    resp = client.post(
        "/api/advisor/choose_deal",
        json={
            "num_players": 4,
            "sellers": [{"revealed": ["Creature:Giant"], "num_hidden": 2}, {"revealed": [], "num_hidden": 3}, {"revealed": ["Nasty:Trojan Horse"], "num_hidden": 2}],
            "model": "random",
        },
    )
    assert resp.status_code == 200
    recs = resp.get_json()["recommendations"]
    assert len(recs) == 3
    assert {r["seller_slot"] for r in recs} == {0, 1, 2}
