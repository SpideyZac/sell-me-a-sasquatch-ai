"""Smoke tests for the local web app (watch, play, and live modes), using
Flask's test client (no real server or port needed). Only exercises the
"random" policy path; model-backed paths need the optional `[train]`
extra plus an actual trained `.zip` on disk, neither of which a fresh
checkout has.
"""

import importlib.util
import os
import random
import sys

import pytest  # type: ignore

pytest.importorskip("flask")

ALL_CLASSES = [
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


def _load_app_module():
    """Loads `webapp/app.py` as a module, since it isn't part of an installed package."""
    webapp_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "webapp")
    )
    path = os.path.join(webapp_dir, "app.py")
    # app.py does bare "import card_display" / "live_game" / "models" (its
    # sibling modules), fine when run normally ("python webapp/app.py"
    # puts its own directory on sys.path[0] automatically), but
    # spec_from_file_location doesn't, so those imports need a hand here
    if webapp_dir not in sys.path:
        sys.path.insert(0, webapp_dir)
    spec = importlib.util.spec_from_file_location("sasquatch_webapp_app", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    # flask's get_root_path() (used to locate templates/ and static/) looks
    # the module up in sys.modules by name; module_from_spec alone doesn't
    # register it there, only importlib's higher-level import machinery does
    sys.modules[spec.name] = module  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


@pytest.fixture()
def client():
    """Flask test client fixture, with a fresh app and cleared GAMES dict."""
    module = _load_app_module()
    module.app.testing = True
    module.GAMES.clear()
    with module.app.test_client() as c:
        yield c


def test_index_page_loads(client):  # pylint: disable=redefined-outer-name
    """Smoke test that the index page loads and contains the expected title."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Sell Me a Sasquatch" in resp.data


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_watch_mode_full_game_via_random_policies(
    client, num_players
):  # pylint: disable=redefined-outer-name
    """Smoke test that a watch-mode game can be created and driven to completion
    using only the random policy, and that the final state is consistent."""
    resp = client.post(
        "/api/games",
        json={
            "mode": "watch",
            "num_players": num_players,
            "seat_models": ["random"] * num_players,
            "seed": 11,
        },
    )
    assert resp.status_code == 200
    game_id = resp.get_json()["game_id"]

    state = None
    for _ in range(3000):
        state = client.post(f"/api/games/{game_id}/advance").get_json()["state"]
        if state["is_game_over"]:
            break
    assert state["is_game_over"], "watch-mode game did not finish in time"  # type: ignore
    assert 0 <= state["winner"] < num_players  # type: ignore
    assert len(state["hands"]) == num_players  # type: ignore


def test_play_mode_full_game_via_random_policies(
    client,
):  # pylint: disable=redefined-outer-name
    """Smoke test that a play-mode game can be created and driven to completion
    using only the random policy, and that the final state is consistent."""
    resp = client.post(
        "/api/games",
        json={
            "mode": "play",
            "num_players": 4,
            "human_seat": 2,
            "seat_models": ["random"] * 4,
            "seed": 22,
        },
    )
    assert resp.status_code == 200
    state = resp.get_json()["state"]
    assert state["your_seat"] == 2

    for _ in range(300):
        if state["is_game_over"]:
            break
        assert state[
            "your_turn"
        ], "human should always be the one needing to act when control returns"
        act_resp = client.post(
            f"/api/games/{resp.get_json()['game_id']}/act", json={"action_index": 0}
        )
        assert act_resp.status_code == 200
        state = act_resp.get_json()["state"]

    assert state["is_game_over"]
    assert 0 <= state["winner"] < 4


def test_play_mode_rejects_out_of_range_action(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: play-mode game should reject an action index that is
    out of range for the current prompt's options, returning a 400 error."""
    resp = client.post(
        "/api/games",
        json={
            "mode": "play",
            "num_players": 4,
            "human_seat": 0,
            "seat_models": ["random"] * 4,
            "seed": 5,
        },
    )
    game_id = resp.get_json()["game_id"]
    bad = client.post(f"/api/games/{game_id}/act", json={"action_index": 99999})
    assert bad.status_code == 400


def _pick_kind(supply: dict) -> str:
    """Returns a uniformly random kind from the supply dict, which maps
    kind string to remaining count. Raises if no kinds are available."""
    available = [c for c in ALL_CLASSES if supply.get(c, 0) > 0]
    return random.choice(available)


def _play_live_game_randomly(
    client, live_id, max_steps=8000
):  # pylint: disable=redefined-outer-name
    """Drives a full live-tracker game by answering every prompt with a
    uniformly random (but supply-respecting) choice, asserting the flow
    never errors and eventually reaches game_over."""
    state = client.get(f"/api/live/{live_id}").get_json()["state"]
    for _ in range(max_steps):
        if state["is_game_over"]:
            return state
        prompt = state["prompt"]
        supply = {k["value"]: k["remaining"] for k in state["kind_options"]}
        if prompt["type"] in ("pin_hand", "pin_resolution"):
            local_supply = dict(supply)
            kinds = []
            for _ in range(prompt["count"]):
                k = _pick_kind(local_supply)
                kinds.append(k)
                local_supply[k] -= 1
            resp = client.post(f"/api/live/{live_id}/respond", json={"kinds": kinds})
        elif prompt["type"] == "reveal_kind":
            resp = client.post(
                f"/api/live/{live_id}/respond", json={"kind": _pick_kind(supply)}
            )
        elif prompt["type"] == "choose_action":
            resp = client.post(
                f"/api/live/{live_id}/respond",
                json={"index": random.randrange(len(prompt["options"]))},
            )
        else:
            raise AssertionError(f"unexpected prompt type: {prompt}")
        assert resp.status_code == 200, resp.get_json()
        state = resp.get_json()["state"]
    raise AssertionError(f"live game did not finish within {max_steps} steps")


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_live_game_full_playthrough(
    client, num_players
):  # pylint: disable=redefined-outer-name
    """Smoke test that a live-tracker game can be created and driven to completion
    using only the random policy, and that the final state is consistent."""
    resp = client.post(
        "/api/live",
        json={
            "num_players": num_players,
            "human_seat": 1 % num_players,
            "seed": 99,
            "advisor_model": "random",
        },
    )
    assert resp.status_code == 200
    live_id = resp.get_json()["live_id"]
    state = _play_live_game_randomly(client, live_id)
    assert 0 <= state["winner"] < num_players
    assert len(state["point_tokens"]) == num_players
    assert len(state["hand"]) == 5, "your hand should always be fully known"
    for card in state["hand"]:
        assert card["name"] is not None


def test_live_game_starts_with_pin_hand_prompt(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: the first prompt of a live-tracker game should be
    a pin_hand prompt, not a choose_action prompt (which would be the case if
    the game were already in the middle of a turn)."""
    resp = client.post(
        "/api/live",
        json={"num_players": 4, "human_seat": 0, "seed": 5, "advisor_model": "random"},
    )
    state = resp.get_json()["state"]
    assert state["prompt"]["type"] == "pin_hand"
    assert state["prompt"]["context"] == "starting_hand"
    assert state["prompt"]["count"] == 5


def test_live_game_pinned_hand_names_match_what_you_entered(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: `pin_kind` overwrites a card's kind, but each card
    also carries a separately-set flavor `name` from deck-shuffle time (see
    `deck.rs`); a nasty or thingamabob pinned to a different kind must not
    keep displaying its old, now-wrong name (e.g. showing "Trojan Horse"
    for a card you entered as some other nasty or thingamabob)."""
    resp = client.post(
        "/api/live",
        json={"num_players": 4, "human_seat": 0, "seed": 9, "advisor_model": "random"},
    )
    live_id = resp.get_json()["live_id"]
    # a deliberately card-varied starting hand so a stale name would be
    # observable regardless of what this seed's engine happened to deal
    entered_kinds = [
        "Nasty:Trojan Horse",
        "Thingamabob:Platonic Isolator",
        "Creature:Medium",
        "Creature:Tiny",
        "Nasty:Loan Shark",
    ]
    expected_names = [
        "Trojan Horse",
        "Platonic Isolator",
        "Medium Creature",
        "Tiny Creature",
        "Loan Shark",
    ]
    resp = client.post(f"/api/live/{live_id}/respond", json={"kinds": entered_kinds})
    state = resp.get_json()["state"]
    assert [c["kind"] for c in state["hand"]] == entered_kinds
    assert [c["name"] for c in state["hand"]] == expected_names


def test_live_game_opponent_turns_hide_unrevealed_kinds(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: when it's an opponent's turn, the prompt options
    must never leak a real card name before it's actually revealed, even if
    the turn leader is you (the human) and the opponent's unknown combo is
    otherwise fully known to the server."""
    resp = client.post(
        "/api/live",
        json={"num_players": 4, "human_seat": 0, "seed": 5, "advisor_model": "random"},
    )
    live_id = resp.get_json()["live_id"]
    state = resp.get_json()["state"]
    kinds = [c["kind"] for c in state["hand"]]
    resp = client.post(f"/api/live/{live_id}/respond", json={"kinds": kinds})
    state = resp.get_json()["state"]
    # it's now an opponent's deal-offer submit turn (turn leader is you
    # only if you happen to be it, either way the other seats' unknown
    # combos must never leak a real card name before it's actually revealed)
    prompt = state["prompt"]
    if prompt["type"] == "choose_action" and not prompt["your_turn"]:
        assert all(
            "???" in o["label"] or "player_" in o["label"] for o in prompt["options"]
        )
        assert not any("score" in o for o in prompt["options"])


def test_live_game_advisor_scoring_only_on_your_own_turn(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: the advisor model should only score options on your
    own turn, not on an opponent's turn (even if the turn leader is you)."""
    resp = client.post(
        "/api/live",
        json={"num_players": 4, "human_seat": 0, "seed": 5, "advisor_model": "random"},
    )
    live_id = resp.get_json()["live_id"]
    state = resp.get_json()["state"]

    saw_your_turn = False
    saw_opponent_turn = False
    for _ in range(200):
        if state["is_game_over"] or (saw_your_turn and saw_opponent_turn):
            break
        prompt = state["prompt"]
        supply = {k["value"]: k["remaining"] for k in state["kind_options"]}
        if prompt["type"] == "choose_action":
            if prompt["your_turn"]:
                saw_your_turn = True
                assert all("score" in o for o in prompt["options"])
                assert any(o.get("recommended") for o in prompt["options"])
            else:
                saw_opponent_turn = True
                assert not any("score" in o for o in prompt["options"])
            resp = client.post(f"/api/live/{live_id}/respond", json={"index": 0})
        elif prompt["type"] in ("pin_hand", "pin_resolution"):
            local_supply = dict(supply)
            kinds = []
            for _ in range(prompt["count"]):
                k = _pick_kind(local_supply)
                kinds.append(k)
                local_supply[k] -= 1
            resp = client.post(f"/api/live/{live_id}/respond", json={"kinds": kinds})
        elif prompt["type"] == "reveal_kind":
            resp = client.post(
                f"/api/live/{live_id}/respond", json={"kind": _pick_kind(supply)}
            )
        else:
            break
        state = resp.get_json()["state"]
    assert (
        saw_your_turn
    ), "should reach the human's own deal-offer turn within a few steps"
    assert saw_opponent_turn, "should also see at least one opponent's deal-offer turn"


def test_live_game_rejects_wrong_prompt_payload(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: if the current prompt is pin_hand,
    a choose_action payload should be rejected with a 400 error, and vice versa."""
    resp = client.post(
        "/api/live",
        json={"num_players": 4, "human_seat": 0, "seed": 5, "advisor_model": "random"},
    )
    live_id = resp.get_json()["live_id"]
    # current prompt is pin_hand, not choose_action
    bad = client.post(f"/api/live/{live_id}/respond", json={"index": 0})
    assert bad.status_code == 400
    assert "error" in bad.get_json()


def test_live_game_rejects_bad_num_players(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: live-tracker game should reject a
    request with an invalid number of players, returning a 400 error."""
    resp = client.post("/api/live", json={"num_players": 1, "human_seat": 0, "seed": 1})
    assert resp.status_code == 400


def test_creature_cards_display_tier_only_name(
    client,
):  # pylint: disable=redefined-outer-name
    """Regression test: creature cards should display only their tier in the name,
    not their full kind (e.g. "Giant Creature" instead of "Creature:Giant")."""
    resp = client.post(
        "/api/games",
        json={
            "mode": "watch",
            "num_players": 4,
            "seat_models": ["random"] * 4,
            "seed": 1,
        },
    )
    state = resp.get_json()["state"]
    creature_names = {
        c["name"]
        for hand in state["hands"]
        for c in hand
        if c["kind"] and c["kind"].startswith("Creature:")
    }
    for name in creature_names:
        assert name.endswith(" Creature")
        assert name.split(" Creature")[0] in ("Giant", "Big", "Medium", "Tiny")
