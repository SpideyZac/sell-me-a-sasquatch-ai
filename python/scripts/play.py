"""Evaluate a trained model's win rate, per table size.

The interesting number is the win rate *relative to chance*: a 4-player
table gives a random agent 25%, a 6-player table 16.7%, so raw win rates
across table sizes are not comparable on their own.

Usage:
    uv run python scripts/play.py models/sasquatch_ppo.zip --episodes 300
    uv run python scripts/play.py models/sasquatch_ppo.zip --players 4 --opponent self
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH, load_deck  # noqa: E402
from sell_me_a_sasquatch.policy import NumpyPointerPolicy  # noqa: E402
from sell_me_a_sasquatch.selfplay_env import SasquatchSelfPlayEnv, random_masked_policy  # noqa: E402


def as_opponent(model, rng: np.random.Generator):
    """Frozen copy of `model` in the `(obs, mask, legal_count) -> index` shape.

    Prefers the numpy mirror, which is the same arithmetic without torch's
    per-call overhead - meaningful here because opponents take most of the
    moves in an episode.
    """
    try:
        return NumpyPointerPolicy.from_model(model, rng=rng)
    except TypeError:

        def _torch_opponent(obs, mask, legal_count):
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            return int(action)

        return _torch_opponent


def evaluate(model, num_players: int, episodes: int, deck, opponent, seed: int) -> tuple[float, float]:
    # The action space must match the one the checkpoint was trained
    # with, not the (possibly narrower) one this table size needs on its
    # own - a model trained across every table size has a wider head.
    env = SasquatchSelfPlayEnv(
        players=num_players,
        deck_config_path=deck,
        opponent_policy=opponent,
        learner_seat="random",
        max_actions=int(model.action_space.n),
    )
    wins = 0
    started = time.perf_counter()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        info: dict = {}
        while not terminated:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, _, terminated, _, info = env.step(int(action))
        wins += int(info.get("winner") == info.get("learner_seat"))
    return wins / episodes, episodes / (time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6], choices=range(2, 7))
    parser.add_argument("--deck", type=str, default=DEFAULT_DECK_PATH)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--opponent", choices=["random", "self"], default="random")
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args(argv)

    deck = load_deck(args.deck)
    model = MaskablePPO.load(args.model_path, device="cpu")
    rng = np.random.default_rng(args.seed)
    opponent = random_masked_policy if args.opponent == "random" else as_opponent(model, rng)

    print(f"{args.model_path} vs '{args.opponent}' opponents, {args.episodes} episodes per table size\n")
    print(f"{'players':>8} {'win rate':>10} {'chance':>8} {'vs chance':>10} {'episodes/s':>11}")
    for num_players in sorted(set(args.players)):
        win_rate, rate = evaluate(model, num_players, args.episodes, deck, opponent, args.seed)
        chance = 1.0 / num_players
        print(f"{num_players:>8} {win_rate:>9.1%} {chance:>7.1%} {win_rate / chance:>9.2f}x {rate:>10.1f}")


if __name__ == "__main__":
    main()
