"""Measure environment throughput, the number that bounds training speed.

Reports learner steps and full episodes per second per table size, with a
random-policy opponent, and (with `--engine`) the raw Rust-side rate for
comparison so it is obvious whether the Python wrapper or the engine is the
limit.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --seconds 5 --engine
    uv run python scripts/bench.py --workers 8      # aggregate across processes
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sell_me_a_sasquatch import _native as native  # noqa: E402
from sell_me_a_sasquatch import spaces as sasquatch_spaces  # noqa: E402
from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH, load_deck  # noqa: E402
from sell_me_a_sasquatch.selfplay_env import SasquatchSelfPlayEnv, random_masked_policy  # noqa: E402


def bench_env(players, seconds: float, deck) -> tuple[float, float]:
    """Steps and episodes per second for the self-play env against a random opponent."""
    env = SasquatchSelfPlayEnv(players=players, deck_config_path=deck, opponent_policy=random_masked_policy)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = episodes = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        action = int(rng.integers(max(1, env.legal_count)))
        _, _, terminated, _, _ = env.step(action)
        steps += 1
        if terminated:
            episodes += 1
            env.reset()
    elapsed = seconds
    return steps / elapsed, episodes / elapsed


def bench_engine(num_players: int, seconds: float, deck) -> float:
    """Rust-side micro-steps per second, including full observation and
    action-feature encoding but with no Python environment around it."""
    max_actions = deck.max_legal_actions(num_players)
    state = np.zeros(sasquatch_spaces.OBS_LEN, dtype=np.float32)
    actions = np.zeros((max_actions, sasquatch_spaces.ACTION_FEAT_LEN), dtype=np.float32)
    game = native.Game(num_players, deck, 1)
    steps = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        player = game.active_player()
        if player is None:
            game = native.Game(num_players, deck, steps)
            continue
        legal = game.encode(player, state, actions)
        game.step_index(player, steps % legal)
        steps += 1
    return steps / seconds


def _worker(args):
    """Multiprocessing entry point: unpacks args and runs `bench_env` in this process."""
    players, seconds, deck_path = args
    return bench_env(players, seconds, load_deck(deck_path))


def main(argv=None):
    """Parses arguments and prints throughput for every configured table size."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=3.0, help="Measurement window per configuration")
    parser.add_argument("--deck", type=str, default=DEFAULT_DECK_PATH)
    parser.add_argument("--engine", action="store_true", help="Also report the raw Rust-side rate")
    parser.add_argument("--workers", type=int, default=1, help="Run this many processes and sum their rates")
    args = parser.parse_args(argv)

    deck = load_deck(args.deck)
    print("observation layout")
    for line in sasquatch_spaces.describe_layout():
        print(f"  {line}")
    print("\naction space width per table size:")
    for n in sasquatch_spaces.ALL_PLAYER_COUNTS:
        print(f"  {n} players: {deck.max_legal_actions(n)}")

    configs: list = list(sasquatch_spaces.ALL_PLAYER_COUNTS) + [sasquatch_spaces.ALL_PLAYER_COUNTS]
    print(f"\nthroughput ({args.workers} worker(s), random opponents)")
    header = f"{'table':>8} {'steps/s':>12} {'episodes/s':>12}"
    print(header + (f" {'engine steps/s':>16}" if args.engine else ""))

    pool = None
    if args.workers > 1:
        import multiprocessing as mp

        pool = mp.Pool(args.workers)

    for players in configs:
        label = "mixed" if isinstance(players, tuple) and len(players) > 1 else str(players)
        if pool is None:
            steps, episodes = bench_env(players, args.seconds, deck)
        else:
            results = pool.map(_worker, [(players, args.seconds, args.deck)] * args.workers)
            steps = sum(r[0] for r in results)
            episodes = sum(r[1] for r in results)
        row = f"{label:>8} {steps:>12,.0f} {episodes:>12,.1f}"
        if args.engine and isinstance(players, int):
            row += f" {bench_engine(players, args.seconds, deck):>16,.0f}"
        print(row)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
