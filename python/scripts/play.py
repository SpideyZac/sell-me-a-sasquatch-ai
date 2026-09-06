"""Evaluate a trained Sell Me a Sasquatch model's win rate against a
random-policy baseline (or against a frozen copy of itself).

Usage:
    uv run python scripts/play.py models/sasquatch_ppo.zip --episodes 200
"""

from __future__ import annotations

import argparse

from sb3_contrib import MaskablePPO

from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH
from sell_me_a_sasquatch.selfplay_env import SasquatchSelfPlayEnv, random_masked_policy


def make_model_policy(model):
    def _policy(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return _policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("--num-players", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--deck", type=str, default=DEFAULT_DECK_PATH)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--opponent", choices=["random", "self"], default="random")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model_path)
    opponent = random_masked_policy if args.opponent == "random" else make_model_policy(model)

    env = SasquatchSelfPlayEnv(num_players=args.num_players, deck_config_path=args.deck, opponent_policy=opponent, learner_seat="random")

    wins = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        terminated = False
        reward = 0.0
        while not terminated:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, _, info = env.step(int(action))
        if reward > 0:
            wins += 1

    win_rate = wins / args.episodes
    chance = 1.0 / args.num_players
    print(f"Win rate over {args.episodes} episodes vs '{args.opponent}' opponents: {win_rate:.1%}")
    print(f"(random-chance baseline for {args.num_players} players: {chance:.1%})")


if __name__ == "__main__":
    main()
