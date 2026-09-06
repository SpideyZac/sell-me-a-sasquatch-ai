"""Train a Sell Me a Sasquatch agent via self-play.

The easiest working recipe against our current env: sb3-contrib's
`MaskablePPO` with a single shared policy playing every seat through
`SasquatchSelfPlayEnv`. The "opponent" seats are played by `OpponentPool`
(`selfplay_env.py`): a mix of the live in-training model and a rotating
pool of its own older, frozen snapshots - periodically refreshed by
`SnapshotCallback` below - so the learner faces a bit of a curriculum
instead of only ever a mirror of its exact current self.

Usage:
    uv run python scripts/train.py --num-players 4 --timesteps 300000

    # continue training an existing checkpoint for 100k more timesteps:
    uv run python scripts/train.py --resume models/sasquatch_ppo.zip --timesteps 100000 --out models/sasquatch_ppo_v2

Self-play needs the opponent callback to call back into the live model (and
occasionally a loaded snapshot), so all parallel env copies run in a single
process (DummyVecEnv) rather than subprocesses.
"""

from __future__ import annotations

import argparse
import glob
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH
from sell_me_a_sasquatch.selfplay_env import OpponentPool, SasquatchSelfPlayEnv, random_masked_policy


class SnapshotCallback(BaseCallback):
    """Every `every_n_steps`, saves the live model and loads it back as a
    frozen (never-trained-further) snapshot into `pool`, keeping the
    opponent pool's "older models" up to date as training progresses."""

    def __init__(self, pool: OpponentPool, save_dir: str, every_n_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.pool = pool
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self._last_snapshot_at = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot_at >= self.every_n_steps:
            path = os.path.join(self.save_dir, f"snapshot_{self.num_timesteps}.zip")
            self.model.save(path)
            snapshot = MaskablePPO.load(path, device="cpu")
            self.pool.add_snapshot(snapshot)
            self._last_snapshot_at = self.num_timesteps
            if self.verbose:
                print(f"[snapshot] added opponent snapshot at {self.num_timesteps} timesteps (pool size {len(self.pool.snapshots)})")
        return True


def make_env_fn(num_players: int, deck_path: str, opponent):
    def _init():
        env = SasquatchSelfPlayEnv(num_players=num_players, deck_config_path=deck_path, opponent_policy=opponent)
        env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-players", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--deck", type=str, default=DEFAULT_DECK_PATH)
    parser.add_argument("--out", type=str, default="models/sasquatch_ppo")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opponent",
        choices=["self", "random"],
        default="self",
        help="'self' = self-play against a mix of the live model and older snapshots (default); "
        "'random' = train only against a random baseline (no opponent pool, much weaker final policy)",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=20_000,
        help="Timesteps between adding a new frozen snapshot to the opponent pool (only used with --opponent self)",
    )
    parser.add_argument(
        "--current-prob",
        type=float,
        default=0.5,
        help="Probability each episode's opponent is the live model rather than a random older snapshot, once any exist",
    )
    parser.add_argument("--max-snapshots", type=int, default=10, help="Max older snapshots kept in the opponent pool")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previously saved model (.zip) to continue training instead of starting fresh. "
        "The timestep counter keeps counting up from the checkpoint's own total.",
    )
    args = parser.parse_args()

    pool = OpponentPool(current_prob=args.current_prob, max_snapshots=args.max_snapshots)
    opponent_fn = pool if args.opponent == "self" else random_masked_policy

    vec_env = DummyVecEnv([make_env_fn(args.num_players, args.deck, opponent_fn) for _ in range(args.n_envs)])

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env, tensorboard_log=args.tensorboard_log)
        print(f"Checkpoint was already trained for {model.num_timesteps} timesteps.")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            n_steps=256,
            batch_size=256,
            tensorboard_log=args.tensorboard_log,
        )
    if args.opponent == "self":
        pool.model = model

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    callback = None
    if args.opponent == "self":
        snapshot_dir = os.path.join(out_dir or ".", "opponent_snapshots")
        if args.resume:
            # A resumed run starts with an empty pool otherwise - reload
            # whatever older snapshots this --out's previous run already
            # accumulated, so "older models" survive across --resume calls.
            existing = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.zip")), key=os.path.getmtime)
            for path in existing[-args.max_snapshots :]:
                pool.add_snapshot(MaskablePPO.load(path, device="cpu"))
            if existing:
                print(f"Reloaded {min(len(existing), args.max_snapshots)} older snapshot(s) from {snapshot_dir}")
        callback = SnapshotCallback(pool, snapshot_dir, args.snapshot_every, verbose=1)

    # reset_num_timesteps=False so a resumed run's progress bar / logging
    # keeps counting from the checkpoint's total rather than restarting at 0.
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True, reset_num_timesteps=not args.resume)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip (total lifetime timesteps: {model.num_timesteps})")


if __name__ == "__main__":
    main()
