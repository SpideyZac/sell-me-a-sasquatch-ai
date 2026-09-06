"""Train one general Sell Me a Sasquatch agent by self-play.

A single `MaskablePPO` policy plays every seat at every table size. The
observation is padded and ego-centric (`sell_me_a_sasquatch/spaces.py`), so
2- through 6-player games, including the materially different two-player
variant, share one observation and action space, and one checkpoint plays
all of them.

Opponents come from `OpponentPool`: a rotating league of the policy's own
frozen snapshots, refreshed by `SnapshotCallback`, so the learner faces a
curriculum instead of only ever a mirror of its current self.

Usage:
    uv run python scripts/train.py --timesteps 2000000
    uv run python scripts/train.py --players 4 --timesteps 300000   # one table size
    uv run python scripts/train.py --resume models/sasquatch.zip --timesteps 500000

Parallelism
-----------
Environments run in worker processes (`SubprocVecEnv`), which is what makes
rollout collection scale with cores, since the engine releases no GIL of
its own, so threads would not help. That means workers cannot call back
into the live model, so each worker keeps its own opponent league on
disk-loaded snapshots: `SnapshotCallback` saves the model and tells every
worker to load it. The opponents are therefore up to `--snapshot-every`
timesteps stale, which is exactly the "play against slightly older
versions of yourself" regime self-play wants anyway. `--vec dummy` keeps
everything in one process and uses the live model directly, which is the
right choice for debugging and for very small runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# scripts/ is run directly, so make the package importable without an install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from sell_me_a_sasquatch.env import DEFAULT_DECK_PATH  # noqa: E402
from sell_me_a_sasquatch.policy import MaskablePointerPolicy  # noqa: E402
from sell_me_a_sasquatch.selfplay_env import (  # noqa: E402
    OpponentPool,
    SasquatchSelfPlayEnv,
    random_masked_policy,
)


def make_env(players, deck_path, use_pool, current_prob, max_snapshots, seed):
    """Env factory. Module-level (not a closure) so `SubprocVecEnv` can
    pickle it for `spawn`-based workers on Windows/macOS."""
    import torch as th

    # each worker is one of many processes already; letting every one of
    # them fan out over all cores just makes them fight each other
    th.set_num_threads(1)

    opponent = OpponentPool(current_prob=current_prob, max_snapshots=max_snapshots) if use_pool else random_masked_policy
    env = SasquatchSelfPlayEnv(players=players, deck_config_path=deck_path, opponent_policy=opponent)
    env.reset(seed=seed)
    env = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


class SnapshotCallback(BaseCallback):
    """Periodically freezes the live model into every worker's opponent pool.

    In-process (`DummyVecEnv`) the pool can hold the live model object
    directly. Across processes it cannot, so the model is written to disk and
    each worker loads it - see this module's docstring on why slightly stale
    opponents are fine, and desirable.
    """

    def __init__(self, save_dir: str, every_n_steps: int, in_process_pool: OpponentPool | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.in_process_pool = in_process_pool
        self._last_snapshot_at = 0
        self._count = 0

    def _init_callback(self) -> None:
        """Ensures the snapshot directory exists before training starts."""
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Saves a snapshot and publishes it to every worker if enough steps have passed."""
        if self.num_timesteps - self._last_snapshot_at < self.every_n_steps:
            return True
        self._last_snapshot_at = self.num_timesteps
        path = os.path.join(self.save_dir, f"snapshot_{self.num_timesteps}.zip")
        self.model.save(path)
        self._count += 1
        if self.in_process_pool is not None:
            self.in_process_pool.add_snapshot(MaskablePPO.load(path, device="cpu"))
        else:
            self.training_env.env_method("add_opponent_snapshot", path)
        if self.verbose:
            print(f"[snapshot] {self._count} opponent snapshot(s) published at {self.num_timesteps} timesteps")
        return True


def build_parser() -> argparse.ArgumentParser:
    """Builds this script's command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--players",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        choices=range(2, 7),
        help="Table sizes to train across; one is drawn per episode. The default trains a single general model.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--deck", type=str, default=DEFAULT_DECK_PATH)
    parser.add_argument("--out", type=str, default="models/sasquatch_ppo")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel environment workers (default: one per core, less one for the learner)",
    )
    parser.add_argument("--vec", choices=["subproc", "dummy"], default="subproc", help="'dummy' keeps every env in this process")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--opponent",
        choices=["self", "random"],
        default="self",
        help="'self' = a league of the policy's own frozen snapshots (default); 'random' = a random baseline only, much weaker",
    )
    parser.add_argument("--snapshot-every", type=int, default=50_000, help="Timesteps between opponent-league snapshots")
    parser.add_argument(
        "--current-prob",
        type=float,
        default=0.5,
        help="Chance an episode's opponent is the newest snapshot rather than a random older one",
    )
    parser.add_argument("--max-snapshots", type=int, default=8, help="Opponent league size per worker")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout length per env between updates")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus; keeps the policy exploring the wide action space")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--action-embed-dim", type=int, default=64, help="Width of the per-action embedding in the pointer head")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (.zip) to continue training from")
    return parser


def main(argv=None):
    """Parses arguments and runs one training run."""
    args = build_parser().parse_args(argv)
    players = tuple(sorted(set(args.players)))
    use_pool = args.opponent == "self"
    # a live-model opponent needs the model in this process, which only
    # the single-process vec env can offer
    in_process = args.vec == "dummy"

    env_fns = [
        (lambda i=i: make_env(players, args.deck, use_pool, args.current_prob, args.max_snapshots, args.seed + i))
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if in_process else SubprocVecEnv(env_fns, start_method="spawn")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env, device=args.device, tensorboard_log=args.tensorboard_log)
        print(f"Checkpoint already trained for {model.num_timesteps} timesteps.")
    else:
        model = MaskablePPO(
            MaskablePointerPolicy,
            vec_env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            tensorboard_log=args.tensorboard_log,
            policy_kwargs=dict(net_arch=list(args.net_arch), action_embed_dim=args.action_embed_dim),
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    snapshot_dir = os.path.join(out_dir or ".", "opponent_snapshots")

    callback = None
    if use_pool:
        live_pool = None
        if in_process:
            live_pool = vec_env.envs[0].unwrapped.opponent_policy
            live_pool.model = model
            # every in-process env shares one pool, so the league (and the
            # live model) stay consistent across them
            for env in vec_env.envs:
                env.unwrapped.opponent_policy = live_pool
        elif args.resume:
            # a resumed run would otherwise start against a random
            # baseline; hand the workers back whatever league the
            # previous run left
            existing = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.zip")), key=os.path.getmtime)
            for path in existing[-args.max_snapshots :]:
                vec_env.env_method("add_opponent_snapshot", path)
            if existing:
                print(f"Reloaded {min(len(existing), args.max_snapshots)} opponent snapshot(s) from {snapshot_dir}")
        callback = SnapshotCallback(snapshot_dir, args.snapshot_every, in_process_pool=live_pool, verbose=1)

    print(
        f"Training one policy across {players} players on {args.n_envs} {'in-process' if in_process else 'worker'} env(s); "
        f"action space width {vec_env.action_space.n}"
    )
    # reset_num_timesteps=False so a resumed run keeps counting from the
    # checkpoint's own total rather than restarting at 0
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )
    model.save(args.out)
    print(f"Saved model to {args.out}.zip (total lifetime timesteps: {model.num_timesteps})")
    vec_env.close()


if __name__ == "__main__":
    main()
