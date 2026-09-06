# Sell Me a Sasquatch AI

A Rust implementation of the _Sell Me a Sasquatch_ card game rules, exposed
to Python for training and evaluating a self-play reinforcement learning
agent, with a small local web app for watching, playing against, or getting
move advice from a trained model.

- `engine/`: the Rust rules engine (pure game logic, no I/O)
- `bindings/`: PyO3 bindings exposing the engine to Python
- `python/sell_me_a_sasquatch/`: the PettingZoo environment, self-play
  training environment, and pointer-network policy
- `python/scripts/`: training, evaluation, and benchmarking scripts
- `python/webapp/`: the local Flask app (watch, play, and live advisor modes)
- `configs/deck.toml`: the confirmed 120-card deck

## Install

Requires Rust (stable) and Python 3.12+. This project uses
[uv](https://docs.astral.sh/uv/) for Python dependencies.

```sh
cd python
uv venv --python 3.12
uv pip install maturin
uv run maturin develop --release
```

Use `--release`: a debug build is roughly an order of magnitude slower, and
the engine runs every micro-step of every training episode.

Re-run `maturin develop --release` after any change under `engine/` or
`bindings/`.

## Formatting

```sh
cargo +nightly fmt --all
npx prettier --write .
black .
```

## Build and test

```sh
# rust engine
cargo test -p sasquatch-engine
cargo bench -p sasquatch-engine

# python
cd python
uv pip install -e ".[web,train]"
uv run pytest tests/ -q
```

`cargo test`/`cargo build` at the workspace root also builds the bindings
crate, which needs a Python interpreter available to link against. Scope
commands to `-p sasquatch-engine` to skip that.

## Train

Trains one general policy across every table size (2 to 6 players) by
self-play with `MaskablePPO`:

```sh
cd python
uv run python scripts/train.py --timesteps 1000000 --out models/sasquatch_general
```

See `scripts/train.py --help` for options (table sizes, opponent pool
settings, resuming a checkpoint, and so on).

## Evaluate

```sh
uv run python scripts/play.py models/sasquatch_general.zip --episodes 300
```

Reports win rate per table size against the random-chance baseline. Measure
throughput with:

```sh
uv run python scripts/bench.py --engine --workers 8
```

## Web app

```sh
cd python
uv pip install -e ".[web]"
uv run python webapp/app.py
```

Then open <http://127.0.0.1:5000/>. Three modes: watch AI models play each
other, play against them yourself, or use the live advisor to get move
recommendations for an actual physical game (nothing is dealt for you; you
tell it what's on the table as it becomes visible, and the real engine
handles all the bookkeeping).

## How it works

The engine tracks full game state and exposes a fixed-shape observation and
an ordinal action space, so one policy can play any table size. The
observation is ego-centric (seat `k` always means "the player `k` seats
after me") and encodes hands and collections as per-class card counts
rather than individual card identities, since identical cards are
mechanically interchangeable. A small per-seat memory of past events (point
tokens stolen, sets completed, and so on) is folded into the observation so
a feed-forward policy still has a sense of what happened earlier in the
game.

Because the action space is ordinal (index `i` means "the `i`-th legal
action right now," not a fixed move), the policy scores each currently
legal action from a feature description of what it actually does, rather
than learning a fixed meaning per index. Training uses self-play: one
learner seat faces a pool of the policy's own past snapshots so it faces a
gradually improving opponent instead of only ever a mirror of itself.

Two rules were ambiguous in the absence of a physical rulebook and were
resolved with a specific, documented choice: which of a seller's hidden
cards flips during the buyer's peek (resolved uniformly at random), and
whether the old or new buyer resolves a nasty-card penalty after a deal is
chosen (resolved by the new buyer). Both are isolated pieces of logic that
can be swapped if the physical rules say otherwise.

## Copyright and acceptable use

_Sell Me a Sasquatch_ is a copyrighted card game. This repository does not
include any of the game's copyrighted artwork, card text, or rulebook, and
is not affiliated with or endorsed by the game's publisher or designers.
The code in this repository (the Rust engine, Python environment, and web
app) will be released under the Apache License 2.0; that license covers
the software only, not the underlying game design, name, or any of its
copyrighted content. Use this project to study or build game-playing AI,
not to reproduce or redistribute the game itself.
