# Sell Me a Sasquatch — Rust Engine + Python Multi-Agent RL Environment

A deterministic, seedable implementation of **Sell Me a Sasquatch** as a Rust
crate (`engine/`), exposed to Python via PyO3 (`bindings/`), wrapped as a
PettingZoo `AECEnv` multi-agent RL environment (`python/`). See `PROMPT.md`
for the full game-rules spec this was built against.

## Repo layout

```
engine/       Rust crate: pure game logic (GameState, Action, Event, rules)
bindings/     PyO3 crate exposing GameState to Python as `_native`
python/       `sell_me_a_sasquatch` package: PettingZoo AECEnv + spaces + renderer
configs/      deck.toml — the confirmed 120-card deck (§2.5), authoritative game data
```

## Building

### Rust engine

```sh
cargo test -p sasquatch-engine    # unit + integration + property tests (engine/tests/)
cargo bench -p sasquatch-engine   # criterion full-game-rollout throughput benchmark
```

`engine/` itself has no Python dependency. `cargo test`/`cargo build` at the
workspace root also builds the `bindings/` crate, which needs a Python 3.x
interpreter discoverable (via `PATH`, an active `VIRTUAL_ENV`, or
`PYO3_PYTHON`) purely to link against — scope commands to
`-p sasquatch-engine` to skip that requirement entirely.

`.cargo/config.toml` points `VIRTUAL_ENV` at `python/.venv` (relative to the
repo root) so `cargo build`/`cargo check` — and rust-analyzer, which shells
out to cargo — find that interpreter automatically, without needing the
venv "activated" in whatever shell or editor process invoked cargo. If
rust-analyzer still reports "no Python 3.x interpreter found," it's almost
always because `python/.venv` doesn't exist yet — create it per the next
section, then reload/restart rust-analyzer.

### Python bindings + environment

Requires a Python interpreter; this project uses `uv`.

```sh
cd python
uv venv --python 3.12
uv pip install maturin pytest
uv run maturin develop     # builds bindings/ and installs it as sell_me_a_sasquatch._native
uv run pytest tests/ -q
```

`maturin develop` needs to be re-run after any change under `engine/` or
`bindings/` to rebuild the native extension.

## Running a random-policy sanity episode

Rust (no Python needed) — the criterion benchmark itself runs full random
rollouts; for a one-off:

```sh
cargo test -p sasquatch-engine --test property_tests
```

Python, via the PettingZoo API:

```python
import random
from sell_me_a_sasquatch.env import env

e = env(num_players=4, deck_config_path="configs/deck.toml")
e.reset(seed=0)
for agent in e.agent_iter():
    obs, reward, termination, truncation, info = e.last()
    if termination or truncation:
        action = None
    else:
        legal = obs["action_mask"].nonzero()[0]
        action = int(random.choice(legal))
    e.step(action)
print("winner:", e.unwrapped._game.winner())
```

Or use the plain-text renderer for a human-readable trace:

```python
from sell_me_a_sasquatch.render import render_state
# inside a rollout, with access to the underlying native Game:
print(render_state(e.unwrapped._game))
```

## Training an AI (self-play PPO)

The easiest working recipe against the current env: [sb3-contrib](https://sb3-contrib.readthedocs.io/)'s
`MaskablePPO` (PPO with invalid-action masking) trained via **self-play** —
one shared policy plays every seat, since all seats are mechanically
identical (§2.2). `SasquatchSelfPlayEnv` (`python/sell_me_a_sasquatch/selfplay_env.py`)
wraps the multi-agent `AECEnv` as a single-agent `gymnasium.Env`: one "hero"
seat (re-randomized every episode) is controlled by the RL policy, and every
other seat's turn is played by an `opponent_policy` callback.

**Opponent pool ("older models").** That callback is `OpponentPool`: each
episode it's either the live in-training model, or a uniformly random older
*frozen* snapshot of it (`current_prob` controls the mix, default 50/50).
`scripts/train.py`'s `SnapshotCallback` saves a new snapshot into the pool
every `--snapshot-every` timesteps. This matters because playing only ever
against an exact mirror of your current self can cycle or over-fit to
beating that mirror specifically rather than learning something robust —
mixing in a handful of past selves gives a bit of curriculum/diversity, a
small-scale version of the "league" idea behind AlphaStar/OpenAI Five's
self-play.

**Reward.** One reward function, not a sparse/dense choice: the terminal
+1 (winner) / -1 (everyone else) from §3.4, plus a *potential-based* dense
shaping term tracking each step's change in **lead margin** — own Point
Tokens minus the best opponent's, the quantity that actually has to go
positive to win (§2.6 requires being *strictly* ahead of everyone, not just
accumulating tokens). Potential-based shaping (`gamma * phi(s') - phi(s)`,
Ng/Harada/Russell 1999) is the standard way to add a denser per-step signal
to a ~29-step sparse-terminal episode without changing what the optimal
policy actually is, unlike an arbitrary bonus (e.g. an earlier version of
this that rewarded raw own-token gains, ignoring whether opponents were
catching up faster).

```sh
cd python
uv pip install -e ".[train]"     # stable-baselines3, sb3-contrib, torch

uv run python scripts/train.py --num-players 4 --timesteps 300000 --n-envs 8 --out models/sasquatch_ppo_4p
uv run python scripts/play.py models/sasquatch_ppo_4p.zip --num-players 4 --episodes 200
```

Continue training an existing checkpoint with `--resume path/to/model.zip`
(timesteps keep counting up from the checkpoint's own total; the opponent
pool's older snapshots from that checkpoint's own run are reloaded too, so
"older models" survive across `--resume` calls — use a different `--out` if
you want to keep the earlier checkpoint file around as well).

`scripts/play.py` reports the trained model's win rate against random-policy
opponents (compare against the `1/num_players` random-chance baseline it
also prints) or, with `--opponent self`, against a frozen copy of itself.

Notes:

- Self-play requires the opponent callback to call back into the live model
  (and occasionally a loaded snapshot), so training runs single-process
  (`DummyVecEnv`) rather than across subprocesses — `--n-envs` controls how
  many env copies run in that one process, not worker processes.
- `--opponent random` trains against a fixed random baseline instead of
  self-play (no opponent pool either), which is faster to sanity-check but
  produces a much weaker final policy (it never has to counter increasingly
  sharp play).
- The observation `Dict` space (`spaces.py`) is fixed-shape by construction
  (§3.4), so `MaskablePPO`'s `MultiInputPolicy` (SB3's `CombinedExtractor`)
  handles it directly — no custom feature extractor was needed. `MultiDiscrete`
  subspaces are kept 1-D (e.g. `collections` is flattened rather than shaped
  `(num_players, MAX_COLLECTION)`), since SB3's observation preprocessing
  doesn't support multi-dimensional `nvec` arrays — this was the one
  non-obvious fix needed to get `MaskablePPO` to accept the space at all.
- Reasonable training runs (hundreds of thousands to a few million
  timesteps) are a starting point for a self-play policy to develop
  above-chance deal-evaluation and bluffing behavior, not a guarantee of
  strong play; genuine multi-agent RL convergence guarantees don't apply to
  this simple opponent-pool self-play setup. A 250k-timestep run measured
  ~29% win rate vs. random opponents in 4-player games (25% = chance) —
  a real but modest edge; more training, more snapshot diversity, and/or a
  richer action encoding (see "Known limitations") would likely help more
  than reward tuning alone at this point.

## Web app

A very simple local Flask app with three modes, all built directly on the
engine/env code above (no new Rust or binding changes needed):

- **Watch** — a game of AI models (or the random policy — pick per seat,
  independently, so you can e.g. pit an older snapshot against the latest
  checkpoint) playing each other, stepped one micro-turn at a time or on
  autoplay. Reveals every seat's hand for spectating (deals-in-progress stay
  properly hidden until the real rules would reveal them).
- **Play** — you take one seat; the rest are AI models or random, auto-
  resolved between your turns. You only ever see your own hand + public
  info, same as `observe()` in the real env.
- **Move Advisor** — no game session needed: manually describe your hand
  (or, for the "which deal should I take?" advisor, what's been revealed of
  each seller's deal) and get a model's ranked recommendation. This is
  necessarily an *approximation* — it builds a hand-entered snapshot of the
  same observation format the model trains on rather than routing through a
  real `GameState`, so anything you don't specify (other players'
  collections, full deal history, etc.) defaults to empty/neutral. Currently
  covers the deal-offer and choose-deal decisions (the two most interesting
  bluffing/evaluation calls) — extending it to the Thingamabob-window or
  Nasty-penalty decisions is a documented but unimplemented extension.

```sh
cd python
uv pip install -e ".[web]"          # Flask; add [train] too to load real .zip models, not just "random"
uv run python webapp/app.py
```

Then open http://127.0.0.1:5000/. The model dropdowns are populated from
whatever `.zip` files exist under `python/models/` (including
`opponent_snapshots/` subdirectories) — train some first, or just use
"random" everywhere to try the UI out.

## Testing

- `engine/src/deck.rs` (`#[cfg(test)] mod tests`): deck-config parsing and
  the required "totals exactly 120" / "set sizes divide evenly into a
  plausible test deck" sanity checks (§2.5).
- `engine/tests/*.rs`: one suite per rule area — deal offer/reveal/peek
  sequencing and Buyer-marker handoff timing (`buyer_mode_flow.rs`), each
  Nasty penalty (`nasty_penalties.rs`), Creature set completion + trade-in
  ordering (`creature_trade_in.rs`), every Thingamabob effect
  (`thingamabob_effects.rs`), the full 2-player variant
  (`two_player_mode.rs`), win/tie logic (`win_condition.rs`), draw-pile
  reshuffle (`draw_pile_reshuffle.rs`), determinism
  (`determinism.rs`), and property-based random-play invariants
  (`property_tests.rs`, via `proptest`).
- `python/tests/test_env_smoke.py`: PettingZoo's own `api_test` across
  2–6 players, plus randomized episodes asserting no exceptions and a
  never-empty `action_mask` for any live agent.
- `python/tests/test_selfplay_env.py`: the self-play training wrapper
  (§"Training an AI") and `OpponentPool` bookkeeping.
- `python/tests/test_webapp.py`: the web app's three modes end-to-end via
  Flask's test client (random-policy path only — skipped automatically if
  `[web]` isn't installed).

Run everything:

```sh
cargo test -p sasquatch-engine
cd python && uv run pytest tests/ -q
```

## Rules assumptions to verify against the physical rulebook

The spec explicitly flagged two rules interpretations as ambiguous in the
absence of a physical rulebook; a third arose implementing §2.7. All three
are implemented as small, isolated, easily-swappable pieces of logic:

1. **Buyer's extra peek — which hidden card flips (§2.3 step 3).** The
   rulebook doesn't say whether the Buyer picks *which* of a seller's two
   remaining hidden cards gets revealed, or whether it's forced. Implemented
   as: the Buyer picks the *target seller* (`Action::BuyerPeek`); which of
   that seller's still-hidden cards flips is then resolved uniformly at
   random by the engine's seeded RNG. See `GameState::apply_buyer_peek` in
   `engine/src/game.rs`.

2. **Buyer-marker timing vs. Nasty penalty resolution (§2.4 note).** The
   Buyer marker passes to the newly-chosen seller *before* the mandatory
   trade-in phase runs (marker handoff is step 5; trade-in is step 6), so
   Nasty-penalty choices (Poison Pill Bug / Loan Shark steals) are resolved
   by the **new** Buyer, not the player who just finished their turn as
   Buyer. Implemented literally: `turn_leader` is reassigned in
   `apply_choose_deal`/`apply_respond_to_deal` *before* `begin_trade_in()`
   runs. Covered explicitly by
   `buyer_mode_flow.rs::nasty_penalty_resolver_is_the_new_buyer_not_the_old_one`.

3. **2-player mode's "two piles" (§2.7).** The rulebook text ("both players
   take their own 3-card pile into their own Collection" under Accept)
   implies two separate 3-card decks exist each turn, but only describes the
   *active* player building one and revealing a card from it. Implemented
   as: **both** players submit a 3-card deal from hand each turn; only the
   active player's deal gets a face-up reveal (matching the literal "on your
   turn" reveal step — the responder's deal stays fully hidden until
   resolution); Accept keeps each pile with its own maker, Reverse swaps
   them between players. See `DealOfferEntry::needs_reveal` and
   `GameState::apply_respond_to_deal`.

## Performance

`cargo bench` measures full random-policy game rollouts against the
confirmed 120-card deck (`configs/deck.toml`), single-threaded:

| Players | Time / game | Approx. games/sec |
|---|---|---|
| 2 | ~61 µs | ~16,000 |
| 4 | ~94 µs | ~10,600 |
| 6 | ~193 µs | ~5,200 |

This is well short of the >100k games/sec aim in the spec (§3.5). The
current bottlenecks (not yet optimized, given the scope of this pass):

- `legal_actions()` fully materializes every combination up front (e.g. all
  3-card hand subsets, all Nasty-penalty card subsets) rather than lazily
  enumerating or using a bitmask-based scheme.
- Nasty/Creature trade-in scanning rescans each player's full Collection
  from scratch after every single completed set (needed for correctness —
  a stolen card can cascade into a new completion — but is more work than
  a smarter incremental tracker would need).
- `Vec::remove` (O(n) shift) is used throughout for hand/collection/deal
  card removal instead of swap-remove or a slotted arena.

None of this affects correctness; it's flagged here as follow-up work
rather than addressed in this pass, since it would require reworking the
core data structures.

## Known limitations

- **Deck-starvation edge case.** If a game runs long enough (or the deck is
  small enough relative to player count) that both the Draw Pile *and*
  Discard Pile are simultaneously exhausted mid-refill, the engine stops
  drawing rather than inventing a house rule — a player can then reach a new
  turn with fewer than 3 hand cards, at which point `legal_actions()` for
  `SubmitDeal` is legitimately empty. This is not expected to occur in
  practice against the confirmed 120-card deck within a normal game length
  (win thresholds of 4-5 tokens keep episodes short), but very small custom
  test decks can hit it — see the enlarged `TEST_DECK_TOML` in
  `engine/tests/common/mod.rs` for why that fixture isn't the same tiny size
  as the deck-composition unit tests.
- **Option B (`ParallelEnv`) is not implemented** — only Option A (§3.4),
  the micro-stepped `AECEnv`, per the spec's "implement Option A first."
