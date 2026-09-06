# Sell Me a Sasquatch — Rust Engine + Python Multi-Agent RL Environment

A deterministic, seedable implementation of **Sell Me a Sasquatch** as a Rust
crate (`engine/`), exposed to Python via PyO3 (`bindings/`), wrapped as a
PettingZoo `AECEnv` multi-agent RL environment (`python/`). See `PROMPT.md`
for the full game-rules spec this was built against.

## Repo layout

```
engine/       Rust crate: pure game logic (GameState, Action, Event, rules)
  src/encode.rs   fixed-shape observation + action-feature encoding
  src/stats.rs    per-episode memory folded out of the rules' own Events
bindings/     PyO3 crate exposing GameState to Python as `_native`
python/       `sell_me_a_sasquatch` package: PettingZoo AECEnv, self-play env,
              spaces, pointer policy, renderer
  scripts/        train.py, play.py (evaluate), bench.py (throughput)
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
uv run maturin develop --release   # builds bindings/ as sell_me_a_sasquatch._native
uv run pytest tests/ -q
```

`maturin develop` needs to be re-run after any change under `engine/` or
`bindings/` to rebuild the native extension.

**Use `--release`.** `maturin develop` defaults to a debug build, which is
roughly an order of magnitude slower here - and since the engine runs every
micro-step of every self-play episode, that is the difference between a
usable training loop and an unusable one.

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

## Observation and action encoding

Worth reading before the training section — everything else follows from it.
Both halves are produced by Rust (`engine/src/encode.rs`) and written
straight into caller-owned numpy buffers, so a micro-step crosses the FFI
boundary exactly once.

**The state vector** (439 float32s) is:

- *Ego-centric.* Seat `k` is always "the player `k` seats after me", never
  absolute seat `k`. Nothing a policy learns is seat-specific, and what it
  learns at one seat transfers to every other.
- *Padded to `MAX_PLAYERS`/`MAX_DEALS`* with explicit validity flags, so a
  2-player game and a 6-player game have byte-identical shapes.
- *Class counts, not card slots.* Hands and Collections are unordered sets,
  so they are per-class tallies (12 classes) rather than one padded slot per
  card — permutation invariant, and an order of magnitude smaller.
- *Card counting.* Per-class tallies of what is in the discard pile and,
  derived from the deck composition, what is still **unseen** (draw pile +
  other players' hands + face-down deal cards). This is the sufficient
  statistic a human card counter tracks, and it is exactly what a policy
  needs to price a face-down deal.
- *Memory.* The engine folds every rules `Event` into a per-seat behavioral
  summary (Thingamabobs played, tokens gained/lost, cards stolen, sets
  completed, turns led …) plus an exponentially decayed histogram of recent
  event kinds (`engine/src/stats.rs`). A single micro-step observation is
  otherwise a snapshot with no history, which makes this a deep POMDP —
  "this seat has already dumped three Thingamabobs and stolen two tokens" is
  exactly the sort of thing a good player tracks. Keeping the memory in the
  engine gets it without a recurrent policy, which does not compose with
  action masking in sb3-contrib anyway.
- *A per-episode persona.* The last 8 slots are a random vector, drawn once
  per seat per episode. A policy that is deterministic given the state plays
  the same opening from the same deal every time — readable, exploitable,
  and a poor explorer. Conditioning on a latent that is constant *within* an
  episode but resampled *across* episodes lets one set of weights express a
  family of coherent strategies and commit to one per game, rather than
  re-rolling its personality on every micro-turn (which is all that sampling
  from the action distribution gives you).

**The action space** is `Discrete(n)` with an *ordinal* encoding: index `i`
means "the i-th entry of `legal_actions()` right now". That avoids a
combinatorially complete encoding of the nested `Action` type while keeping
a fixed-shape space with an explicit mask (§3.4). Two consequences:

- `n` is *derived*, not guessed: `GameState::max_legal_actions()` computes
  the exact worst case from the deck composition and table size (116/117/171/234/306
  for 2–6 players with the confirmed deck), so a custom `deck.toml` resizes
  the policy head instead of silently truncating legal moves. Legal actions
  are also deduplicated by card class — two Detrital Repositioners offer
  identical menus — which cut the worst case from 919 to 306 without
  removing a single distinct choice. Cards the acting player *cannot* see
  are never deduplicated, since a class-collapsed count would leak how many
  distinct kinds are hidden.
- Because index `i` means something different in every state, the
  observation also carries an `actions` matrix: one 43-float row per
  candidate, describing what that action *does* (its type, which card
  classes it commits, which seat it targets, how many face-down cards it
  touches). `MaskablePointerPolicy` scores each candidate against the state
  from its own description —
  `logit(i) = <encode(action_i), query(state)> / sqrt(d) + bias(action_i)` —
  the standard pointer/attention formulation. Without it, a policy head has
  to reverse-engineer the engine's enumeration order out of the state before
  its outputs can mean anything.

## Training an AI (self-play PPO)

One policy, every table size. [sb3-contrib](https://sb3-contrib.readthedocs.io/)'s
`MaskablePPO` (PPO with invalid-action masking) with the pointer policy
above, trained by self-play: `SasquatchSelfPlayEnv`
(`python/sell_me_a_sasquatch/selfplay_env.py`) gives one "hero" seat to the
learner and plays every other seat with an `opponent_policy` callback.
Because the observation is padded and ego-centric, the env resamples the
**table size** each episode too, so a single checkpoint plays 2- through
6-player games — including the materially different 2-player variant (§2.7).

```sh
cd python
uv pip install -e ".[train]"     # stable-baselines3, sb3-contrib, torch

uv run python scripts/train.py --timesteps 1000000 --out models/sasquatch_general
uv run python scripts/play.py models/sasquatch_general.zip --episodes 300
```

`scripts/play.py` reports win rate per table size against the
`1/num_players` chance baseline. A 400k-timestep run (about 4½ minutes on 12
workers) already beats random opponents by 1.4x–2.7x chance at every table
size:

| Players | Win rate | Chance | vs chance |
|---|---|---|---|
| 2 | 69.0% | 50.0% | 1.38x |
| 3 | 66.7% | 33.3% | 2.00x |
| 4 | 59.3% | 25.0% | 2.37x |
| 5 | 50.7% | 20.0% | 2.53x |
| 6 | 44.7% | 16.7% | 2.68x |

**Parallelism.** Envs run in worker *processes* (`SubprocVecEnv`, one per
core by default) — the engine holds no lock of its own to release, so
threads would not help. Workers cannot call back into the live model, so
each keeps its own opponent league of disk-loaded snapshots that
`SnapshotCallback` refreshes; opponents are therefore up to
`--snapshot-every` timesteps stale, which is exactly the "play slightly
older versions of yourself" regime self-play wants. `--vec dummy` keeps
everything in one process and uses the live model directly, which is the
right choice for debugging and very small runs.

**Opponent pool ("older models").** `OpponentPool` picks, once per episode,
either the newest snapshot or a uniformly random older one (`--current-prob`
controls the mix). Playing only ever against an exact mirror of your current
self can cycle, or overfit to beating that mirror rather than learning
something robust; mixing in past selves is the standard fix, a small-scale
version of the league idea behind AlphaStar/OpenAI Five. Snapshots are
converted to `NumpyPointerPolicy` — the same arithmetic as the torch policy
(asserted in `tests/test_policy.py`), without torch's per-call overhead on
batch-of-one observations. That matters because opponents take
(table size − 1) moves for every one of the learner's.

**Reward.** The terminal +1 (winner) / −1 (everyone else) from §3.4, plus a
*potential-based* dense shaping term tracking each step's change in **lead
margin** — own Point Tokens minus the best opponent's, the quantity that
actually has to go positive to win (§2.6 requires being *strictly* ahead,
not just accumulating tokens). Potential-based shaping
(`gamma * phi(s') - phi(s)`, Ng/Harada/Russell 1999) is the standard way to
add a denser per-step signal to a ~30-step sparse-terminal episode without
changing what the optimal policy is, unlike an arbitrary bonus (e.g. an
earlier version here that rewarded raw own-token gains, ignoring whether
opponents were catching up faster).

Other flags worth knowing:

- `--players 4` (or any subset) pins training to specific table sizes. The
  action space narrows to match, so such a checkpoint is *not* loadable at a
  larger table — the default trains the general model.
- `--resume path/to/model.zip` continues a checkpoint; timesteps keep
  counting from its own total, and that run's opponent snapshots are
  reloaded so the league survives across `--resume` calls.
- `--opponent random` trains against a fixed random baseline (no league).
  Faster to sanity-check, much weaker final policy — it never has to counter
  increasingly sharp play.
- Checkpoints trained against the previous observation format will not
  load: the state vector, the action space width and the policy head all
  changed. `scripts/play.py` and the web app both say so explicitly
  rather than failing with a tensor-shape error mid-game.
- Reasonable runs (hundreds of thousands to a few million timesteps) are a
  starting point for above-chance deal evaluation and bluffing, not a
  guarantee of strong play; genuine multi-agent RL convergence guarantees do
  not apply to this opponent-pool setup.

## Web app

A very simple local Flask app with three modes, built directly on the
engine/env code above. Every mode's setup form also lets you pick who goes
first / buys first (or leave it random) via `Game`'s optional `first_player`
constructor argument (`bindings/src/lib.rs`, wrapping
`GameState::new_with_starting_leader`):

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
  (§"Training an AI"), mixed-table-size episodes, and `OpponentPool`
  bookkeeping.
- `python/tests/test_policy.py`: the pointer policy end to end through
  `MaskablePPO` (rollout, bootstrap and update all take different paths
  through the custom head), save/load round-tripping, and that
  `NumpyPointerPolicy` computes the same logits as the torch original.
  Skipped automatically if the `[train]` extra is not installed.
- `python/tests/test_webapp.py`: the web app's three modes end-to-end via
  Flask's test client (random-policy path only — skipped automatically if
  `[web]` isn't installed).

Run everything:

```sh
cargo test -p sasquatch-engine
cd python && uv run pytest tests/ -q
```

Measure throughput (see "Performance"):

```sh
cd python && uv run python scripts/bench.py --engine --workers 8
```

## Rules assumptions to verify against the physical rulebook

The spec explicitly flagged two rules interpretations as ambiguous in the
absence of a physical rulebook. Both are implemented as small, isolated,
easily-swappable pieces of logic:

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

**2-player mode's split deal (§2.7) — confirmed, no longer an assumption.**
On their turn, the active player splits exactly 3 cards from their *own*
hand between "my pile" and "their pile" (any split summing to 3: 3/0, 2/1,
1/2, 0/3 - which *specific* cards land on which side is itself part of the
choice, not just the count), then reveals exactly one of those 3 cards
face up themselves. The other player then Accepts (each pile goes where it
was placed) or Reverses (the two piles swap owners). The opponent's hand is
never touched and never needs to be known. See
`Action::TwoPlayerSubmitDeal`, `GameState::apply_two_player_submit_deal`,
and `Phase::TwoPlayerDealOffer` in `engine/src/game.rs`/`phase.rs`.

**2-player Nasty trade-ins are always resolved by the *other* player.**
Buyer mode's "new Buyer resolves it" shortcut (rules assumption 2 above)
doesn't generalize to 2-player mode: an Accept can hand the responder their
own pile back and complete a Nasty set in their *own* Collection the same
instant they become the new turn leader, which would otherwise have them
deciding what they themselves lose. `GameState::nasty_beneficiary` resolves
this explicitly as "the other player relative to whoever's set completed",
not "whoever currently holds `turn_leader`" - see
`nasty_penalties.rs::two_player_nasty_trade_in_is_always_resolved_by_the_other_player`
for the exact scenario this fixes.

## Performance

Two numbers matter, and they are not the same one. `cargo bench` measures
the pure Rust engine; `scripts/bench.py` measures what training actually
consumes — the engine *plus* observation encoding, reward shaping and the
Python environment around it.

Rust engine, full random-policy games against the confirmed 120-card deck,
single-threaded:

| Players | Time / game | Games/sec | Micro-steps/sec |
|---|---|---|---|
| 2 | ~142 µs | ~7,000 | ~289,000 |
| 4 | ~76 µs | ~13,000 | ~706,000 |
| 6 | ~148 µs | ~6,800 | ~684,000 |

The training environment, including the full observation + action-feature
encoding, with random opponents (`uv run python scripts/bench.py`):

| Players | Learner steps/sec (1 proc) | (8 procs) | Episodes/sec (8 procs) |
|---|---|---|---|
| 2 | ~40,000 | ~160,000 | ~3,600 |
| 4 | ~31,000 | ~159,000 | ~5,400 |
| 6 | ~20,000 | ~97,000 | ~2,900 |

That is roughly 8-9x the throughput this environment had before the encoding
moved into Rust, from four changes:

- **One FFI crossing per micro-step.** The observation used to be rebuilt
  card by card in Python, calling back into Rust for each card's kind and
  allocating a string every time. `Game.encode` now fills caller-owned numpy
  buffers in one call.
- **`Vec`-indexed cards.** `CardId`s are contiguous `0..n`, so the engine's
  hottest lookup is an index, not a hash. Kinds live in their own array, so
  the hot path never touches a card's `String` name.
- **Fewer, deduplicated actions.** Collapsing mechanically identical options
  by card class cut the worst-case legal-action count from 919 to 306 —
  which is less enumeration per step *and* a smaller policy head.
- **A cheaper Python hot path.** The deck is parsed once rather than per
  episode; reward shaping reads Point Tokens directly instead of building a
  whole filtered `Observation` per agent per step; and the action mask,
  being a prefix of ones by construction, is rebuilt only when its width
  changes.

Remaining known hot spots, in rough order:

- 2-player deal offers are the most expensive single enumeration in the
  engine (~3.5 µs/micro-step against ~1.4 µs for other table sizes): every
  split allocates two `Vec` piles, and there are up to 80 of them per offer.
  A compact `[CardId; 3]` + keep-mask representation would remove that, at
  the cost of a wider API change than this pass took on.
- Trade-in scanning rescans a Collection from scratch after every completed
  set. That is needed for correctness — a stolen card can cascade into a new
  completion — but an incremental per-class tally would do less work.
- `Vec::remove` (an O(n) shift) is still used for hand/Collection/deal
  removal rather than swap-remove.

At the training level, throughput is no longer bound by the environment at
all: with 12 workers, PPO reaches ~2,900 timesteps/sec end to end, and the
limit is the policy update on the main process, not rollout collection.

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
- **A checkpoint's action-space width is fixed by the table sizes it was
  trained on.** `--players 4` produces a 171-wide head, which cannot be
  evaluated at a 6-player table (306). The default trains across every
  table size, so the general checkpoint plays all of them; `scripts/play.py`
  sizes its env from the loaded model rather than from the table.
- **Worker-process opponents are stale by up to `--snapshot-every`
  timesteps.** `SubprocVecEnv` workers cannot hold the live model, so the
  league is refreshed from disk. This is a deliberate trade (parallelism for
  slightly older opponents, which self-play wants anyway), but it does mean
  `--current-prob` means "newest snapshot", not "live model", unless you run
  `--vec dummy`.
- **The memory features are a summary, not a transcript.** Per-seat counters
  and a decayed event histogram capture *how* a seat has been playing, but
  not the exact sequence - a policy cannot, say, recall which specific card
  an opponent revealed four turns ago. A recurrent policy would, but
  sb3-contrib's `RecurrentPPO` and `MaskablePPO` do not compose, so that
  would mean writing the maskable-recurrent combination from scratch.
