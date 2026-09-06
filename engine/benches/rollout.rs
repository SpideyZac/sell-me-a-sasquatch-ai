//! §3.5 performance target: >100k full-game simulations/sec single-threaded
//! for random-policy self-play, measured against the confirmed 120-card
//! deck (`configs/deck.toml`).

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use sasquatch_engine::deck::DeckConfig;
use sasquatch_engine::game::GameState;

const CONFIRMED_DECK_TOML: &str = include_str!("../../configs/deck.toml");

/// Cheap, allocation-free "random" policy: picks an action deterministically
/// from a running counter so the benchmark measures engine throughput, not
/// RNG call overhead from a full-blown random source.
fn play_one_full_game(deck: DeckConfig, num_players: usize, seed: u64) {
    let mut game = GameState::new(num_players, deck, seed).unwrap();
    let mut counter: u64 = seed;
    let max_steps = 5_000;

    for _ in 0..max_steps {
        if game.is_game_over() {
            break;
        }
        let player = game.active_players()[0];
        let actions = game.legal_actions(player);
        if actions.is_empty() {
            break; // deck-starvation edge case on a very long episode; stop this rollout
        }
        counter = counter.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (counter as usize) % actions.len();
        game.apply_action(player, actions[idx].clone()).unwrap();
    }
}

fn bench_full_game_rollouts(c: &mut Criterion) {
    let deck = DeckConfig::from_toml_str(CONFIRMED_DECK_TOML).unwrap();
    let mut group = c.benchmark_group("full_game_rollout");
    for &num_players in &[2usize, 4, 6] {
        let mut seed = 0u64;
        group.bench_function(format!("{num_players}_players"), |b| {
            b.iter_batched(
                || {
                    seed += 1;
                    (deck.clone(), seed)
                },
                |(deck, seed)| play_one_full_game(deck, num_players, seed),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_full_game_rollouts);
criterion_main!(benches);
