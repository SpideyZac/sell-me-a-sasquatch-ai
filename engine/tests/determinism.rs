//! The same seed plus the same action sequence produces identical
//! observable outcomes, including RNG-driven resolutions like the buyer's
//! forced random peek and draw-pile reshuffles.

mod common;

use common::new_test_game;
use sasquatch_engine::game::{Event, GameState};

/// Deterministic policy: always the first legal action. Applied identically
/// to two independently constructed games with the same seed.
fn run_first_legal_action_policy(game: &mut GameState, max_steps: usize) -> Vec<Event> {
    let mut all_events = Vec::new();
    for _ in 0..max_steps {
        if game.is_game_over() {
            break;
        }
        let player = game.active_players()[0];
        let action = game
            .legal_actions(player)
            .into_iter()
            .next()
            .expect("action_mask must never be empty for the acting agent");
        all_events.extend(game.apply_action(player, action).unwrap());
    }
    all_events
}

#[test]
fn same_seed_and_actions_produce_identical_event_sequences_and_observations() {
    for (num_players, seed) in [(4usize, 111u64), (3, 222), (5, 333), (6, 444), (2, 555)] {
        let mut game_a = new_test_game(num_players, seed);
        let mut game_b = new_test_game(num_players, seed);

        let events_a = run_first_legal_action_policy(&mut game_a, 400);
        let events_b = run_first_legal_action_policy(&mut game_b, 400);

        assert_eq!(events_a, events_b, "seed {seed}: event sequences diverged");
        for p in 0..num_players {
            assert_eq!(
                game_a.observation_for(p),
                game_b.observation_for(p),
                "seed {seed} player {p}: observations diverged"
            );
        }
        assert_eq!(game_a.winner(), game_b.winner());
        assert_eq!(game_a.turn_leader(), game_b.turn_leader());
    }
}

#[test]
fn different_seeds_produce_different_initial_hands() {
    let a = new_test_game(4, 1);
    let b = new_test_game(4, 2);
    assert_ne!(a.player_hand(0), b.player_hand(0));
}
