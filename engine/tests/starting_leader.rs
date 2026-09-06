//! Letting a human-configured game pin who goes first / who buys first
//! (§2.3 step 1; §2.7 for 2-player mode) instead of always picking randomly.

mod common;

use common::TEST_DECK_TOML;
use sasquatch_engine::deck::DeckConfig;
use sasquatch_engine::game::{GameState, SetupError};

fn deck() -> DeckConfig {
    DeckConfig::from_toml_str(TEST_DECK_TOML).unwrap()
}

#[test]
fn starting_leader_pins_the_first_turn_leader() {
    for num_players in 2..=6 {
        for leader in 0..num_players {
            let game = GameState::new_with_starting_leader(num_players, deck(), 1, Some(leader)).unwrap();
            assert_eq!(game.turn_leader(), leader);
        }
    }
}

#[test]
fn starting_leader_none_keeps_the_usual_random_pick() {
    // Same seed as `new`'s implicit random pick should agree exactly.
    let random_game = GameState::new(4, deck(), 42).unwrap();
    let explicit_game = GameState::new_with_starting_leader(4, deck(), 42, None).unwrap();
    assert_eq!(random_game.turn_leader(), explicit_game.turn_leader());
}

#[test]
fn out_of_range_starting_leader_is_rejected() {
    let Err(err) = GameState::new_with_starting_leader(4, deck(), 1, Some(4)) else { panic!("expected an error") };
    assert_eq!(err, SetupError::InvalidStartingLeader { given: 4, num_players: 4 });
}
