//! §5 property-based tests: random legal-action sequences never panic, total
//! card count is conserved, and Point Token totals never underflow.

mod common;

use common::new_test_game;
use proptest::prelude::*;

fn run_random_legal_policy(num_players: usize, seed: u64, choice_indices: &[u32], max_steps: usize) -> sasquatch_engine::game::GameState {
    let mut game = new_test_game(num_players, seed);
    let total_before = game.total_card_count();

    for step in 0..max_steps {
        if game.is_game_over() {
            break;
        }
        let player = game.active_players()[0];
        let actions = game.legal_actions(player);
        assert!(!actions.is_empty(), "action_mask must never be empty for the acting agent (phase={})", game.current_phase());
        assert!(
            actions.len() <= game.max_legal_actions(),
            "legal action count {} exceeded the declared bound {} (phase={})",
            actions.len(),
            game.max_legal_actions(),
            game.current_phase()
        );
        let idx = choice_indices[step % choice_indices.len()] as usize % actions.len();
        game.apply_action(player, actions[idx].clone()).unwrap();

        assert_eq!(game.total_card_count(), total_before, "card count must be conserved at every step");
        for p in 0..num_players {
            assert!(game.player_point_tokens(p) < 1_000_000, "sanity bound, would also catch a wraparound underflow");
        }
    }
    game
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_legal_action_sequences_never_panic_and_conserve_cards(
        num_players in 2usize..=6,
        seed in any::<u64>(),
        choice_indices in prop::collection::vec(any::<u32>(), 50..150),
    ) {
        run_random_legal_policy(num_players, seed, &choice_indices, 300);
    }
}

/// The confirmed 120-card deck's bound is what the Python action space is
/// sized from, so pin it: a change here means every trained checkpoint's
/// policy head no longer matches the environment.
#[test]
fn confirmed_deck_action_bounds_are_stable() {
    use sasquatch_engine::deck::DeckConfig;
    use sasquatch_engine::game::GameState;

    let toml = include_str!("../../configs/deck.toml");
    let bounds: Vec<usize> = (2..=6)
        .map(|n| GameState::new(n, DeckConfig::from_toml_str(toml).unwrap(), 0).unwrap().max_legal_actions())
        .collect();
    assert_eq!(bounds, vec![116, 117, 171, 234, 306]);
}
