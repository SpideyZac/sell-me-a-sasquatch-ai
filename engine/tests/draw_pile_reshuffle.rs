//! §2.3 step 7: if the Draw Pile is empty when a draw is needed, the
//! Discard Pile is shuffled to form a new Draw Pile first.

mod common;

use sasquatch_engine::action::Action;
use sasquatch_engine::deck::DeckConfig;
use sasquatch_engine::game::{Event, GameState};

/// A deliberately tiny all-Giant (set_size 1) deck: 2 players x 5-card
/// hands + a 3-card draw pile. Every Giant completes an instant 1-card set
/// worth 1 token, guaranteeing the discard pile has cards to reshuffle by
/// the time refill runs dry.
const TINY_DECK_TOML: &str = r#"
    nasties = []
    thingamabobs = []

    [[creature_tiers]]
    tier = "Giant"
    set_size = 1
    copies = 13
"#;

#[test]
fn draw_pile_reshuffles_from_discard_when_exhausted() {
    let deck = DeckConfig::from_toml_str(TINY_DECK_TOML).unwrap();
    let mut game = GameState::new(2, deck, 1).unwrap();

    let active = game.turn_leader();
    let responder = 1 - active;

    let submit = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, submit).unwrap();
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, reveal).unwrap();
    let submit = game.legal_actions(responder).into_iter().next().unwrap();
    game.apply_action(responder, submit).unwrap();

    assert_eq!(game.current_phase(), "thingamabob_window");
    while game.current_phase() == "thingamabob_window" {
        let p = game.active_players()[0];
        game.apply_action(p, Action::PassThingamabobWindow).unwrap();
    }

    let events = game.apply_action(responder, Action::RespondToDeal { reverse: false }).unwrap();

    assert!(events.iter().any(|e| matches!(e, Event::DrawPileReshuffledFromDiscard)), "expected a reshuffle event: {events:#?}");
    assert_eq!(game.player_hand(active).len(), 5);
    assert_eq!(game.player_hand(responder).len(), 5);
    assert_eq!(game.player_point_tokens(active), 3);
    assert_eq!(game.player_point_tokens(responder), 3);
}
