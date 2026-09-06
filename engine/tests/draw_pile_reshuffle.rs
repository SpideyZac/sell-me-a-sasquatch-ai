//! If the draw pile is empty when a draw is needed, the discard pile is
//! shuffled to form a new draw pile first.

mod common;

use common::*;
use sasquatch_engine::{
    action::Action,
    deck::DeckConfig,
    game::{Event, GameState},
};

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

    let first = game.turn_leader();
    let second = 1 - first;

    // turn 1 (two-player mode): the active player's 3-card split-deal
    // removes exactly 3 cards from their own hand, so refilling them
    // exactly drains the 3-card draw pile, no reshuffle needed yet
    auto_play_deal_offers(&mut game);
    auto_pass_thingamabob_window(&mut game);
    game.apply_action(second, Action::RespondToDeal { reverse: false })
        .unwrap();
    assert_eq!(
        game.turn_leader(),
        second,
        "active player alternates every turn"
    );

    // turn 2: the new active player (second) needs another 3-card refill,
    // but the draw pile is now empty while the discard pile (from turn 1's
    // instant giant trade-ins) has cards, forcing a reshuffle
    auto_play_deal_offers(&mut game);
    auto_pass_thingamabob_window(&mut game);
    let events = game
        .apply_action(first, Action::RespondToDeal { reverse: false })
        .unwrap();

    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::DrawPileReshuffledFromDiscard)),
        "expected a reshuffle event: {events:#?}"
    );
    assert_eq!(game.player_hand(first).len(), 5);
    assert_eq!(game.player_hand(second).len(), 5);
    assert_eq!(game.player_point_tokens(first), 3);
    assert_eq!(game.player_point_tokens(second), 3);
}
