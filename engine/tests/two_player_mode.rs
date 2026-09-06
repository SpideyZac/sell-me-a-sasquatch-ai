//! §2.7: the materially different 2-player flow. Note the flagged
//! assumption in README - both players build a deal each turn, only the
//! active player's gets a face-up reveal, and Accept/Reverse decides
//! whether the two piles swap collections.

mod common;

use common::*;
use sasquatch_engine::action::Action;
use sasquatch_engine::game::Event;

#[test]
fn two_player_mode_is_selected_automatically_for_2_players() {
    let game = new_test_game(2, 1);
    assert_eq!(game.mode(), sasquatch_engine::game::GameMode::TwoPlayer);
    assert_eq!(game.num_players(), 2);
}

#[test]
fn both_players_start_with_a_5_card_hand() {
    let game = new_test_game(2, 3);
    assert_eq!(game.player_hand(0).len(), 5);
    assert_eq!(game.player_hand(1).len(), 5);
}

#[test]
fn only_the_active_players_deal_gets_a_face_up_reveal() {
    let mut game = new_test_game(2, 4);
    let active = game.turn_leader();
    let responder = 1 - active;

    // Active player: submit then must reveal.
    assert_eq!(game.active_players(), vec![active]);
    let submit = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, submit).unwrap();
    assert_eq!(game.current_phase(), "deal_offer_reveal");
    assert_eq!(game.active_players(), vec![active]);
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    assert!(matches!(reveal, Action::RevealCard { .. }));
    game.apply_action(active, reveal).unwrap();

    // Responder: submit only, straight into the Thingamabob window (no
    // BuyerPeek phase exists in 2-player mode, and no reveal is required).
    assert_eq!(game.current_phase(), "deal_offer_submit");
    assert_eq!(game.active_players(), vec![responder]);
    let submit = game.legal_actions(responder).into_iter().next().unwrap();
    let events = game.apply_action(responder, submit).unwrap();
    assert!(!events.iter().any(|e| matches!(e, Event::CardRevealed { .. })));
    assert_eq!(game.current_phase(), "thingamabob_window");
    assert_eq!(game.active_players(), vec![active], "Thingamabob window starts with the active player");
}

#[test]
fn accept_keeps_each_piles_cards_with_its_own_maker() {
    let mut game = new_test_game(2, 5);
    let active = game.turn_leader();
    let responder = 1 - active;
    auto_play_deal_offers(&mut game);
    auto_pass_thingamabob_window(&mut game);
    assert_eq!(game.current_phase(), "respond_to_deal");
    assert_eq!(game.active_players(), vec![responder]);

    let events = game.apply_action(responder, Action::RespondToDeal { reverse: false }).unwrap();
    let awarded_to_active = events.iter().find_map(|e| if let Event::CardsAwarded { player, cards } = e { (*player == active).then(|| cards.len()) } else { None });
    let awarded_to_responder = events.iter().find_map(|e| if let Event::CardsAwarded { player, cards } = e { (*player == responder).then(|| cards.len()) } else { None });
    assert_eq!(awarded_to_active, Some(3));
    assert_eq!(awarded_to_responder, Some(3));
}

#[test]
fn reverse_swaps_the_two_piles_between_players() {
    let mut game = new_test_game(2, 6);
    let active = game.turn_leader();
    let responder = 1 - active;

    // Track exactly which card ids each player submitted.
    let active_submit = game.legal_actions(active).into_iter().next().unwrap();
    let Action::SubmitDeal { cards: active_cards } = active_submit.clone() else { panic!() };
    game.apply_action(active, active_submit).unwrap();
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, reveal).unwrap();
    let responder_submit = game.legal_actions(responder).into_iter().next().unwrap();
    let Action::SubmitDeal { cards: responder_cards } = responder_submit.clone() else { panic!() };
    game.apply_action(responder, responder_submit).unwrap();

    auto_pass_thingamabob_window(&mut game);
    game.apply_action(responder, Action::RespondToDeal { reverse: true }).unwrap();

    for c in active_cards {
        assert!(game.player_collection(responder).contains(&c) || is_discarded_via_trade_in(&game, c), "active's card should land with responder on reverse");
    }
    for c in responder_cards {
        assert!(game.player_collection(active).contains(&c) || is_discarded_via_trade_in(&game, c), "responder's card should land with active on reverse");
    }
}

fn is_discarded_via_trade_in(game: &sasquatch_engine::game::GameState, card: sasquatch_engine::card::CardId) -> bool {
    // A card can legitimately be gone from the recipient's Collection if it
    // was immediately consumed by a completed-set trade-in this same turn.
    !game.player_collection(0).contains(&card) && !game.player_collection(1).contains(&card)
}

#[test]
fn turn_leader_always_alternates_regardless_of_accept_or_reverse() {
    let mut game = new_test_game(2, 8);
    let first = game.turn_leader();
    auto_play_deal_offers(&mut game);
    auto_pass_thingamabob_window(&mut game);
    let responder = 1 - first;
    game.apply_action(responder, Action::RespondToDeal { reverse: false }).unwrap();
    while game.current_phase() == "nasty_resolution" {
        let r = game.active_players()[0];
        game.apply_action(r, Action::ResolveNastyPenalty { taken_cards: vec![] }).unwrap();
    }
    if !game.is_game_over() {
        assert_eq!(game.turn_leader(), 1 - first, "active player alternates every turn");
    }
}

#[test]
fn two_player_win_threshold_is_5_regardless_of_multiplayer_thresholds() {
    let game = new_test_game(2, 9);
    assert_eq!(game.win_threshold(), 5);
}

#[test]
fn multiplayer_win_thresholds_match_table_size() {
    assert_eq!(new_test_game(3, 1).win_threshold(), 4);
    assert_eq!(new_test_game(4, 1).win_threshold(), 4);
    assert_eq!(new_test_game(5, 1).win_threshold(), 5);
    assert_eq!(new_test_game(6, 1).win_threshold(), 5);
}
