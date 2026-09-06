//! §2.7: the materially different 2-player flow. Confirmed against the
//! physical rulebook: on your turn you split exactly 3 of your *own* hand
//! cards between "my pile" and "their pile" (any split summing to 3 - 3/0,
//! 2/1, 1/2, 0/3), flip exactly one of those 3 cards face up yourself, and
//! the other player then Accepts (each pile goes where it was placed) or
//! Reverses (the two piles swap owners). Nasty-set trade-ins always let the
//! *other* player (not whoever's set completed) decide what's taken.

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
fn only_the_active_player_acts_during_the_deal_split_and_reveal() {
    let mut game = new_test_game(2, 4);
    let active = game.turn_leader();
    let responder = 1 - active;

    assert_eq!(game.current_phase(), "deal_offer_submit");
    assert_eq!(game.active_players(), vec![active]);
    let Action::TwoPlayerSubmitDeal { own_pile, other_pile } = game.legal_actions(active).into_iter().next().unwrap() else { panic!() };
    assert_eq!(own_pile.len() + other_pile.len(), 3, "the split must always total 3 cards");
    game.apply_action(active, Action::TwoPlayerSubmitDeal { own_pile, other_pile }).unwrap();

    assert_eq!(game.current_phase(), "deal_offer_reveal");
    assert_eq!(game.active_players(), vec![active], "only the active player reveals - never the responder");
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    assert!(matches!(reveal, Action::RevealCard { .. }));
    let events = game.apply_action(active, reveal).unwrap();

    assert!(events.iter().any(|e| matches!(e, Event::CardRevealed { .. })));
    assert_eq!(game.current_phase(), "thingamabob_window");
    assert_eq!(game.active_players(), vec![active], "Thingamabob window starts with the active player");
    assert_eq!(game.legal_actions(responder), vec![], "the responder never gets a deal-offer action of their own");
}

#[test]
fn every_split_size_is_offered_as_a_legal_action() {
    let game = new_test_game(2, 7);
    let active = game.turn_leader();
    let mut sizes: Vec<(usize, usize)> =
        game.legal_actions(active).into_iter().map(|a| { let Action::TwoPlayerSubmitDeal { own_pile, other_pile } = a else { panic!() }; (own_pile.len(), other_pile.len()) }).collect();
    sizes.sort_unstable();
    sizes.dedup();
    assert_eq!(sizes, vec![(0, 3), (1, 2), (2, 1), (3, 0)]);
}

#[test]
fn all_3_cards_come_from_the_active_players_own_hand() {
    let mut game = new_test_game(2, 11);
    let active = game.turn_leader();
    let responder = 1 - active;
    let responder_hand_before = game.player_hand(responder).to_vec();

    let Action::TwoPlayerSubmitDeal { own_pile, other_pile } = game.legal_actions(active).into_iter().next().unwrap() else { panic!() };
    let mut offered: Vec<_> = own_pile.iter().chain(other_pile.iter()).copied().collect();
    offered.sort_unstable();

    let mut active_hand_before: Vec<_> = game.player_hand(active).to_vec();
    active_hand_before.sort_unstable();
    for c in &offered {
        assert!(active_hand_before.contains(c), "every offered card must come from the active player's own hand");
    }

    game.apply_action(active, Action::TwoPlayerSubmitDeal { own_pile, other_pile }).unwrap();
    assert_eq!(game.player_hand(responder), responder_hand_before.as_slice(), "the responder's hand is never touched by the split");
}

#[test]
fn accept_sends_each_pile_where_it_was_placed() {
    let mut game = new_test_game(2, 5);
    let active = game.turn_leader();
    let responder = 1 - active;

    // Force a genuine 2/1 split so both piles are non-empty and distinct.
    let Action::TwoPlayerSubmitDeal { own_pile, other_pile } = game
        .legal_actions(active)
        .into_iter()
        .find(|a| matches!(a, Action::TwoPlayerSubmitDeal { own_pile, other_pile } if own_pile.len() == 2 && other_pile.len() == 1))
        .unwrap()
    else {
        panic!()
    };
    let (own_pile, other_pile) = (own_pile, other_pile);
    game.apply_action(active, Action::TwoPlayerSubmitDeal { own_pile: own_pile.clone(), other_pile: other_pile.clone() }).unwrap();
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, reveal).unwrap();
    auto_pass_thingamabob_window(&mut game);

    let events = game.apply_action(responder, Action::RespondToDeal { reverse: false }).unwrap();
    let awarded_to_active = events.iter().find_map(|e| if let Event::CardsAwarded { player, cards } = e { (*player == active).then(|| cards.clone()) } else { None });
    let awarded_to_responder = events.iter().find_map(|e| if let Event::CardsAwarded { player, cards } = e { (*player == responder).then(|| cards.clone()) } else { None });
    assert_eq!(awarded_to_active.map(|mut c| { c.sort_unstable(); c }), Some({ let mut v = own_pile; v.sort_unstable(); v }));
    assert_eq!(awarded_to_responder.map(|mut c| { c.sort_unstable(); c }), Some({ let mut v = other_pile; v.sort_unstable(); v }));
}

#[test]
fn reverse_swaps_the_two_piles_between_players() {
    let mut game = new_test_game(2, 6);
    let active = game.turn_leader();
    let responder = 1 - active;

    let Action::TwoPlayerSubmitDeal { own_pile, other_pile } = game
        .legal_actions(active)
        .into_iter()
        .find(|a| matches!(a, Action::TwoPlayerSubmitDeal { own_pile, other_pile } if own_pile.len() == 2 && other_pile.len() == 1))
        .unwrap()
    else {
        panic!()
    };
    game.apply_action(active, Action::TwoPlayerSubmitDeal { own_pile: own_pile.clone(), other_pile: other_pile.clone() }).unwrap();
    let reveal = game.legal_actions(active).into_iter().next().unwrap();
    game.apply_action(active, reveal).unwrap();
    auto_pass_thingamabob_window(&mut game);
    game.apply_action(responder, Action::RespondToDeal { reverse: true }).unwrap();

    for c in own_pile {
        assert!(game.player_collection(responder).contains(&c) || is_discarded_via_trade_in(&game, c), "active's own-pile card should land with the responder on reverse");
    }
    for c in other_pile {
        assert!(game.player_collection(active).contains(&c) || is_discarded_via_trade_in(&game, c), "active's other-pile card should land with the active player on reverse");
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
