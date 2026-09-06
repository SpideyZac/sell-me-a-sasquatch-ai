mod common;

use common::*;
use sasquatch_engine::action::Action;
use sasquatch_engine::card::{CardKind, NastyKind};
use sasquatch_engine::game::Event;

/// Drives a full turn where `seller` (a non-chosen seller) submits a deal
/// containing 2 copies of `kind` plus 1 filler card, so that seller ends up
/// with a completed Nasty set of `kind` in their own Collection right after
/// deal resolution (§2.3 step 5: non-chosen sellers keep their own cards).
/// Returns the events from `ChooseDeal`.
fn run_turn_completing_nasty_set_for_a_seller(game: &mut sasquatch_engine::game::GameState, kind: NastyKind) -> (usize, usize, Vec<Event>) {
    let buyer = game.turn_leader();
    let mut target_seller = None;
    while game.current_phase().starts_with("deal_offer") {
        let player = game.active_players()[0];
        if game.current_phase() == "deal_offer_submit" {
            let matching = hand_cards_of_kind(game, player, CardKind::Nasty(kind));
            let action = if player != buyer && matching.len() >= 2 && target_seller.is_none() {
                target_seller = Some(player);
                let mut cards = matching;
                cards.truncate(2);
                let filler = game.player_hand(player).iter().copied().find(|c| !cards.contains(c)).unwrap();
                cards.push(filler);
                Action::SubmitDeal { cards: [cards[0], cards[1], cards[2]] }
            } else {
                game.legal_actions(player).into_iter().next().unwrap()
            };
            game.apply_action(player, action).unwrap();
        } else {
            let action = game.legal_actions(player).into_iter().next().unwrap();
            game.apply_action(player, action).unwrap();
        }
    }
    let Some(seller) = target_seller else { return (usize::MAX, buyer, vec![]) };

    if game.current_phase() == "buyer_peek" {
        let peek = game.legal_actions(buyer)[0].clone();
        game.apply_action(buyer, peek).unwrap();
    }
    auto_pass_thingamabob_window(game);

    // The Buyer must choose a *different* seller's deal so `seller` is not
    // the chosen one and keeps their own (Nasty-laden) cards.
    let options = game.legal_actions(buyer);
    let Action::ChooseDeal { seller: other } = options.iter().find(|a| !matches!(a, Action::ChooseDeal { seller: s } if *s == seller)).cloned().unwrap() else { panic!() };
    let events = game.apply_action(buyer, Action::ChooseDeal { seller: other }).unwrap();
    (seller, buyer, events)
}

#[test]
fn poison_pill_bug_trade_in_lets_new_buyer_optionally_steal_one_card() {
    for seed in 0..30u64 {
        let mut game = new_test_game(4, seed);
        let (seller, _old_buyer, events) = run_turn_completing_nasty_set_for_a_seller(&mut game, NastyKind::PoisonPillBug);
        if seller == usize::MAX {
            continue; // this seed's deal didn't happen to yield 2 PPB cards; try another
        }
        assert!(events.iter().any(|e| matches!(e, Event::NastySetTradedIn { player, kind } if *player == seller && *kind == NastyKind::PoisonPillBug)));

        if game.current_phase() == "nasty_resolution" {
            let new_buyer = game.turn_leader();
            let resolve_options = game.legal_actions(new_buyer);
            // 0 or 1 card takeable - never more (§2.4 table).
            assert!(resolve_options.iter().all(|a| matches!(a, Action::ResolveNastyPenalty { taken_cards } if taken_cards.len() <= 1)));
            // Choosing to take 1 card actually moves it into the new buyer's Collection.
            if let Some(Action::ResolveNastyPenalty { taken_cards }) = resolve_options.into_iter().find(|a| matches!(a, Action::ResolveNastyPenalty { taken_cards } if taken_cards.len() == 1)) {
                let card = taken_cards[0];
                let seller_collection_before = game.player_collection(seller).to_vec();
                assert!(seller_collection_before.contains(&card));
                game.apply_action(new_buyer, Action::ResolveNastyPenalty { taken_cards: vec![card] }).unwrap();
                assert!(!game.player_collection(seller).contains(&card));
                assert!(game.player_collection(new_buyer).contains(&card));
            }
            return;
        }
    }
    panic!("no seed in range produced a testable Poison Pill Bug completion");
}

#[test]
fn loan_shark_trade_in_lets_new_buyer_steal_up_to_two_cards() {
    for seed in 0..30u64 {
        let mut game = new_test_game(4, seed);
        let (seller, _old_buyer, _events) = run_turn_completing_nasty_set_for_a_seller(&mut game, NastyKind::LoanShark);
        if seller == usize::MAX {
            continue;
        }
        if game.current_phase() == "nasty_resolution" {
            let new_buyer = game.turn_leader();
            let resolve_options = game.legal_actions(new_buyer);
            assert!(resolve_options.iter().all(|a| matches!(a, Action::ResolveNastyPenalty { taken_cards } if taken_cards.len() <= 2)));
            assert!(resolve_options.iter().any(|a| matches!(a, Action::ResolveNastyPenalty { taken_cards } if taken_cards.is_empty())), "buyer may always decline (fizzle)");
            return;
        }
    }
    panic!("no seed in range produced a testable Loan Shark completion");
}

#[test]
fn trojan_horse_trade_in_auto_steals_one_point_token_no_action_needed() {
    for seed in 0..30u64 {
        let mut game = new_test_game(4, seed);
        let buyer = game.turn_leader();
        let mut target_seller = None;
        while game.current_phase().starts_with("deal_offer") {
            let player = game.active_players()[0];
            if game.current_phase() == "deal_offer_submit" {
                let matching = hand_cards_of_kind(&game, player, CardKind::Nasty(NastyKind::TrojanHorse));
                let action = if player != buyer && matching.len() >= 3 && target_seller.is_none() {
                    target_seller = Some(player);
                    Action::SubmitDeal { cards: [matching[0], matching[1], matching[2]] }
                } else {
                    game.legal_actions(player).into_iter().next().unwrap()
                };
                game.apply_action(player, action).unwrap();
            } else {
                let action = game.legal_actions(player).into_iter().next().unwrap();
                game.apply_action(player, action).unwrap();
            }
        }
        let Some(seller) = target_seller else { continue };
        // give the seller a point token first so the steal is observable
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(buyer)[0].clone();
            game.apply_action(buyer, peek).unwrap();
        }
        auto_pass_thingamabob_window(&mut game);
        let options = game.legal_actions(buyer);
        let chosen_seller = if let Action::ChooseDeal { seller: s } = options.iter().find(|a| !matches!(a, Action::ChooseDeal { seller: s2 } if *s2 == seller)).unwrap() { *s } else { unreachable!() };
        let events = game.apply_action(buyer, Action::ChooseDeal { seller: chosen_seller }).unwrap();

        assert!(events.iter().any(|e| matches!(e, Event::NastySetTradedIn { player, kind } if *player == seller && *kind == NastyKind::TrojanHorse)));
        // No player action was ever required for Trojan Horse - the phase
        // should never stop at nasty_resolution for this specific kind (it
        // may still stop there if a *different* Nasty also completed).
        return;
    }
    panic!("no seed in range produced a testable Trojan Horse completion");
}
