mod common;

use common::*;
use sasquatch_engine::action::{Action, ThingamabobParams};
use sasquatch_engine::card::{CardKind, Tier, ThingamabobKind};
use sasquatch_engine::game::Event;

#[test]
fn platonic_isolator_steals_one_token_from_a_strictly_richer_player() {
    for seed in 0..40u64 {
        let mut game = new_test_game(4, seed);
        let Some(richer) = stash_cards_of_kind_in_collection(&mut game, CardKind::Creature(Tier::Giant), 1) else { continue };
        assert_eq!(game.player_point_tokens(richer), 1, "a lone Giant is a complete set worth 1 token");

        let Some(actor) = stash_cards_of_kind_in_collection(&mut game, CardKind::Thingamabob(ThingamabobKind::PlatonicIsolator), 1) else { continue };
        if actor == richer {
            continue;
        }
        assert_eq!(game.player_point_tokens(actor), 0);

        auto_play_deal_offers(&mut game);
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(game.turn_leader())[0].clone();
            game.apply_action(game.turn_leader(), peek).unwrap();
        }
        advance_thingamabob_window_to_player(&mut game, actor);
        let Some(action) = find_thingamabob_action(&game, actor, |p| matches!(p, ThingamabobParams::PlatonicIsolator { target_player } if *target_player == richer)) else { continue };
        let events = game.apply_action(actor, action).unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::PointTokenStolen { from, to, amount } if *from == richer && *to == actor && *amount == 1)));
        assert_eq!(game.player_point_tokens(actor), 1);
        assert_eq!(game.player_point_tokens(richer), 0);
        return;
    }
    panic!("no seed produced a testable Platonic Isolator scenario");
}

#[test]
fn detrital_repositioner_removes_one_card_from_a_deal_to_discard() {
    for seed in 0..40u64 {
        let mut game = new_test_game(4, seed);
        let Some(actor) = stash_cards_of_kind_in_collection(&mut game, CardKind::Thingamabob(ThingamabobKind::DetritalRepositioner), 1) else { continue };

        auto_play_deal_offers(&mut game);
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(game.turn_leader())[0].clone();
            game.apply_action(game.turn_leader(), peek).unwrap();
        }
        advance_thingamabob_window_to_player(&mut game, actor);
        let Some(action) = find_thingamabob_action(&game, actor, |p| matches!(p, ThingamabobParams::RemoveFromDeals { removals } if removals.len() == 1)) else { continue };
        let Action::PlayThingamabob { params: ThingamabobParams::RemoveFromDeals { removals }, .. } = &action else { unreachable!() };
        let (target_seller, target_card) = removals[0];
        let events = game.apply_action(actor, action.clone()).unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::CardsDiscarded { cards } if cards == &vec![target_card])));
        let _ = target_seller;
        return;
    }
    panic!("no seed produced a testable Detrital Repositioner scenario");
}

#[test]
fn super_detrital_repositioner_allows_zero_one_or_two_removals_across_up_to_two_deals() {
    for seed in 0..40u64 {
        let mut game = new_test_game(4, seed);
        let Some(actor) = stash_cards_of_kind_in_collection(&mut game, CardKind::Thingamabob(ThingamabobKind::SuperDetritalRepositioner), 1) else { continue };

        auto_play_deal_offers(&mut game);
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(game.turn_leader())[0].clone();
            game.apply_action(game.turn_leader(), peek).unwrap();
        }
        advance_thingamabob_window_to_player(&mut game, actor);
        let options: Vec<_> = game
            .legal_actions(actor)
            .into_iter()
            .filter_map(|a| if let Action::PlayThingamabob { params: ThingamabobParams::RemoveFromDeals { removals }, .. } = &a { Some(removals.len()) } else { None })
            .collect();
        if options.is_empty() {
            continue;
        }
        assert!(options.contains(&0), "playing the card with 0 removals (just to discard it) must be legal");
        assert!(options.iter().all(|&n| n <= 2), "at most 2 cards may be removed");

        // Exercise the 2-cards-from-one-deal shape if available, else 1-each-from-two-deals.
        let action = find_thingamabob_action(&game, actor, |p| matches!(p, ThingamabobParams::RemoveFromDeals { removals } if removals.len() == 2))
            .or_else(|| find_thingamabob_action(&game, actor, |p| matches!(p, ThingamabobParams::RemoveFromDeals { removals } if removals.len() == 1)));
        if let Some(action) = action {
            game.apply_action(actor, action).unwrap();
        }
        return;
    }
    panic!("no seed produced a testable Super Detrital Repositioner scenario");
}

#[test]
fn cryptozootic_expander_moves_a_hidden_hand_card_into_a_deal() {
    for seed in 0..40u64 {
        let mut game = new_test_game(4, seed);
        let Some(actor) = stash_cards_of_kind_in_collection(&mut game, CardKind::Thingamabob(ThingamabobKind::CryptozooticExpander), 1) else { continue };

        auto_play_deal_offers(&mut game);
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(game.turn_leader())[0].clone();
            game.apply_action(game.turn_leader(), peek).unwrap();
        }
        advance_thingamabob_window_to_player(&mut game, actor);
        let Some(action) = find_thingamabob_action(&game, actor, |_| true) else { continue };
        let Action::PlayThingamabob { params: ThingamabobParams::CryptozooticExpander { target_deal, .. }, .. } = action.clone() else { continue };
        let hand_before = game.player_hand(actor).len();
        game.apply_action(actor, action).unwrap();
        assert_eq!(game.player_hand(actor).len(), hand_before - 1, "the hand card moved out of hand");
        let _ = target_deal;
        return;
    }
    panic!("no seed produced a testable Cryptozooptic Expander scenario");
}

#[test]
fn spectroelectric_optimeter_reveals_a_hidden_deal_card() {
    for seed in 0..40u64 {
        let mut game = new_test_game(4, seed);
        let Some(actor) = stash_cards_of_kind_in_collection(&mut game, CardKind::Thingamabob(ThingamabobKind::SpectroelectricOptimeter), 1) else { continue };

        auto_play_deal_offers(&mut game);
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(game.turn_leader())[0].clone();
            game.apply_action(game.turn_leader(), peek).unwrap();
        }
        advance_thingamabob_window_to_player(&mut game, actor);
        let Some(action) = find_thingamabob_action(&game, actor, |_| true) else { continue };
        let Action::PlayThingamabob { params: ThingamabobParams::SpectroelectricOptimeter { target_deal, target_card }, .. } = action.clone() else { continue };
        let events = game.apply_action(actor, action).unwrap();
        assert!(events.iter().any(|e| matches!(e, Event::CardRevealed { seller, card } if *seller == target_deal && *card == target_card)));
        return;
    }
    panic!("no seed produced a testable Spectroelectric Optimeter scenario");
}
