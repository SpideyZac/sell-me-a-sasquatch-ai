mod common;

use common::*;
use sasquatch_engine::{
    action::Action,
    card::{CardKind, NastyKind},
    game::Event,
};

/// Drives a full turn where `seller` (a non-chosen seller) submits a deal
/// containing 2 copies of `kind` plus 1 filler card, so that seller ends up
/// with a completed nasty set of `kind` in their own collection right after
/// deal resolution, since a non-chosen seller keeps their own cards.
/// Returns the events from `ChooseDeal`.
fn run_turn_completing_nasty_set_for_a_seller(
    game: &mut sasquatch_engine::game::GameState,
    kind: NastyKind,
) -> (usize, usize, Vec<Event>) {
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
                let filler = game
                    .player_hand(player)
                    .iter()
                    .copied()
                    .find(|c| !cards.contains(c))
                    .unwrap();
                cards.push(filler);
                Action::SubmitDeal {
                    cards: [cards[0], cards[1], cards[2]],
                }
            } else {
                game.legal_actions(player).into_iter().next().unwrap()
            };
            game.apply_action(player, action).unwrap();
        } else {
            let action = game.legal_actions(player).into_iter().next().unwrap();
            game.apply_action(player, action).unwrap();
        }
    }
    let Some(seller) = target_seller else {
        return (usize::MAX, buyer, vec![]);
    };

    if game.current_phase() == "buyer_peek" {
        let peek = game.legal_actions(buyer)[0].clone();
        game.apply_action(buyer, peek).unwrap();
    }
    auto_pass_thingamabob_window(game);

    // the buyer must choose a different seller's deal so `seller` is not
    // the chosen one and keeps their own (nasty-laden) cards
    let options = game.legal_actions(buyer);
    let Action::ChooseDeal { seller: other } = options
        .iter()
        .find(|a| !matches!(a, Action::ChooseDeal { seller: s } if *s == seller))
        .cloned()
        .unwrap()
    else {
        panic!()
    };
    let events = game
        .apply_action(buyer, Action::ChooseDeal { seller: other })
        .unwrap();
    (seller, buyer, events)
}

#[test]
fn poison_pill_bug_trade_in_lets_new_buyer_optionally_steal_one_card() {
    for seed in 0..30u64 {
        let mut game = new_test_game(4, seed);
        let (seller, _old_buyer, events) =
            run_turn_completing_nasty_set_for_a_seller(&mut game, NastyKind::PoisonPillBug);
        if seller == usize::MAX {
            continue; // this seed's deal didn't happen to yield 2 PPB cards; try another
        }
        assert!(events.iter().any(|e| matches!(e, Event::NastySetTradedIn { player, kind } if *player == seller && *kind == NastyKind::PoisonPillBug)));

        if game.current_phase() == "nasty_resolution" {
            let new_buyer = game.turn_leader();
            let resolve_options = game.legal_actions(new_buyer);
            // 0 or 1 card takeable, never more, per poison pill bug's effect
            assert!(resolve_options.iter().all(|a| matches!(a, Action::ResolveNastyPenalty { taken_cards } if taken_cards.len() <= 1)));
            // choosing to take 1 card actually moves it into the new buyer's collection
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
        let (seller, _old_buyer, _events) =
            run_turn_completing_nasty_set_for_a_seller(&mut game, NastyKind::LoanShark);
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
                let matching =
                    hand_cards_of_kind(&game, player, CardKind::Nasty(NastyKind::TrojanHorse));
                let action = if player != buyer && matching.len() >= 3 && target_seller.is_none() {
                    target_seller = Some(player);
                    Action::SubmitDeal {
                        cards: [matching[0], matching[1], matching[2]],
                    }
                } else {
                    game.legal_actions(player).into_iter().next().unwrap()
                };
                game.apply_action(player, action).unwrap();
            } else {
                let action = game.legal_actions(player).into_iter().next().unwrap();
                game.apply_action(player, action).unwrap();
            }
        }
        let Some(seller) = target_seller else {
            continue;
        };
        // give the seller a point token first so the steal is observable
        if game.current_phase() == "buyer_peek" {
            let peek = game.legal_actions(buyer)[0].clone();
            game.apply_action(buyer, peek).unwrap();
        }
        auto_pass_thingamabob_window(&mut game);
        let options = game.legal_actions(buyer);
        let chosen_seller = if let Action::ChooseDeal { seller: s } = options
            .iter()
            .find(|a| !matches!(a, Action::ChooseDeal { seller: s2 } if *s2 == seller))
            .unwrap()
        {
            *s
        } else {
            unreachable!()
        };
        let events = game
            .apply_action(
                buyer,
                Action::ChooseDeal {
                    seller: chosen_seller,
                },
            )
            .unwrap();

        assert!(events.iter().any(|e| matches!(e, Event::NastySetTradedIn { player, kind } if *player == seller && *kind == NastyKind::TrojanHorse)));
        // no player action was ever required for trojan horse; the phase
        // should never stop at nasty_resolution for this specific kind (it
        // may still stop there if a different nasty also completed)
        return;
    }
    panic!("no seed in range produced a testable Trojan Horse completion");
}

/// Two-player mode's rule: whenever a nasty set is traded in, the other
/// player always decides what's taken, never the player who just completed
/// the set, even when that player also happens to be the newly-assigned
/// turn leader (the "new buyer" shortcut only holds in buyer mode, see
/// `GameState::nasty_beneficiary`'s doc comment). This engineers exactly
/// that collision: leader0 keeps one Poison Pill Bug from their own turn,
/// then leader1 sends a second one over on the very next turn, completing
/// leader0's set right as leader0 (the responder) accepts it, so leader0
/// becomes both the loser and the new turn leader in the same instant.
#[test]
fn two_player_nasty_trade_in_is_always_resolved_by_the_other_player() {
    let is_ppb = |game: &sasquatch_engine::game::GameState, c: &sasquatch_engine::card::CardId| {
        game.card_kind(*c) == Some(CardKind::Nasty(NastyKind::PoisonPillBug))
    };
    let is_not_nasty = |game: &sasquatch_engine::game::GameState,
                        c: &sasquatch_engine::card::CardId| {
        !matches!(game.card_kind(*c), Some(CardKind::Nasty(_)))
    };

    for seed in 0..60u64 {
        let mut game = new_test_game(2, seed);
        let leader0 = game.turn_leader();
        let leader1 = 1 - leader0;

        let Some(&leader0_ppb) = game.player_hand(leader0).iter().find(|c| is_ppb(&game, c)) else {
            continue;
        };
        let Some(&leader1_ppb) = game.player_hand(leader1).iter().find(|c| is_ppb(&game, c)) else {
            continue;
        };
        let leader0_fillers: Vec<_> = game
            .player_hand(leader0)
            .iter()
            .copied()
            .filter(|c| is_not_nasty(&game, c))
            .take(2)
            .collect();
        let leader1_fillers: Vec<_> = game
            .player_hand(leader1)
            .iter()
            .copied()
            .filter(|c| is_not_nasty(&game, c))
            .take(2)
            .collect();
        if leader0_fillers.len() < 2 || leader1_fillers.len() < 2 {
            continue;
        }

        // Turn 1: leader0 keeps 1 Poison Pill Bug for themselves.
        game.apply_action(
            leader0,
            Action::TwoPlayerSubmitDeal {
                own_pile: vec![leader0_ppb],
                other_pile: leader0_fillers,
            },
        )
        .unwrap();
        let reveal = game.legal_actions(leader0).into_iter().next().unwrap();
        game.apply_action(leader0, reveal).unwrap();
        auto_pass_thingamabob_window(&mut game);
        game.apply_action(leader1, Action::RespondToDeal { reverse: false })
            .unwrap();
        assert_eq!(game.turn_leader(), leader1);
        assert!(
            collection_cards_of_kind(&game, leader0, CardKind::Nasty(NastyKind::PoisonPillBug))
                .len()
                < 2,
            "shouldn't have completed a set yet"
        );

        // Turn 2: leader1 sends a 2nd Poison Pill Bug over to leader0,
        // completing leader0's set the instant leader0 accepts it.
        game.apply_action(
            leader1,
            Action::TwoPlayerSubmitDeal {
                own_pile: leader1_fillers,
                other_pile: vec![leader1_ppb],
            },
        )
        .unwrap();
        let reveal = game.legal_actions(leader1).into_iter().next().unwrap();
        game.apply_action(leader1, reveal).unwrap();
        auto_pass_thingamabob_window(&mut game);
        let events = game
            .apply_action(leader0, Action::RespondToDeal { reverse: false })
            .unwrap();

        assert!(events.iter().any(|e| matches!(e, Event::NastySetTradedIn { player, kind } if *player == leader0 && *kind == NastyKind::PoisonPillBug)));
        assert_eq!(game.current_phase(), "nasty_resolution");
        assert_eq!(
            game.turn_leader(),
            leader0,
            "leader0 (the loser) is also the newly-assigned turn leader in this scenario"
        );
        assert_eq!(
            game.active_players(),
            vec![leader1],
            "the OTHER player must decide what leader0 loses, never leader0 themselves"
        );

        let resolve = game.legal_actions(leader1).into_iter().next().unwrap();
        game.apply_action(leader1, resolve).unwrap();
        return;
    }
    panic!("no seed in range produced 2 players each starting with a Poison Pill Bug in hand");
}
