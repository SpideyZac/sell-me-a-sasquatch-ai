//! `pin_kind` / `kind_supply` / `hidden_cards_in_deal`: the primitive that
//! lets a live-tracking caller overwrite a card's randomly-dealt kind with
//! the kind actually revealed at a physical table, so every other rule
//! (set completion, tokens, discards) then runs on truth.

mod common;

use common::new_test_game;
use sasquatch_engine::card::{CardKind, Tier};
use sasquatch_engine::game::PinError;

#[test]
fn pin_kind_overwrites_and_consumes_supply() {
    let mut game = new_test_game(4, 1);
    let card = game.player_hand(0)[0];
    let before = *game.kind_supply().get(&CardKind::Creature(Tier::Giant)).unwrap();

    game.pin_kind(card, CardKind::Creature(Tier::Giant)).unwrap();

    assert_eq!(game.card_kind(card), Some(CardKind::Creature(Tier::Giant)));
    assert!(game.is_pinned(card));
    assert_eq!(*game.kind_supply().get(&CardKind::Creature(Tier::Giant)).unwrap(), before - 1);
}

#[test]
fn pin_kind_rejects_double_pinning_the_same_card() {
    let mut game = new_test_game(4, 2);
    let card = game.player_hand(0)[0];
    game.pin_kind(card, CardKind::Creature(Tier::Tiny)).unwrap();
    let err = game.pin_kind(card, CardKind::Creature(Tier::Big)).unwrap_err();
    assert_eq!(err, PinError::AlreadyPinned(card));
}

#[test]
fn pin_kind_rejects_exceeding_the_deck_composition() {
    let mut game = new_test_game(4, 3);
    let supply = *game.kind_supply().get(&CardKind::Creature(Tier::Giant)).unwrap();
    // Pin every hand card to Giant until supply for Giant is exhausted, then
    // the next attempt must fail rather than silently manufacturing an extra
    // Giant beyond what the deck actually contains.
    let all_cards: Vec<_> = (0..4).flat_map(|p| game.player_hand(p).to_vec()).collect();
    let mut pinned = 0u32;
    for &card in all_cards.iter() {
        if pinned == supply {
            break;
        }
        game.pin_kind(card, CardKind::Creature(Tier::Giant)).unwrap();
        pinned += 1;
    }
    assert_eq!(*game.kind_supply().get(&CardKind::Creature(Tier::Giant)).unwrap(), supply - pinned);
    if pinned == supply {
        // one more distinct, not-yet-pinned card should now be rejected
        if let Some(&extra) = game.player_hand(0).iter().find(|c| !all_cards.contains(c)) {
            let err = game.pin_kind(extra, CardKind::Creature(Tier::Giant)).unwrap_err();
            assert_eq!(err, PinError::NoSupplyRemaining);
        }
    }
}

#[test]
fn hidden_cards_in_deal_lists_only_unrevealed_cards() {
    let mut game = new_test_game(4, 4);
    let buyer = game.turn_leader();
    let seller = (0..4).find(|&p| p != buyer).unwrap();

    // Drive to this seller's submit step.
    while game.active_players() != vec![seller] || game.current_phase() != "deal_offer_submit" {
        let player = game.active_players()[0];
        let action = game.legal_actions(player).into_iter().next().unwrap();
        game.apply_action(player, action).unwrap();
    }
    let hand = game.player_hand(seller).to_vec();
    let cards = [hand[0], hand[1], hand[2]];
    use sasquatch_engine::action::Action;
    game.apply_action(seller, Action::SubmitDeal { cards }).unwrap();

    let hidden = game.hidden_cards_in_deal(seller);
    assert_eq!(hidden.len(), 3, "nothing revealed yet");

    // Buyer mode always needs an immediate reveal for every seller.
    game.apply_action(seller, Action::RevealCard { card: cards[0] }).unwrap();
    let hidden_after = game.hidden_cards_in_deal(seller);
    assert_eq!(hidden_after.len(), 2);
    assert!(!hidden_after.contains(&cards[0]));
}
