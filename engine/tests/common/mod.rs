//! Shared test helpers. Not a test binary itself (subdirectory convention).
//! Each test binary that includes this module only uses a subset of these
//! helpers, so `dead_code` warnings here are expected noise, not signal.
#![allow(dead_code)]

use sasquatch_engine::action::{Action, ThingamabobParams};
use sasquatch_engine::card::{CardId, CardKind, PlayerId};
use sasquatch_engine::deck::DeckConfig;
use sasquatch_engine::game::{Event, GameState};

/// A deck small enough for fast, deterministic tests but with a rich mix of
/// every card kind, all set sizes divisible so trade-ins are easy to reason
/// about. Not the confirmed physical-game deck (`configs/deck.toml`) - see
/// PROMPT.md §2.5's explicit call-out that smaller decks are fine for tests.
pub const TEST_DECK_TOML: &str = r#"
    [[creature_tiers]]
    tier = "Giant"
    set_size = 1
    copies = 12

    [[creature_tiers]]
    tier = "Big"
    set_size = 2
    copies = 24

    [[creature_tiers]]
    tier = "Medium"
    set_size = 3
    copies = 27

    [[creature_tiers]]
    tier = "Tiny"
    set_size = 4
    copies = 24

    [[nasties]]
    name = "Poison Pill Bug"
    set_size = 2
    copies = 24
    effect = "buyer_may_steal_up_to_1_card"

    [[nasties]]
    name = "Loan Shark"
    set_size = 2
    copies = 24
    effect = "buyer_may_steal_up_to_2_cards"

    [[nasties]]
    name = "Trojan Horse"
    set_size = 3
    copies = 27
    effect = "buyer_steals_1_point_token"

    [[thingamabobs]]
    name = "Platonic Isolator"
    copies = 18
    effect = "steal_point_token_from_richer_player"

    [[thingamabobs]]
    name = "Detrital Repositioner"
    copies = 18
    effect = "remove_1_card_from_1_deal"

    [[thingamabobs]]
    name = "Super Detrital Repositioner"
    copies = 18
    effect = "remove_up_to_2_cards_from_1_or_2_deals"

    [[thingamabobs]]
    name = "Cryptozooptic Expander"
    copies = 18
    effect = "add_hidden_hand_card_to_deal"

    [[thingamabobs]]
    name = "Spectroelectric Optimeter"
    copies = 18
    effect = "reveal_1_card_in_a_deal"
"#;

pub fn new_test_game(num_players: usize, seed: u64) -> GameState {
    let deck = DeckConfig::from_toml_str(TEST_DECK_TOML).unwrap();
    GameState::new(num_players, deck, seed).unwrap()
}

/// Cards of the given kind currently in `player`'s hand.
pub fn hand_cards_of_kind(game: &GameState, player: PlayerId, kind: CardKind) -> Vec<CardId> {
    game.player_hand(player).iter().copied().filter(|&c| game.card_kind(c) == Some(kind)).collect()
}

pub fn collection_cards_of_kind(game: &GameState, player: PlayerId, kind: CardKind) -> Vec<CardId> {
    game.player_collection(player).iter().copied().filter(|&c| game.card_kind(c) == Some(kind)).collect()
}

/// Drives the deal-offer micro-turns for every entry automatically: submits
/// the first 3 hand cards, reveals the first submitted card whenever a
/// reveal is required. Returns once the deal-offer phase is fully done.
pub fn auto_play_deal_offers(game: &mut GameState) -> Vec<Event> {
    let mut events = Vec::new();
    while game.current_phase() == "deal_offer_submit" || game.current_phase() == "deal_offer_reveal" {
        let players = game.active_players();
        let player = players[0];
        let actions = game.legal_actions(player);
        let action = actions.into_iter().next().expect("no legal deal-offer action");
        events.extend(game.apply_action(player, action).unwrap());
    }
    events
}

/// Passes the Thingamabob window for every player until it closes.
pub fn auto_pass_thingamabob_window(game: &mut GameState) -> Vec<Event> {
    let mut events = Vec::new();
    while game.current_phase() == "thingamabob_window" {
        let player = game.active_players()[0];
        events.extend(game.apply_action(player, Action::PassThingamabobWindow).unwrap());
    }
    events
}

/// Drives one full turn from the start of `deal_offer_submit`, having the
/// first eligible non-buyer player with `count` (<=3) hand cards of `kind`
/// submit them (padded with filler up to 3) as their deal, then having the
/// Buyer choose a *different* seller's deal so this player keeps their own
/// cards - including the target ones - in their Collection (§2.3 step 5).
/// Returns the seller who ended up stashing them, if any player qualified.
pub fn stash_cards_of_kind_in_collection(game: &mut GameState, kind: CardKind, count: usize) -> Option<PlayerId> {
    assert!(game.current_phase() == "deal_offer_submit", "must be called at the start of a turn");
    let buyer = game.turn_leader();
    let mut target_seller = None;
    while game.current_phase().starts_with("deal_offer") {
        let player = game.active_players()[0];
        if game.current_phase() == "deal_offer_submit" {
            let matching = hand_cards_of_kind(game, player, kind);
            let action = if player != buyer && matching.len() >= count && target_seller.is_none() {
                target_seller = Some(player);
                let mut cards: Vec<_> = matching.into_iter().take(count).collect();
                while cards.len() < 3 {
                    let filler = game.player_hand(player).iter().copied().find(|c| !cards.contains(c)).unwrap();
                    cards.push(filler);
                }
                cards.truncate(3);
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
    // Regardless of whether a qualifying seller was found, always finish
    // driving the rest of the turn (peek / Thingamabob / commit / nasty
    // drain) so the game always lands back at the start of a fresh turn's
    // deal offer (or game_over) before returning - callers may retry in a
    // loop, and the precondition assert above depends on this.
    if game.current_phase() == "buyer_peek" {
        let peek = game.legal_actions(buyer)[0].clone();
        game.apply_action(buyer, peek).unwrap();
    }
    auto_pass_thingamabob_window(game);
    if game.current_phase() == "buyer_chooses_deal" {
        let options = game.legal_actions(buyer);
        let other = target_seller
            .and_then(|seller| options.iter().find_map(|a| if let Action::ChooseDeal { seller: s } = a { (*s != seller).then_some(*s) } else { None }))
            .unwrap_or_else(|| if let Action::ChooseDeal { seller } = options[0] { seller } else { unreachable!() });
        game.apply_action(buyer, Action::ChooseDeal { seller: other }).unwrap();
    } else if game.current_phase() == "respond_to_deal" {
        let responder = game.active_players()[0];
        game.apply_action(responder, Action::RespondToDeal { reverse: false }).unwrap();
    }
    // Drain any pending Nasty-penalty choices (decline every time) so the
    // caller always lands back at the start of a fresh turn's deal offer.
    while game.current_phase() == "nasty_resolution" {
        let resolver = game.active_players()[0];
        game.apply_action(resolver, Action::ResolveNastyPenalty { taken_cards: vec![] }).unwrap();
    }
    target_seller
}

/// Passes for whichever players are currently up until it's `target`'s turn
/// in the Thingamabob window (without passing for `target` itself).
pub fn advance_thingamabob_window_to_player(game: &mut GameState, target: PlayerId) {
    while game.current_phase() == "thingamabob_window" && game.active_players()[0] != target {
        let player = game.active_players()[0];
        game.apply_action(player, Action::PassThingamabobWindow).unwrap();
    }
}

/// Finds a `PlayThingamabob` action in `player`'s legal actions matching a
/// predicate over its params, for tests that want to exercise one specific
/// effect without hand-crafting the exact `CardId`.
pub fn find_thingamabob_action(
    game: &GameState,
    player: PlayerId,
    mut pred: impl FnMut(&ThingamabobParams) -> bool,
) -> Option<Action> {
    game.legal_actions(player).into_iter().find(|a| matches!(a, Action::PlayThingamabob { params, .. } if pred(params)))
}
