//! `GameState`: the full rules engine (§2, §3.2). Pure data + one entry
//! point, `apply_action`, that returns structured `Event`s. No I/O.

use crate::action::{Action, ThingamabobParams};
use crate::card::{Card, CardId, CardKind, NastyKind, PlayerId, Tier, ThingamabobEffect, ThingamabobKind, NastyEffect};
use crate::deal::Deal;
use crate::deck::{Catalog, DeckConfig};
use crate::phase::{DealOfferEntry, DealOfferState, DealOfferStep, NastyResolutionState, PendingNasty, Phase, ThingamabobWindowState};
use crate::player::{try_take_cards, try_take_point_tokens, Player};
use crate::rng::GameRng;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

pub const HAND_SIZE: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    Buyer,
    TwoPlayer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    DealSubmitted { player: PlayerId },
    CardRevealed { seller: PlayerId, card: CardId },
    BuyerPeeked { target_seller: PlayerId, card: CardId },
    ThingamabobPlayed { player: PlayerId, card: CardId, kind: ThingamabobKind },
    ThingamabobWindowClosed,
    DealChosen { buyer: PlayerId, seller: PlayerId },
    DealResponded { active: PlayerId, reverse: bool },
    CardsAwarded { player: PlayerId, cards: Vec<CardId> },
    CardsDiscarded { cards: Vec<CardId> },
    NastySetTradedIn { player: PlayerId, kind: NastyKind },
    PointTokenStolen { from: PlayerId, to: PlayerId, amount: u32 },
    CardsStolen { from: PlayerId, to: PlayerId, cards: Vec<CardId> },
    CreatureSetTradedIn { player: PlayerId, tier: Tier, tokens_gained: u32 },
    HandRefilled { player: PlayerId, drawn: usize },
    DrawPileReshuffledFromDiscard,
    TurnLeaderPassed { new_leader: PlayerId },
    GameOver { winner: PlayerId },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RulesError {
    #[error("player {0} is not currently allowed to act")]
    NotActivePlayer(PlayerId),
    #[error("wrong action for current phase ({0})")]
    WrongPhaseAction(&'static str),
    #[error("card {0} not found in player's hand")]
    CardNotInHand(CardId),
    #[error("card {0} not found in player's collection")]
    CardNotInCollection(CardId),
    #[error("submitted deal must be 3 distinct cards")]
    DealMustBeThreeDistinctCards,
    #[error("revealed card must be one of the submitted deal cards")]
    RevealCardNotInDeal,
    #[error("no such active deal for seller {0}")]
    NoSuchDeal(PlayerId),
    #[error("card {0} is not a still-hidden card in that deal")]
    CardNotHiddenInDeal(CardId),
    #[error("card {0} is not a thingamabob")]
    NotAThingamabob(CardId),
    #[error("invalid thingamabob params for this card's effect")]
    InvalidThingamabobParams,
    #[error("platonic isolator target must currently have strictly more point tokens")]
    IsolatorTargetNotRicher,
    #[error("removal targets exceed this card's max cards/deals")]
    TooManyRemovals,
    #[error("duplicate target in removal list")]
    DuplicateRemovalTarget,
    #[error("nasty penalty resolution requested more cards than the effect allows")]
    TooManyCardsTaken,
    #[error("duplicate card in nasty penalty resolution")]
    DuplicateCardTaken,
    #[error("the game has already ended")]
    GameAlreadyOver,
}

/// Errors from "live-tracking" a physical game (§below `pin_kind`): pinning a
/// card's kind to match what was actually revealed at the table, instead of
/// the kind randomly assigned at deal-shuffle time.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PinError {
    #[error("card {0} does not exist in this game")]
    UnknownCard(CardId),
    #[error("card {0} was already pinned to a kind")]
    AlreadyPinned(CardId),
    #[error("no cards of that kind remain unaccounted for in the deck")]
    NoSupplyRemaining,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SetupError {
    #[error("num_players must be between 2 and 6, got {0}")]
    InvalidPlayerCount(usize),
    #[error("deck has {have} cards, not enough to deal {need} initial hand cards")]
    NotEnoughCardsToDeal { have: usize, need: usize },
}

/// A player's own hand card, with kind resolved for convenience (full
/// information - it's their own hand).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedDeal {
    pub seller: PlayerId,
    pub revealed_cards: Vec<CardId>,
    pub num_hidden: usize,
}

/// Filtered, player-specific view of `GameState` (§3.2 "full information
/// internally, filtered externally"). Never exposes other players' hands or
/// still-hidden deal cards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub player: PlayerId,
    pub phase: &'static str,
    pub turn_leader: PlayerId,
    pub own_hand: Vec<CardId>,
    pub collections: Vec<Vec<CardId>>,
    pub point_tokens: Vec<u32>,
    pub deals: Vec<ObservedDeal>,
    pub draw_pile_len: usize,
    pub discard_pile_len: usize,
    pub winner: Option<PlayerId>,
}

pub struct GameState {
    catalog: Catalog,
    cards: HashMap<CardId, Card>,
    mode: GameMode,
    num_players: usize,
    win_threshold: u32,
    players: Vec<Player>,
    draw_pile: Vec<CardId>,
    discard_pile: Vec<CardId>,
    deals: Vec<Deal>,
    turn_leader: PlayerId,
    phase: Phase,
    rng: GameRng,
    /// Live-tracking support (`pin_kind`): remaining not-yet-pinned supply of
    /// each kind, seeded from the deck's true composition. Cards start with
    /// an arbitrary (random) kind from the shuffle; `pin_kind` overwrites it
    /// once, when the card is actually revealed at the table, so all of the
    /// engine's own bookkeeping (set completion, tokens, discards) runs on
    /// truth instead of the random deal.
    kind_supply: HashMap<CardKind, u32>,
    pinned: HashSet<CardId>,
}

impl GameState {
    pub fn new(num_players: usize, deck: DeckConfig, seed: u64) -> Result<GameState, SetupError> {
        if !(2..=6).contains(&num_players) {
            return Err(SetupError::InvalidPlayerCount(num_players));
        }
        let needed = num_players * HAND_SIZE;
        if deck.cards.len() < needed {
            return Err(SetupError::NotEnoughCardsToDeal { have: deck.cards.len(), need: needed });
        }

        let mut rng = GameRng::from_seed(seed);
        let mut ids: Vec<CardId> = deck.cards.iter().map(|c| c.id).collect();
        rng.shuffle(&mut ids);

        let cards: HashMap<CardId, Card> = deck.cards.into_iter().map(|c| (c.id, c)).collect();
        let mut kind_supply: HashMap<CardKind, u32> = HashMap::new();
        for c in cards.values() {
            *kind_supply.entry(c.kind).or_insert(0) += 1;
        }

        let mut players: Vec<Player> = (0..num_players).map(|_| Player::new()).collect();
        let mut cursor = 0usize;
        for p in players.iter_mut() {
            p.hand = ids[cursor..cursor + HAND_SIZE].to_vec();
            cursor += HAND_SIZE;
        }
        let draw_pile = ids[cursor..].to_vec();

        let mode = if num_players == 2 { GameMode::TwoPlayer } else { GameMode::Buyer };
        let win_threshold = match (mode, num_players) {
            (GameMode::TwoPlayer, _) => 5,
            (GameMode::Buyer, 3 | 4) => 4,
            (GameMode::Buyer, _) => 5,
        };
        let turn_leader = rng.gen_index(num_players);

        let mut game = GameState {
            catalog: deck.catalog,
            cards,
            mode,
            num_players,
            win_threshold,
            players,
            draw_pile,
            discard_pile: Vec::new(),
            deals: Vec::new(),
            turn_leader,
            phase: Phase::GameOver { winner: 0 }, // placeholder, overwritten by start_new_turn
            rng,
            kind_supply,
            pinned: HashSet::new(),
        };
        game.start_new_turn();
        Ok(game)
    }

    pub fn num_players(&self) -> usize {
        self.num_players
    }

    pub fn mode(&self) -> GameMode {
        self.mode
    }

    pub fn turn_leader(&self) -> PlayerId {
        self.turn_leader
    }

    pub fn win_threshold(&self) -> u32 {
        self.win_threshold
    }

    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => Some(winner),
            _ => None,
        }
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.phase, Phase::GameOver { .. })
    }

    pub fn current_phase(&self) -> &'static str {
        self.phase.name()
    }

    pub fn card_kind(&self, id: CardId) -> Option<CardKind> {
        self.cards.get(&id).map(|c| c.kind)
    }

    pub fn card_name(&self, id: CardId) -> Option<&str> {
        self.cards.get(&id).map(|c| c.name.as_str())
    }

    pub fn player_hand(&self, player: PlayerId) -> &[CardId] {
        &self.players[player].hand
    }

    pub fn player_collection(&self, player: PlayerId) -> &[CardId] {
        &self.players[player].collection
    }

    pub fn player_point_tokens(&self, player: PlayerId) -> u32 {
        self.players[player].point_tokens
    }

    /// Still-hidden card ids in `seller`'s active deal (empty if no such
    /// deal). Bypasses the usual observation privacy filtering - only
    /// meant for a live-tracking caller who *is* the sole source of truth
    /// for what these cards really are, not an in-game player peeking.
    pub fn hidden_cards_in_deal(&self, seller: PlayerId) -> Vec<CardId> {
        self.deal_for_seller(seller).map(|d| d.hidden_cards().collect()).unwrap_or_default()
    }

    pub fn is_pinned(&self, card: CardId) -> bool {
        self.pinned.contains(&card)
    }

    /// Remaining not-yet-pinned supply per kind (how many more cards of that
    /// kind could still be truthfully assigned via `pin_kind`).
    pub fn kind_supply(&self) -> &HashMap<CardKind, u32> {
        &self.kind_supply
    }

    /// For live-tracking a physical game (see `kind_supply` field docs):
    /// overwrites `card`'s kind to `kind`, matching what was actually
    /// revealed at the table, and consumes one unit of that kind's
    /// remaining supply so the deck's true composition (from `deck.toml`)
    /// is never exceeded. Every downstream rule (set completion, Nasty/
    /// Thingamabob effects, tokens, discards) reads `card_kind` normally
    /// afterward, so once pinned a card behaves exactly like a "real" one.
    pub fn pin_kind(&mut self, card: CardId, kind: CardKind) -> Result<(), PinError> {
        if !self.cards.contains_key(&card) {
            return Err(PinError::UnknownCard(card));
        }
        if self.pinned.contains(&card) {
            return Err(PinError::AlreadyPinned(card));
        }
        let remaining = self.kind_supply.get_mut(&kind).ok_or(PinError::NoSupplyRemaining)?;
        if *remaining == 0 {
            return Err(PinError::NoSupplyRemaining);
        }
        *remaining -= 1;
        let entry = self.cards.get_mut(&card).expect("checked above");
        entry.kind = kind;
        // `name` is a separate field, set once at deck-build time for
        // whatever kind the card *originally, randomly* got - it must be
        // overwritten too, or `card_name()` keeps returning the old
        // (now-wrong) flavor name for Nasties/Thingamabobs, where display
        // falls back to it (Creatures are shown by tier only, so this only
        // ever bit Nasty/Thingamabob cards - see `card_display.display_name`
        // on the Python side).
        entry.name = match kind {
            CardKind::Creature(tier) => tier.to_string(),
            CardKind::Nasty(k) => k.name().to_string(),
            CardKind::Thingamabob(k) => k.name().to_string(),
        };
        self.pinned.insert(card);
        Ok(())
    }

    /// Total card count across hand+draw+discard+deals+collections. Used by
    /// property tests to assert conservation (§5).
    pub fn total_card_count(&self) -> usize {
        let mut n = self.draw_pile.len() + self.discard_pile.len();
        for p in &self.players {
            n += p.hand.len() + p.collection.len();
        }
        for d in &self.deals {
            n += d.cards.len();
        }
        n
    }

    // turn setup

    /// Table order of every other player starting immediately after `from`.
    fn table_order_after(&self, from: PlayerId) -> Vec<PlayerId> {
        (1..self.num_players).map(|offset| (from + offset) % self.num_players).collect()
    }

    fn start_new_turn(&mut self) {
        self.deals.clear();
        let entries = match self.mode {
            GameMode::Buyer => self
                .table_order_after(self.turn_leader)
                .into_iter()
                .map(|p| DealOfferEntry { player: p, needs_reveal: true })
                .collect(),
            GameMode::TwoPlayer => {
                let other = 1 - self.turn_leader;
                vec![
                    DealOfferEntry { player: self.turn_leader, needs_reveal: true },
                    DealOfferEntry { player: other, needs_reveal: false },
                ]
            }
        };
        self.phase = Phase::DealOffer(DealOfferState { entries, idx: 0, step: DealOfferStep::AwaitingSubmit });
    }

    fn start_thingamabob_window(&mut self) {
        let order = match self.mode {
            GameMode::Buyer => {
                let mut o = vec![self.turn_leader];
                o.extend(self.table_order_after(self.turn_leader));
                o
            }
            GameMode::TwoPlayer => vec![self.turn_leader, 1 - self.turn_leader],
        };
        self.phase = Phase::ThingamabobWindow(ThingamabobWindowState { order, turn_idx: 0, consecutive_passes: 0 });
    }

    // active player / legality

    pub fn active_players(&self) -> Vec<PlayerId> {
        match &self.phase {
            Phase::DealOffer(s) => {
                if s.is_done() {
                    vec![]
                } else {
                    vec![s.current().player]
                }
            }
            Phase::BuyerPeek => vec![self.turn_leader],
            Phase::ThingamabobWindow(s) => vec![s.current_player()],
            Phase::BuyerChoosesDeal => vec![self.turn_leader],
            Phase::RespondToDeal => vec![1 - self.turn_leader],
            Phase::NastyResolution(s) => vec![self.turn_leader_for_pending(s)],
            Phase::GameOver { .. } => vec![],
        }
    }

    fn turn_leader_for_pending(&self, _s: &NastyResolutionState) -> PlayerId {
        // The resolver is always the new turn leader - the Buyer marker
        // (or, in 2-player mode, the responder) has already been assigned
        // to `self.turn_leader` at deal-resolution time (§2.4 note).
        self.turn_leader
    }

    fn deal_for_seller(&self, seller: PlayerId) -> Option<&Deal> {
        self.deals.iter().find(|d| d.seller == seller)
    }

    fn deal_for_seller_mut(&mut self, seller: PlayerId) -> Option<&mut Deal> {
        self.deals.iter_mut().find(|d| d.seller == seller)
    }

    pub fn legal_actions(&self, player: PlayerId) -> Vec<Action> {
        if !self.active_players().contains(&player) {
            return vec![];
        }
        match &self.phase {
            Phase::DealOffer(s) => {
                if s.step == DealOfferStep::AwaitingSubmit {
                    combinations(&self.players[player].hand, 3)
                        .into_iter()
                        .map(|c| Action::SubmitDeal { cards: [c[0], c[1], c[2]] })
                        .collect()
                } else {
                    let deal = self.deal_for_seller(player).expect("submitted deal must exist");
                    deal.cards.iter().map(|c| Action::RevealCard { card: c.card }).collect()
                }
            }
            Phase::BuyerPeek => self.deals.iter().map(|d| Action::BuyerPeek { target_seller: d.seller }).collect(),
            Phase::ThingamabobWindow(_) => self.legal_thingamabob_actions(player),
            Phase::BuyerChoosesDeal => self.deals.iter().map(|d| Action::ChooseDeal { seller: d.seller }).collect(),
            Phase::RespondToDeal => vec![Action::RespondToDeal { reverse: false }, Action::RespondToDeal { reverse: true }],
            Phase::NastyResolution(s) => {
                let max = match self.catalog.nasty_effect(s.pending.kind) {
                    NastyEffect::BuyerMayStealUpToNCards(max) => max,
                    NastyEffect::BuyerStealsOnePointToken => 0, // never queued (auto-resolved)
                };
                let target_collection = &self.players[s.pending.player].collection;
                (0..=max)
                    .flat_map(|k| combinations(target_collection, k as usize))
                    .map(|cards| Action::ResolveNastyPenalty { taken_cards: cards })
                    .collect()
            }
            Phase::GameOver { .. } => vec![],
        }
    }

    fn legal_thingamabob_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = vec![Action::PassThingamabobWindow];
        let collection = self.players[player].collection.clone();
        for card in collection {
            let kind = match self.cards[&card].kind {
                CardKind::Thingamabob(k) => k,
                _ => continue,
            };
            let effect = self.catalog.thingamabob_effect(kind);
            match effect {
                ThingamabobEffect::StealPointTokenFromRicherPlayer => {
                    let my_tokens = self.players[player].point_tokens;
                    for (target, p) in self.players.iter().enumerate() {
                        if target != player && p.point_tokens > my_tokens {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::PlatonicIsolator { target_player: target },
                            });
                        }
                    }
                }
                ThingamabobEffect::RemoveCardsFromDeals { max_cards, max_deals } => {
                    for removals in self.removal_combinations(max_cards, max_deals) {
                        actions.push(Action::PlayThingamabob { card, params: ThingamabobParams::RemoveFromDeals { removals } });
                    }
                }
                ThingamabobEffect::AddHiddenHandCardToDeal => {
                    for &hand_card in &self.players[player].hand {
                        for deal in &self.deals {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::CryptozooticExpander { hand_card, target_deal: deal.seller },
                            });
                        }
                    }
                }
                ThingamabobEffect::RevealCardInDeal => {
                    for deal in &self.deals {
                        for hidden in deal.hidden_cards() {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::SpectroelectricOptimeter { target_deal: deal.seller, target_card: hidden },
                            });
                        }
                    }
                }
            }
        }
        actions
    }

    /// All valid `(seller, card)` removal sets of size `0..=max_cards`,
    /// touching at most `max_deals` distinct deals (§2.4 Thingamabob table).
    fn removal_combinations(&self, max_cards: u8, max_deals: u8) -> Vec<Vec<(PlayerId, CardId)>> {
        let mut all_targets: Vec<(PlayerId, CardId)> = Vec::new();
        for deal in &self.deals {
            for c in &deal.cards {
                all_targets.push((deal.seller, c.card));
            }
        }
        let mut results = Vec::new();
        for k in 0..=max_cards as usize {
            for combo in combinations(&all_targets, k) {
                let distinct_deals: std::collections::HashSet<PlayerId> = combo.iter().map(|(s, _)| *s).collect();
                if distinct_deals.len() as u8 <= max_deals {
                    results.push(combo);
                }
            }
        }
        results
    }

    // apply_action

    pub fn apply_action(&mut self, player: PlayerId, action: Action) -> Result<Vec<Event>, RulesError> {
        if self.is_game_over() {
            return Err(RulesError::GameAlreadyOver);
        }
        if !self.active_players().contains(&player) {
            return Err(RulesError::NotActivePlayer(player));
        }

        match action {
            Action::SubmitDeal { cards } => self.apply_submit_deal(player, cards),
            Action::RevealCard { card } => self.apply_reveal_card(player, card),
            Action::BuyerPeek { target_seller } => self.apply_buyer_peek(player, target_seller),
            Action::PlayThingamabob { card, params } => self.apply_play_thingamabob(player, card, params),
            Action::PassThingamabobWindow => self.apply_pass_thingamabob(player),
            Action::ChooseDeal { seller } => self.apply_choose_deal(player, seller),
            Action::RespondToDeal { reverse } => self.apply_respond_to_deal(player, reverse),
            Action::ResolveNastyPenalty { taken_cards } => self.apply_resolve_nasty_penalty(player, taken_cards),
        }
    }

    fn apply_submit_deal(&mut self, player: PlayerId, cards: [CardId; 3]) -> Result<Vec<Event>, RulesError> {
        let Phase::DealOffer(s) = &self.phase else { return Err(RulesError::WrongPhaseAction(self.phase.name())) };
        if s.step != DealOfferStep::AwaitingSubmit {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        let unique: std::collections::HashSet<_> = cards.iter().collect();
        if unique.len() != 3 {
            return Err(RulesError::DealMustBeThreeDistinctCards);
        }
        for c in cards {
            if !self.players[player].hand.contains(&c) {
                return Err(RulesError::CardNotInHand(c));
            }
        }
        for c in cards {
            self.players[player].remove_from_hand(c);
        }
        self.deals.push(Deal::new(player, cards));

        let Phase::DealOffer(s) = &mut self.phase else { unreachable!() };
        let needs_reveal = s.current().needs_reveal;
        let mut events = vec![Event::DealSubmitted { player }];
        if needs_reveal {
            s.step = DealOfferStep::AwaitingReveal;
        } else {
            self.advance_deal_offer();
        }
        events.extend(self.maybe_finish_deal_offer());
        Ok(events)
    }

    fn apply_reveal_card(&mut self, player: PlayerId, card: CardId) -> Result<Vec<Event>, RulesError> {
        let Phase::DealOffer(s) = &self.phase else { return Err(RulesError::WrongPhaseAction(self.phase.name())) };
        if s.step != DealOfferStep::AwaitingReveal {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        let deal = self.deal_for_seller_mut(player).ok_or(RulesError::NoSuchDeal(player))?;
        if !deal.cards.iter().any(|c| c.card == card) {
            return Err(RulesError::RevealCardNotInDeal);
        }
        deal.reveal(card);
        self.advance_deal_offer();
        let mut events = vec![Event::CardRevealed { seller: player, card }];
        events.extend(self.maybe_finish_deal_offer());
        Ok(events)
    }

    fn advance_deal_offer(&mut self) {
        let Phase::DealOffer(s) = &mut self.phase else { unreachable!() };
        s.idx += 1;
        s.step = DealOfferStep::AwaitingSubmit;
    }

    fn maybe_finish_deal_offer(&mut self) -> Vec<Event> {
        let Phase::DealOffer(s) = &self.phase else { unreachable!() };
        if !s.is_done() {
            return vec![];
        }
        match self.mode {
            GameMode::Buyer => {
                self.phase = Phase::BuyerPeek;
            }
            GameMode::TwoPlayer => {
                self.start_thingamabob_window();
            }
        }
        vec![]
    }

    fn apply_buyer_peek(&mut self, player: PlayerId, target_seller: PlayerId) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::BuyerPeek) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        debug_assert_eq!(player, self.turn_leader);
        let hidden: Vec<CardId> = self.deal_for_seller(target_seller).ok_or(RulesError::NoSuchDeal(target_seller))?.hidden_cards().collect();
        if hidden.is_empty() {
            return Err(RulesError::NoSuchDeal(target_seller));
        }
        // Rulebook doesn't specify which hidden card gets flipped - resolved
        // uniformly at random. Flagged assumption (see README).
        let idx = self.rng.gen_index(hidden.len());
        let card = hidden[idx];
        self.deal_for_seller_mut(target_seller).unwrap().reveal(card);
        self.start_thingamabob_window();
        Ok(vec![Event::BuyerPeeked { target_seller, card }])
    }

    fn apply_pass_thingamabob(&mut self, player: PlayerId) -> Result<Vec<Event>, RulesError> {
        let Phase::ThingamabobWindow(s) = &mut self.phase else { return Err(RulesError::WrongPhaseAction(self.phase.name())) };
        debug_assert_eq!(player, s.current_player());
        s.consecutive_passes += 1;
        s.turn_idx += 1;
        let mut events = vec![];
        if s.consecutive_passes >= s.order.len() {
            events.push(Event::ThingamabobWindowClosed);
            self.phase = match self.mode {
                GameMode::Buyer => Phase::BuyerChoosesDeal,
                GameMode::TwoPlayer => Phase::RespondToDeal,
            };
        }
        Ok(events)
    }

    fn apply_play_thingamabob(&mut self, player: PlayerId, card: CardId, params: ThingamabobParams) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::ThingamabobWindow(_)) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        if !self.players[player].collection.contains(&card) {
            return Err(RulesError::CardNotInCollection(card));
        }
        let kind = match self.cards[&card].kind {
            CardKind::Thingamabob(k) => k,
            _ => return Err(RulesError::NotAThingamabob(card)),
        };
        let effect = self.catalog.thingamabob_effect(kind);

        let mut events = self.resolve_thingamabob_effect(player, effect, &params)?;

        self.players[player].remove_from_collection(card);
        self.discard_pile.push(card);
        events.insert(0, Event::ThingamabobPlayed { player, card, kind });

        let Phase::ThingamabobWindow(s) = &mut self.phase else { unreachable!() };
        s.consecutive_passes = 0;
        Ok(events)
    }

    fn resolve_thingamabob_effect(
        &mut self,
        player: PlayerId,
        effect: ThingamabobEffect,
        params: &ThingamabobParams,
    ) -> Result<Vec<Event>, RulesError> {
        match (effect, params) {
            (ThingamabobEffect::StealPointTokenFromRicherPlayer, ThingamabobParams::PlatonicIsolator { target_player }) => {
                let target_player = *target_player;
                if target_player == player {
                    return Err(RulesError::InvalidThingamabobParams);
                }
                if self.players[target_player].point_tokens <= self.players[player].point_tokens {
                    return Err(RulesError::IsolatorTargetNotRicher);
                }
                let taken = try_take_point_tokens(&mut self.players[target_player], 1);
                self.players[player].point_tokens += taken;
                Ok(vec![Event::PointTokenStolen { from: target_player, to: player, amount: taken }])
            }
            (ThingamabobEffect::RemoveCardsFromDeals { max_cards, max_deals }, ThingamabobParams::RemoveFromDeals { removals }) => {
                if removals.len() > max_cards as usize {
                    return Err(RulesError::TooManyRemovals);
                }
                let distinct: std::collections::HashSet<_> = removals.iter().map(|(s, _)| *s).collect();
                if distinct.len() > max_deals as usize {
                    return Err(RulesError::TooManyRemovals);
                }
                let card_set: std::collections::HashSet<_> = removals.iter().map(|(_, c)| *c).collect();
                if card_set.len() != removals.len() {
                    return Err(RulesError::DuplicateRemovalTarget);
                }
                let mut removed = Vec::new();
                for (seller, card) in removals {
                    let deal = self.deal_for_seller_mut(*seller).ok_or(RulesError::NoSuchDeal(*seller))?;
                    if !deal.remove_card(*card) {
                        return Err(RulesError::CardNotHiddenInDeal(*card));
                    }
                    self.discard_pile.push(*card);
                    removed.push(*card);
                }
                Ok(vec![Event::CardsDiscarded { cards: removed }])
            }
            (ThingamabobEffect::AddHiddenHandCardToDeal, ThingamabobParams::CryptozooticExpander { hand_card, target_deal }) => {
                if !self.players[player].hand.contains(hand_card) {
                    return Err(RulesError::CardNotInHand(*hand_card));
                }
                if self.deal_for_seller(*target_deal).is_none() {
                    return Err(RulesError::NoSuchDeal(*target_deal));
                }
                self.players[player].remove_from_hand(*hand_card);
                self.deal_for_seller_mut(*target_deal).unwrap().add_hidden_card(*hand_card);
                Ok(vec![])
            }
            (ThingamabobEffect::RevealCardInDeal, ThingamabobParams::SpectroelectricOptimeter { target_deal, target_card }) => {
                let deal = self.deal_for_seller_mut(*target_deal).ok_or(RulesError::NoSuchDeal(*target_deal))?;
                if !deal.hidden_cards().any(|c| c == *target_card) {
                    return Err(RulesError::CardNotHiddenInDeal(*target_card));
                }
                deal.reveal(*target_card);
                Ok(vec![Event::CardRevealed { seller: *target_deal, card: *target_card }])
            }
            _ => Err(RulesError::InvalidThingamabobParams),
        }
    }

    fn apply_choose_deal(&mut self, player: PlayerId, seller: PlayerId) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::BuyerChoosesDeal) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        debug_assert_eq!(player, self.turn_leader);
        if self.deal_for_seller(seller).is_none() {
            return Err(RulesError::NoSuchDeal(seller));
        }
        let buyer = self.turn_leader;
        let mut events = vec![Event::DealChosen { buyer, seller }];

        let deals = std::mem::take(&mut self.deals);
        for deal in deals {
            let cards = deal.all_card_ids();
            let recipient = if deal.seller == seller { buyer } else { deal.seller };
            self.players[recipient].collection.extend(cards.iter().copied());
            events.push(Event::CardsAwarded { player: recipient, cards });
        }

        self.turn_leader = seller;
        events.push(Event::TurnLeaderPassed { new_leader: seller });
        events.extend(self.begin_trade_in());
        Ok(events)
    }

    fn apply_respond_to_deal(&mut self, player: PlayerId, reverse: bool) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::RespondToDeal) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        let active = self.turn_leader;
        let responder = 1 - active;
        debug_assert_eq!(player, responder);

        let mut events = vec![Event::DealResponded { active, reverse }];
        let deals = std::mem::take(&mut self.deals);
        for deal in deals {
            let cards = deal.all_card_ids();
            let recipient = if reverse { 1 - deal.seller } else { deal.seller };
            self.players[recipient].collection.extend(cards.iter().copied());
            events.push(Event::CardsAwarded { player: recipient, cards });
        }

        self.turn_leader = responder;
        events.push(Event::TurnLeaderPassed { new_leader: responder });
        events.extend(self.begin_trade_in());
        Ok(events)
    }

    // mandatory trade-in (§2.3 step 6)

    fn begin_trade_in(&mut self) -> Vec<Event> {
        self.advance_nasty_resolution()
    }

    /// Scans from scratch for the next completed Nasty set requiring a
    /// choice, auto-resolving (discard + effect) every set that doesn't
    /// need one along the way. Re-invoked after each `ResolveNastyPenalty`
    /// so cascades (a stolen card completing a further set, possibly in the
    /// resolver's own Collection) are caught. When nothing is left, moves
    /// straight on to Creature trade-in + refill + end-of-turn.
    fn advance_nasty_resolution(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        loop {
            let mut found = None;
            let configured_kinds: Vec<NastyKind> = self.catalog.nasty_set_size.keys().copied().collect();
            'scan: for player in 0..self.num_players {
                for kind in configured_kinds.iter().copied() {
                    let set_size = self.catalog.nasty_set_size(kind);
                    let count = self.players[player].collection.iter().filter(|&&c| self.cards[&c].kind == CardKind::Nasty(kind)).count();
                    if count as u32 >= set_size {
                        found = Some((player, kind, set_size));
                        break 'scan;
                    }
                }
            }
            let Some((player, kind, set_size)) = found else { break };

            let to_discard: Vec<CardId> = self.players[player]
                .collection
                .iter()
                .copied()
                .filter(|&c| self.cards[&c].kind == CardKind::Nasty(kind))
                .take(set_size as usize)
                .collect();
            for &c in &to_discard {
                self.players[player].remove_from_collection(c);
            }
            self.discard_pile.extend(to_discard);
            events.push(Event::NastySetTradedIn { player, kind });

            match self.catalog.nasty_effect(kind) {
                NastyEffect::BuyerStealsOnePointToken => {
                    let taken = try_take_point_tokens(&mut self.players[player], 1);
                    self.players[self.turn_leader].point_tokens += taken;
                    if taken > 0 {
                        events.push(Event::PointTokenStolen { from: player, to: self.turn_leader, amount: taken });
                    }
                }
                NastyEffect::BuyerMayStealUpToNCards(_) => {
                    self.phase = Phase::NastyResolution(NastyResolutionState { pending: PendingNasty { player, kind } });
                    return events;
                }
            }
        }
        events.extend(self.finish_creature_trade_in_and_refill());
        events
    }

    fn apply_resolve_nasty_penalty(&mut self, player: PlayerId, taken_cards: Vec<CardId>) -> Result<Vec<Event>, RulesError> {
        let Phase::NastyResolution(s) = &self.phase else { return Err(RulesError::WrongPhaseAction(self.phase.name())) };
        debug_assert_eq!(player, self.turn_leader);
        let pending = s.pending.clone();
        let max = match self.catalog.nasty_effect(pending.kind) {
            NastyEffect::BuyerMayStealUpToNCards(max) => max,
            NastyEffect::BuyerStealsOnePointToken => 0,
        };
        if taken_cards.len() > max as usize {
            return Err(RulesError::TooManyCardsTaken);
        }
        let unique: std::collections::HashSet<_> = taken_cards.iter().collect();
        if unique.len() != taken_cards.len() {
            return Err(RulesError::DuplicateCardTaken);
        }
        for &c in &taken_cards {
            if !self.players[pending.player].collection.contains(&c) {
                return Err(RulesError::CardNotInCollection(c));
            }
        }

        let taken = try_take_cards(&mut self.players[pending.player], &taken_cards, max as u32);
        self.players[player].collection.extend(taken.iter().copied());
        let mut events = Vec::new();
        if !taken.is_empty() {
            events.push(Event::CardsStolen { from: pending.player, to: player, cards: taken });
        }

        events.extend(self.advance_nasty_resolution());
        Ok(events)
    }

    fn finish_creature_trade_in_and_refill(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let configured_tiers: Vec<Tier> = self.catalog.creature_set_size.keys().copied().collect();
        for player in 0..self.num_players {
            for tier in configured_tiers.iter().copied() {
                let set_size = self.catalog.creature_set_size(tier);
                loop {
                    let count = self.players[player].collection.iter().filter(|&&c| self.cards[&c].kind == CardKind::Creature(tier)).count();
                    if (count as u32) < set_size {
                        break;
                    }
                    let to_discard: Vec<CardId> = self.players[player]
                        .collection
                        .iter()
                        .copied()
                        .filter(|&c| self.cards[&c].kind == CardKind::Creature(tier))
                        .take(set_size as usize)
                        .collect();
                    for &c in &to_discard {
                        self.players[player].remove_from_collection(c);
                    }
                    self.discard_pile.extend(to_discard);
                    self.players[player].point_tokens += 1;
                    events.push(Event::CreatureSetTradedIn { player, tier, tokens_gained: 1 });
                }
            }
        }

        for player in 0..self.num_players {
            let mut drawn = 0usize;
            while self.players[player].hand.len() < HAND_SIZE {
                if self.draw_pile.is_empty() {
                    if self.discard_pile.is_empty() {
                        break; // total-card starvation edge case; nothing left to draw
                    }
                    let mut reshuffled = std::mem::take(&mut self.discard_pile);
                    self.rng.shuffle(&mut reshuffled);
                    self.draw_pile = reshuffled;
                    events.push(Event::DrawPileReshuffledFromDiscard);
                }
                if let Some(c) = self.draw_pile.pop() {
                    self.players[player].hand.push(c);
                    drawn += 1;
                } else {
                    break;
                }
            }
            if drawn > 0 {
                events.push(Event::HandRefilled { player, drawn });
            }
        }

        if let Some(winner) = self.check_win_condition() {
            self.phase = Phase::GameOver { winner };
            events.push(Event::GameOver { winner });
        } else {
            self.start_new_turn();
        }
        events
    }

    fn check_win_condition(&self) -> Option<PlayerId> {
        let mut order: Vec<PlayerId> = (0..self.num_players).collect();
        order.sort_by_key(|&p| std::cmp::Reverse(self.players[p].point_tokens));
        let leader = order[0];
        let leader_tokens = self.players[leader].point_tokens;
        if leader_tokens < self.win_threshold {
            return None;
        }
        if self.num_players == 1 {
            return Some(leader);
        }
        let runner_up = order[1];
        if leader_tokens > self.players[runner_up].point_tokens {
            Some(leader)
        } else {
            None
        }
    }

    // observation

    pub fn observation_for(&self, player: PlayerId) -> Observation {
        let deals = self
            .deals
            .iter()
            .map(|d| ObservedDeal {
                seller: d.seller,
                revealed_cards: d.cards.iter().filter(|c| c.revealed).map(|c| c.card).collect(),
                num_hidden: d.cards.iter().filter(|c| !c.revealed).count(),
            })
            .collect();
        Observation {
            player,
            phase: self.phase.name(),
            turn_leader: self.turn_leader,
            own_hand: self.players[player].hand.clone(),
            collections: self.players.iter().map(|p| p.collection.clone()).collect(),
            point_tokens: self.players.iter().map(|p| p.point_tokens).collect(),
            deals,
            draw_pile_len: self.draw_pile.len(),
            discard_pile_len: self.discard_pile.len(),
            winner: self.winner(),
        }
    }
}

/// All k-element combinations of `items`, preserving relative order.
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    let mut results = Vec::new();
    combinations_helper(items, k, 0, &mut Vec::new(), &mut results);
    results
}

fn combinations_helper<T: Clone>(items: &[T], k: usize, start: usize, current: &mut Vec<T>, results: &mut Vec<Vec<T>>) {
    if current.len() == k {
        results.push(current.clone());
        return;
    }
    for i in start..items.len() {
        current.push(items[i].clone());
        combinations_helper(items, k, i + 1, current, results);
        current.pop();
    }
}
