//! [`GameState`]: the full rules engine. Pure data plus one entry point,
//! [`GameState::apply_action`], that returns structured [`Event`]s. No I/O.

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    action::{Action, ThingamabobParams},
    card::{
        Card, CardId, CardKind, NastyEffect, NastyKind, PlayerId, ThingamabobEffect,
        ThingamabobKind, Tier, NUM_CARD_CLASSES,
    },
    deal::Deal,
    deck::{Catalog, DeckConfig},
    phase::{
        DealOfferEntry, DealOfferState, DealOfferStep, NastyResolutionState, PendingNasty, Phase,
        ThingamabobWindowState, TwoPlayerDealState, TwoPlayerDealStep,
    },
    player::{try_take_cards, try_take_point_tokens, Player},
    rng::GameRng,
    stats::{event_kind_index, PlayerStats, EVENT_MEMORY_DECAY, NUM_EVENT_KINDS},
};

/// Number of cards dealt into each player's starting hand.
pub const HAND_SIZE: usize = 5;

/// Which table-size variant of the rules is in effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    /// Three or more players: one buyer, multiple sellers per turn.
    Buyer,
    /// Exactly two players: the split-pile variant.
    TwoPlayer,
}

/// Something that happened as a result of applying an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A seller submitted a deal.
    DealSubmitted {
        /// The seller.
        player: PlayerId,
    },
    /// A card in a deal was revealed.
    CardRevealed {
        /// The deal's seller.
        seller: PlayerId,
        /// The card revealed.
        card: CardId,
    },
    /// The buyer peeked at a hidden card.
    BuyerPeeked {
        /// The seller peeked at.
        target_seller: PlayerId,
        /// The card revealed by the peek.
        card: CardId,
    },
    /// A thingamabob was played.
    ThingamabobPlayed {
        /// The player who played it.
        player: PlayerId,
        /// The card played.
        card: CardId,
        /// The thingamabob kind.
        kind: ThingamabobKind,
    },
    /// The thingamabob response window closed.
    ThingamabobWindowClosed,
    /// The buyer chose a deal.
    DealChosen {
        /// The buyer.
        buyer: PlayerId,
        /// The chosen seller.
        seller: PlayerId,
    },
    /// Two-player mode: the responder answered a deal.
    DealResponded {
        /// The active player who made the offer.
        active: PlayerId,
        /// Whether the responder reversed the deal.
        reverse: bool,
    },
    /// Cards were awarded to a player's collection.
    CardsAwarded {
        /// The player receiving the cards.
        player: PlayerId,
        /// The cards awarded.
        cards: Vec<CardId>,
    },
    /// Cards went to the discard pile.
    CardsDiscarded {
        /// The cards discarded.
        cards: Vec<CardId>,
    },
    /// A nasty set was traded in.
    NastySetTradedIn {
        /// The player who traded it in.
        player: PlayerId,
        /// The nasty kind.
        kind: NastyKind,
    },
    /// A point token was stolen.
    PointTokenStolen {
        /// The player it was taken from.
        from: PlayerId,
        /// The player who received it.
        to: PlayerId,
        /// How many tokens moved.
        amount: u32,
    },
    /// Cards were stolen.
    CardsStolen {
        /// The player they were taken from.
        from: PlayerId,
        /// The player who received them.
        to: PlayerId,
        /// The cards stolen.
        cards: Vec<CardId>,
    },
    /// A creature set was traded in.
    CreatureSetTradedIn {
        /// The player who traded it in.
        player: PlayerId,
        /// The creature tier.
        tier: Tier,
        /// Point tokens gained for the trade-in.
        tokens_gained: u32,
    },
    /// A player's hand was refilled from the draw pile.
    HandRefilled {
        /// The player refilled.
        player: PlayerId,
        /// Number of cards drawn.
        drawn: usize,
    },
    /// The draw pile was reshuffled from the discard pile.
    DrawPileReshuffledFromDiscard,
    /// The turn leader passed to a new player.
    TurnLeaderPassed {
        /// The new turn leader.
        new_leader: PlayerId,
    },
    /// The game ended.
    GameOver {
        /// The winning player.
        winner: PlayerId,
    },
}

/// Everything that can go wrong applying an action.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RulesError {
    /// Someone other than the active player tried to act.
    #[error("player {0} is not currently allowed to act")]
    NotActivePlayer(PlayerId),
    /// The action doesn't match what the current phase expects.
    #[error("wrong action for current phase ({0})")]
    WrongPhaseAction(&'static str),
    /// The card isn't in the acting player's hand.
    #[error("card {0} not found in player's hand")]
    CardNotInHand(CardId),
    /// The card isn't in the acting player's collection.
    #[error("card {0} not found in player's collection")]
    CardNotInCollection(CardId),
    /// A submitted deal wasn't exactly three distinct cards.
    #[error("submitted deal must be 3 distinct cards")]
    DealMustBeThreeDistinctCards,
    /// The card being revealed isn't part of the deal.
    #[error("revealed card must be one of the submitted deal cards")]
    RevealCardNotInDeal,
    /// The named seller has no active deal.
    #[error("no such active deal for seller {0}")]
    NoSuchDeal(PlayerId),
    /// The card isn't a still-hidden card in that deal.
    #[error("card {0} is not a still-hidden card in that deal")]
    CardNotHiddenInDeal(CardId),
    /// The card isn't a thingamabob.
    #[error("card {0} is not a thingamabob")]
    NotAThingamabob(CardId),
    /// The thingamabob params don't match this card's effect.
    #[error("invalid thingamabob params for this card's effect")]
    InvalidThingamabobParams,
    /// Platonic Isolator's target doesn't currently have more point tokens.
    #[error("platonic isolator target must currently have strictly more point tokens")]
    IsolatorTargetNotRicher,
    /// The removal targets exceed this card's max cards or deals.
    #[error("removal targets exceed this card's max cards/deals")]
    TooManyRemovals,
    /// The removal list named the same target twice.
    #[error("duplicate target in removal list")]
    DuplicateRemovalTarget,
    /// The nasty penalty resolution asked for more cards than the effect allows.
    #[error("nasty penalty resolution requested more cards than the effect allows")]
    TooManyCardsTaken,
    /// The nasty penalty resolution named the same card twice.
    #[error("duplicate card in nasty penalty resolution")]
    DuplicateCardTaken,
    /// The game has already ended.
    #[error("the game has already ended")]
    GameAlreadyOver,
}

/// Errors from live-tracking a physical game, see [`GameState::pin_kind`]:
/// pinning a card's kind to match what was actually revealed at the table,
/// instead of the kind randomly assigned at deal-shuffle time.
/// Everything that can go wrong pinning a live-tracked card's kind.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PinError {
    /// The card id doesn't exist in this game.
    #[error("card {0} does not exist in this game")]
    UnknownCard(CardId),
    /// The card was already pinned to a kind.
    #[error("card {0} was already pinned to a kind")]
    AlreadyPinned(CardId),
    /// No cards of that kind remain unaccounted for in the deck.
    #[error("no cards of that kind remain unaccounted for in the deck")]
    NoSupplyRemaining,
}

/// Everything that can go wrong setting up a new game.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SetupError {
    /// `num_players` was outside the supported 2 to 6 range.
    #[error("num_players must be between 2 and 6, got {0}")]
    InvalidPlayerCount(usize),
    /// The deck didn't have enough cards to deal starting hands.
    #[error("deck has {have} cards, not enough to deal {need} initial hand cards")]
    NotEnoughCardsToDeal {
        /// Cards actually in the deck.
        have: usize,
        /// Cards needed for starting hands.
        need: usize,
    },
    /// The requested starting leader seat doesn't exist at this table size.
    #[error("starting_leader {given} is out of range for {num_players} players")]
    InvalidStartingLeader {
        /// The seat that was requested.
        given: PlayerId,
        /// The table size.
        num_players: usize,
    },
}

/// One deal as seen from outside, with hidden cards collapsed to a count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedDeal {
    /// The seller who made this deal.
    pub seller: PlayerId,
    /// Cards currently revealed in the deal.
    pub revealed_cards: Vec<CardId>,
    /// Number of cards still hidden.
    pub num_hidden: usize,
}

/// Filtered, player-specific view of [`GameState`]: full information
/// internally, filtered externally. Never exposes other players' hands or
/// still-hidden deal cards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// The player this observation is for.
    pub player: PlayerId,
    /// Current phase name.
    pub phase: &'static str,
    /// Current turn leader.
    pub turn_leader: PlayerId,
    /// This player's own hand.
    pub own_hand: Vec<CardId>,
    /// Every player's collection, public information.
    pub collections: Vec<Vec<CardId>>,
    /// Every player's point token count.
    pub point_tokens: Vec<u32>,
    /// Active deals, with hidden cards collapsed.
    pub deals: Vec<ObservedDeal>,
    /// Number of cards left in the draw pile.
    pub draw_pile_len: usize,
    /// Number of cards in the discard pile.
    pub discard_pile_len: usize,
    /// The winner, if the game has ended.
    pub winner: Option<PlayerId>,
}

/// The full state of one game in progress.
pub struct GameState {
    /// Rules data for this game's deck.
    pub(crate) catalog: Catalog,
    /// Indexed by [`CardId`] (deck ids are contiguous `0..n`) rather than
    /// hashed, since card lookup is the hottest operation in the engine.
    pub(crate) cards: Vec<Card>,
    /// `cards[i].kind`, split out so the hot path never touches the
    /// `String` name field's cache line.
    pub(crate) kinds: Vec<CardKind>,
    /// The table-size variant in effect.
    pub(crate) mode: GameMode,
    /// Number of players at the table.
    pub(crate) num_players: usize,
    /// Point tokens needed to win.
    pub(crate) win_threshold: u32,
    /// Every player's hand, collection, and token count.
    pub(crate) players: Vec<Player>,
    /// Cards left to draw.
    pub(crate) draw_pile: Vec<CardId>,
    /// Discarded cards.
    pub(crate) discard_pile: Vec<CardId>,
    /// Active deals on the table.
    pub(crate) deals: Vec<Deal>,
    /// Current turn leader.
    pub(crate) turn_leader: PlayerId,
    /// Current phase.
    pub(crate) phase: Phase,
    /// This game's RNG.
    rng: GameRng,
    /// True composition of the deck per card class. Constant for a game; the
    /// denominator of the observation's card-counting features.
    pub(crate) deck_counts: [u32; NUM_CARD_CLASSES],
    /// Turns completed so far, lets a policy tell an opening from an endgame.
    pub(crate) turn_index: u32,
    /// Deck- and table-size-derived ceiling on the legal-action count, see
    /// [`Self::max_legal_actions`].
    max_legal_actions: usize,
    /// Per-player running behavioral summary, see the `stats` module: the
    /// engine's half of an agent's memory, so a policy can condition on
    /// what each seat has done this game, not only on the current tableau.
    pub(crate) stats: Vec<PlayerStats>,
    /// Exponentially decayed histogram of recent event kinds, a compact
    /// "what just happened" trace over the last handful of micro-steps.
    pub(crate) event_memory: [f32; NUM_EVENT_KINDS],
    /// Live-tracking support, see [`Self::pin_kind`]: remaining
    /// not-yet-pinned supply of each kind, seeded from the deck's true
    /// composition. Cards start with an arbitrary (random) kind from the
    /// shuffle; `pin_kind` overwrites it once, when the card is actually
    /// revealed at the table, so all of the engine's own bookkeeping (set
    /// completion, tokens, discards) runs on truth instead of the random
    /// deal.
    kind_supply: [u32; NUM_CARD_CLASSES],
    /// Whether each card id has already been pinned via [`Self::pin_kind`].
    pinned: Vec<bool>,
}

impl GameState {
    /// Builds a new game with a randomly chosen starting leader.
    pub fn new(num_players: usize, deck: DeckConfig, seed: u64) -> Result<GameState, SetupError> {
        Self::new_with_starting_leader(num_players, deck, seed, None)
    }

    /// Like [`Self::new`], but pins the first turn's leader/buyer (two-player
    /// mode included) to `starting_leader` instead of picking one uniformly
    /// at random. Used for a human-configured game where the players have
    /// already agreed who goes first or who buys first.
    pub fn new_with_starting_leader(
        num_players: usize,
        deck: DeckConfig,
        seed: u64,
        starting_leader: Option<PlayerId>,
    ) -> Result<GameState, SetupError> {
        if !(2..=6).contains(&num_players) {
            return Err(SetupError::InvalidPlayerCount(num_players));
        }
        if let Some(given) = starting_leader {
            if given >= num_players {
                return Err(SetupError::InvalidStartingLeader { given, num_players });
            }
        }
        let needed = num_players * HAND_SIZE;
        if deck.cards.len() < needed {
            return Err(SetupError::NotEnoughCardsToDeal {
                have: deck.cards.len(),
                need: needed,
            });
        }

        let mut rng = GameRng::from_seed(seed);
        let mut ids: Vec<CardId> = deck.cards.iter().map(|c| c.id).collect();
        rng.shuffle(&mut ids);

        let mut cards = deck.cards;
        cards.sort_unstable_by_key(|c| c.id);
        debug_assert!(
            cards.iter().enumerate().all(|(i, c)| c.id as usize == i),
            "deck card ids must be contiguous 0..n for Vec-indexed lookup"
        );
        let kinds: Vec<CardKind> = cards.iter().map(|c| c.kind).collect();
        let mut kind_supply = [0u32; NUM_CARD_CLASSES];
        for &k in &kinds {
            kind_supply[k.class_index()] += 1;
        }
        let deck_counts = kind_supply;

        let mut players: Vec<Player> = (0..num_players).map(|_| Player::new()).collect();
        let mut cursor = 0usize;
        for p in players.iter_mut() {
            p.hand = ids[cursor..cursor + HAND_SIZE].to_vec();
            cursor += HAND_SIZE;
        }
        let draw_pile = ids[cursor..].to_vec();

        let mode = if num_players == 2 {
            GameMode::TwoPlayer
        } else {
            GameMode::Buyer
        };
        let win_threshold = match (mode, num_players) {
            (GameMode::TwoPlayer, _) => 5,
            (GameMode::Buyer, 3 | 4) => 4,
            (GameMode::Buyer, _) => 5,
        };
        let turn_leader = starting_leader.unwrap_or_else(|| rng.gen_index(num_players));

        let num_cards = cards.len();
        let catalog = deck.catalog;
        let mut game = GameState {
            catalog: catalog.clone(),
            cards,
            kinds,
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
            deck_counts,
            turn_index: 0,
            max_legal_actions: Self::compute_max_legal_actions(
                &catalog,
                &deck_counts,
                mode,
                num_players,
            ),
            stats: vec![PlayerStats::default(); num_players],
            event_memory: [0.0; NUM_EVENT_KINDS],
            kind_supply,
            pinned: vec![false; num_cards],
        };
        game.start_new_turn();
        Ok(game)
    }

    /// Number of players at the table.
    pub fn num_players(&self) -> usize {
        self.num_players
    }

    /// The table-size variant in effect.
    pub fn mode(&self) -> GameMode {
        self.mode
    }

    /// Current turn leader.
    pub fn turn_leader(&self) -> PlayerId {
        self.turn_leader
    }

    /// Point tokens needed to win.
    pub fn win_threshold(&self) -> u32 {
        self.win_threshold
    }

    /// The winning player, if the game has ended.
    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => Some(winner),
            _ => None,
        }
    }

    /// Whether the game has ended.
    pub fn is_game_over(&self) -> bool {
        matches!(self.phase, Phase::GameOver { .. })
    }

    /// Stable string name of the current phase.
    pub fn current_phase(&self) -> &'static str {
        self.phase.name()
    }

    /// The mechanical kind of a card, or `None` if the id doesn't exist.
    pub fn card_kind(&self, id: CardId) -> Option<CardKind> {
        self.kinds.get(id as usize).copied()
    }

    /// Unchecked hot-path variant of [`Self::card_kind`], for ids the
    /// engine itself produced (every id in a hand, collection, deal, or
    /// pile is by construction a valid index).
    #[inline]
    pub(crate) fn kind_of(&self, id: CardId) -> CardKind {
        self.kinds[id as usize]
    }

    /// A card's class index, see [`CardKind::class_index`].
    #[inline]
    pub(crate) fn class_of(&self, id: CardId) -> usize {
        self.kinds[id as usize].class_index()
    }

    /// A card's display name, or `None` if the id doesn't exist.
    pub fn card_name(&self, id: CardId) -> Option<&str> {
        self.cards.get(id as usize).map(|c| c.name.as_str())
    }

    /// True composition of the deck per card class, the denominator for the
    /// observation's "what is still out there" card-counting features.
    pub fn deck_counts(&self) -> &[u32; NUM_CARD_CLASSES] {
        &self.deck_counts
    }

    /// Turns completed so far.
    pub fn turn_index(&self) -> u32 {
        self.turn_index
    }

    /// Exact upper bound on `legal_actions(..).len()` for this deck and
    /// table size, derived from the deck composition rather than guessed.
    ///
    /// The RL action space is ordinal; index `i` means "the i-th entry of
    /// [`Self::legal_actions`] right now", so it has to be wide enough that
    /// no legal action is ever unreachable, and no wider, since every
    /// unused slot is dead weight in the policy head. Deriving it here
    /// means a custom `deck.toml` resizes the action space automatically
    /// instead of silently truncating.
    pub fn max_legal_actions(&self) -> usize {
        self.max_legal_actions
    }

    /// Computes [`Self::max_legal_actions`] from the deck composition and table size.
    fn compute_max_legal_actions(
        catalog: &Catalog,
        deck_counts: &[u32; NUM_CARD_CLASSES],
        mode: GameMode,
        num_players: usize,
    ) -> usize {
        // worst case a deal pool can reach: every deal at its initial 3
        // cards, plus one extra per cryptozootic expander in the deck (each
        // adds exactly one hand card to some deal). removal effects only
        // shrink it
        let max_deals = match mode {
            GameMode::Buyer => num_players - 1,
            GameMode::TwoPlayer => 2,
        };
        let expanders = deck_counts
            [CardKind::Thingamabob(ThingamabobKind::CryptozooticExpander).class_index()]
            as usize;
        let deal_cards = 3 * max_deals + expanders;

        let mut thingamabob_window = 1; // pass
        for kind in ThingamabobKind::ALL {
            if deck_counts[CardKind::Thingamabob(kind).class_index()] == 0 {
                continue;
            }
            // one representative card per class, legal_actions dedups the
            // rest away
            thingamabob_window += match catalog.thingamabob_effect(kind) {
                ThingamabobEffect::StealPointTokenFromRicherPlayer => num_players - 1,
                ThingamabobEffect::RemoveCardsFromDeals { max_cards, .. } => (0..=max_cards
                    as usize)
                    .map(|k| binomial(deal_cards, k))
                    .sum(),
                ThingamabobEffect::AddHiddenHandCardToDeal => {
                    HAND_SIZE.min(NUM_CARD_CLASSES) * max_deals
                }
                ThingamabobEffect::RevealCardInDeal => deal_cards,
            };
        }

        // nasty penalty targets are class-deduped, so the count is the
        // number of distinct class multisets of size <= the steal cap
        let max_steal = catalog
            .nasty_effect
            .values()
            .map(|e| match e {
                NastyEffect::BuyerMayStealUpToNCards(n) => *n as usize,
                NastyEffect::BuyerStealsOnePointToken => 0,
            })
            .max()
            .unwrap_or(0);
        let nasty_resolution: usize = (0..=max_steal)
            .map(|k| binomial(NUM_CARD_CLASSES + k.saturating_sub(1), k))
            .sum();

        let deal_offer = match mode {
            GameMode::Buyer => binomial(HAND_SIZE, 3),
            // each 3-card pick crossed with the 2^3 ways to assign those
            // cards to the two piles
            GameMode::TwoPlayer => binomial(HAND_SIZE, 3) * 8,
        };

        [
            thingamabob_window,
            nasty_resolution,
            deal_offer,
            max_deals,
            3,
            2,
        ]
        .into_iter()
        .max()
        .unwrap_or(1)
    }

    /// A player's current hand.
    pub fn player_hand(&self, player: PlayerId) -> &[CardId] {
        &self.players[player].hand
    }

    /// A player's current collection.
    pub fn player_collection(&self, player: PlayerId) -> &[CardId] {
        &self.players[player].collection
    }

    /// A player's current point token count.
    pub fn player_point_tokens(&self, player: PlayerId) -> u32 {
        self.players[player].point_tokens
    }

    /// Still-hidden card ids in `seller`'s active deal, empty if no such
    /// deal. Bypasses the usual observation privacy filtering; only meant
    /// for a live-tracking caller who is the sole source of truth for what
    /// these cards really are, not an in-game player peeking.
    pub fn hidden_cards_in_deal(&self, seller: PlayerId) -> Vec<CardId> {
        self.deal_for_seller(seller)
            .map(|d| d.hidden_cards().collect())
            .unwrap_or_default()
    }

    /// Whether a card has already been pinned via [`Self::pin_kind`].
    pub fn is_pinned(&self, card: CardId) -> bool {
        self.pinned.get(card as usize).copied().unwrap_or(false)
    }

    /// Remaining not-yet-pinned supply per kind (how many more cards of that
    /// kind could still be truthfully assigned via `pin_kind`).
    pub fn kind_supply(&self) -> HashMap<CardKind, u32> {
        let kinds = Tier::ALL
            .iter()
            .map(|&t| CardKind::Creature(t))
            .chain(NastyKind::ALL.iter().map(|&k| CardKind::Nasty(k)))
            .chain(
                ThingamabobKind::ALL
                    .iter()
                    .map(|&k| CardKind::Thingamabob(k)),
            );
        kinds
            .map(|k| (k, self.kind_supply[k.class_index()]))
            .collect()
    }

    /// For live-tracking a physical game: overwrites `card`'s kind to
    /// `kind`, matching what was actually revealed at the table, and
    /// consumes one unit of that kind's remaining supply so the deck's true
    /// composition (from `deck.toml`) is never exceeded. Every downstream
    /// rule (set completion, nasty and thingamabob effects, tokens,
    /// discards) reads [`Self::card_kind`] normally afterward, so once
    /// pinned a card behaves exactly like a "real" one.
    pub fn pin_kind(&mut self, card: CardId, kind: CardKind) -> Result<(), PinError> {
        if card as usize >= self.cards.len() {
            return Err(PinError::UnknownCard(card));
        }
        if self.pinned[card as usize] {
            return Err(PinError::AlreadyPinned(card));
        }
        let remaining = &mut self.kind_supply[kind.class_index()];
        if *remaining == 0 {
            return Err(PinError::NoSupplyRemaining);
        }
        *remaining -= 1;
        self.kinds[card as usize] = kind;
        let entry = &mut self.cards[card as usize];
        entry.kind = kind;
        // name is a separate field, set once at deck-build time for
        // whatever kind the card originally, randomly got. it must be
        // overwritten too, or card_name() keeps returning the old,
        // now-wrong flavor name for nasties and thingamabobs, where display
        // falls back to it. creatures are shown by tier only, so this only
        // ever bit nasty and thingamabob cards, see card_display.display_name
        // on the python side
        entry.name = match kind {
            CardKind::Creature(tier) => tier.to_string(),
            CardKind::Nasty(k) => k.name().to_string(),
            CardKind::Thingamabob(k) => k.name().to_string(),
        };
        self.pinned[card as usize] = true;
        Ok(())
    }

    /// Total card count across hands, draw pile, discard pile, deals, and
    /// collections. Used by property tests to assert conservation.
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
        (1..self.num_players)
            .map(|offset| (from + offset) % self.num_players)
            .collect()
    }

    /// Resets deals and enters the first phase of a new turn.
    fn start_new_turn(&mut self) {
        self.deals.clear();
        self.phase = match self.mode {
            GameMode::Buyer => {
                let entries = self
                    .table_order_after(self.turn_leader)
                    .into_iter()
                    .map(|p| DealOfferEntry {
                        player: p,
                        needs_reveal: true,
                    })
                    .collect();
                Phase::DealOffer(DealOfferState {
                    entries,
                    idx: 0,
                    step: DealOfferStep::AwaitingSubmit,
                })
            }
            GameMode::TwoPlayer => Phase::TwoPlayerDealOffer(TwoPlayerDealState {
                step: TwoPlayerDealStep::AwaitingSplit,
            }),
        };
    }

    /// Enters the thingamabob response window in table order starting from the turn leader.
    fn start_thingamabob_window(&mut self) {
        let order = match self.mode {
            GameMode::Buyer => {
                let mut o = vec![self.turn_leader];
                o.extend(self.table_order_after(self.turn_leader));
                o
            }
            GameMode::TwoPlayer => vec![self.turn_leader, 1 - self.turn_leader],
        };
        self.phase = Phase::ThingamabobWindow(ThingamabobWindowState {
            order,
            turn_idx: 0,
            consecutive_passes: 0,
        });
    }

    // active player / legality

    /// The single seat to act right now, if any. Every phase is a
    /// micro-turn belonging to exactly one player, so this, not the
    /// allocating [`Self::active_players`], is what the engine itself uses.
    pub fn active_player(&self) -> Option<PlayerId> {
        match &self.phase {
            Phase::DealOffer(s) => (!s.is_done()).then(|| s.current().player),
            Phase::TwoPlayerDealOffer(_) => Some(self.turn_leader),
            Phase::BuyerPeek => Some(self.turn_leader),
            Phase::ThingamabobWindow(s) => Some(s.current_player()),
            Phase::BuyerChoosesDeal => Some(self.turn_leader),
            Phase::RespondToDeal => Some(1 - self.turn_leader),
            Phase::NastyResolution(s) => Some(self.nasty_beneficiary(s.pending.player)),
            Phase::GameOver { .. } => None,
        }
    }

    /// PettingZoo-shaped view of [`Self::active_player`], keeping the door
    /// open for a simultaneous-phase variant with more than one active seat.
    pub fn active_players(&self) -> Vec<PlayerId> {
        self.active_player().into_iter().collect()
    }

    /// Who resolves a completed nasty set belonging to `loser`. In buyer
    /// mode, the buyer marker has already been reassigned to the new
    /// buyer/seller at deal-resolution time, so it's always `turn_leader`
    /// (the newly-chosen seller never receives cards this same turn, so
    /// `loser` is never that same player). In two-player mode the same
    /// "new turn leader" shortcut doesn't hold: an accept can hand the
    /// responder their own pile back and complete a set in their own
    /// collection, so the resolver there is always explicitly "the other
    /// player" relative to `loser`, not whichever seat currently holds
    /// `turn_leader`.
    fn nasty_beneficiary(&self, loser: PlayerId) -> PlayerId {
        match self.mode {
            GameMode::TwoPlayer => 1 - loser,
            GameMode::Buyer => self.turn_leader,
        }
    }

    /// The active deal made by `seller`, if any.
    pub(crate) fn deal_for_seller(&self, seller: PlayerId) -> Option<&Deal> {
        self.deals.iter().find(|d| d.seller == seller)
    }

    /// Mutable version of [`Self::deal_for_seller`].
    fn deal_for_seller_mut(&mut self, seller: PlayerId) -> Option<&mut Deal> {
        self.deals.iter_mut().find(|d| d.seller == seller)
    }

    /// Legal actions for `player`, deduplicated by card class.
    ///
    /// Two cards of the same class are mechanically interchangeable
    /// (creatures carry no individual behavior, and a second Detrital
    /// Repositioner does exactly what the first one does), so enumerating
    /// one option per distinct card id produces large blocks of literally
    /// equivalent actions. Collapsing them shrinks the worst case by about
    /// 3x, which shrinks the policy's action space, the per-step
    /// enumeration cost, and the number of indistinguishable choices a
    /// learner has to waste probability mass on.
    ///
    /// The dedup only ever applies to cards the acting player can already
    /// see (their own hand, their own just-submitted deal, public
    /// collections). Still-hidden cards in a deal stay one option each: the
    /// count of options there is public (`num_hidden`), while a
    /// class-deduped count would leak how many distinct kinds are hidden.
    pub fn legal_actions(&self, player: PlayerId) -> Vec<Action> {
        if self.active_player() != Some(player) {
            return Vec::new();
        }
        match &self.phase {
            Phase::DealOffer(s) => {
                if s.step == DealOfferStep::AwaitingSubmit {
                    self.class_distinct_combinations(&self.players[player].hand, 3)
                        .into_iter()
                        .map(|c| Action::SubmitDeal {
                            cards: [c[0], c[1], c[2]],
                        })
                        .collect()
                } else {
                    let deal = self
                        .deal_for_seller(player)
                        .expect("submitted deal must exist");
                    let cards: Vec<CardId> = deal.cards.iter().map(|c| c.card).collect();
                    self.class_distinct(&cards)
                        .into_iter()
                        .map(|card| Action::RevealCard { card })
                        .collect()
                }
            }
            Phase::TwoPlayerDealOffer(s) => match s.step {
                TwoPlayerDealStep::AwaitingSplit => self.legal_two_player_splits(player),
                TwoPlayerDealStep::AwaitingReveal => {
                    // Both piles are the active player's own just-offered
                    // cards, so they are fully known to them - dedup applies.
                    let cards: Vec<CardId> = self
                        .deals
                        .iter()
                        .flat_map(|d| d.cards.iter().map(|c| c.card))
                        .collect();
                    self.class_distinct(&cards)
                        .into_iter()
                        .map(|card| Action::RevealCard { card })
                        .collect()
                }
            },
            Phase::BuyerPeek => self
                .deals
                .iter()
                .map(|d| Action::BuyerPeek {
                    target_seller: d.seller,
                })
                .collect(),
            Phase::ThingamabobWindow(_) => self.legal_thingamabob_actions(player),
            Phase::BuyerChoosesDeal => self
                .deals
                .iter()
                .map(|d| Action::ChooseDeal { seller: d.seller })
                .collect(),
            Phase::RespondToDeal => vec![
                Action::RespondToDeal { reverse: false },
                Action::RespondToDeal { reverse: true },
            ],
            Phase::NastyResolution(s) => {
                let max = match self.catalog.nasty_effect(s.pending.kind) {
                    NastyEffect::BuyerMayStealUpToNCards(max) => max,
                    NastyEffect::BuyerStealsOnePointToken => 0, // never queued (auto-resolved)
                };
                // Collections are public, so class-deduping the steal
                // targets hides nothing the thief could otherwise see.
                let target_collection = self.players[s.pending.player].collection.clone();
                (0..=max)
                    .flat_map(|k| self.class_distinct_combinations(&target_collection, k as usize))
                    .map(|cards| Action::ResolveNastyPenalty { taken_cards: cards })
                    .collect()
            }
            Phase::GameOver { .. } => Vec::new(),
        }
    }

    /// How many actions `legal_actions(player)` would return. Kept as its own
    /// entry point so callers that only need the action-mask width (the RL
    /// wrapper, every step) do not have to look at the actions themselves.
    pub fn legal_action_count(&self, player: PlayerId) -> usize {
        self.legal_actions(player).len()
    }

    /// One representative card id per distinct class, preserving order.
    fn class_distinct(&self, cards: &[CardId]) -> Vec<CardId> {
        let mut seen = [false; NUM_CARD_CLASSES];
        let mut out = Vec::with_capacity(cards.len().min(NUM_CARD_CLASSES));
        for &c in cards {
            let class = self.class_of(c);
            if !seen[class] {
                seen[class] = true;
                out.push(c);
            }
        }
        out
    }

    /// `combinations(cards, k)`, keeping only the first combination for each
    /// distinct *multiset of classes* (so picking Tiny+Tiny+Big is offered
    /// once, however many interchangeable Tiny cards back it).
    fn class_distinct_combinations(&self, cards: &[CardId], k: usize) -> Vec<Vec<CardId>> {
        let mut seen: Vec<u64> = Vec::new();
        let mut out = Vec::new();
        for combo in combinations(cards, k) {
            let key = self.class_multiset_key(&combo);
            if !seen.contains(&key) {
                seen.push(key);
                out.push(combo);
            }
        }
        out
    }

    /// Packs a card multiset into one integer: four bits of count per class.
    ///
    /// Deduplication runs inside the legal-action enumerator, which is the
    /// engine's hottest loop, so the key has to be allocation-free - an
    /// earlier sorted-`Vec<u8>` version cost more than the duplicate actions
    /// it was removing.
    #[inline]
    fn class_multiset_key(&self, cards: &[CardId]) -> u64 {
        debug_assert!(
            cards.len() <= 15,
            "four bits per class holds at most 15 of a kind"
        );
        let mut key = 0u64;
        for &card in cards {
            key += 1u64 << (4 * self.class_of(card));
        }
        key
    }

    /// Every way the active player can split exactly three of their own
    /// hand cards between "my pile" and "their pile": each 3-card subset
    /// of the hand, crossed with every one of the 2^3 ways to assign each
    /// of those three specific cards to a side. This naturally covers every
    /// size split (3/0, 2/1, 1/2, 0/3), since which specific card ends up
    /// on which side is itself a real, distinct choice.
    fn legal_two_player_splits(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        let mut seen: Vec<(u64, u64)> = Vec::new();
        for combo in self.class_distinct_combinations(&self.players[player].hand, 3) {
            let combo_key = self.class_multiset_key(&combo);
            for mask in 0u8..8 {
                // Which *class* lands on which side is the real decision, so
                // the offered classes plus which of them are kept identify
                // the offer; masks that only shuffle interchangeable cards
                // between the same two sides are the same offer.
                let mut own_key = 0u64;
                for (i, &card) in combo.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        own_key += 1u64 << (4 * self.class_of(card));
                    }
                }
                if seen.contains(&(combo_key, own_key)) {
                    continue;
                }
                seen.push((combo_key, own_key));

                let mut own_pile = Vec::new();
                let mut other_pile = Vec::new();
                for (i, &card) in combo.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        own_pile.push(card);
                    } else {
                        other_pile.push(card);
                    }
                }
                actions.push(Action::TwoPlayerSubmitDeal {
                    own_pile,
                    other_pile,
                });
            }
        }
        actions
    }

    /// Every legal way `player` can respond during the thingamabob window, including passing.
    fn legal_thingamabob_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = vec![Action::PassThingamabobWindow];
        // one representative per thingamabob class, a second copy of the
        // same card offers an identical menu

        let mut seen_kind = [false; NUM_CARD_CLASSES];
        for &card in &self.players[player].collection {
            let kind = match self.kind_of(card) {
                CardKind::Thingamabob(k) => k,
                _ => continue,
            };
            let class = self.kind_of(card).class_index();
            if seen_kind[class] {
                continue;
            }
            seen_kind[class] = true;
            let effect = self.catalog.thingamabob_effect(kind);
            match effect {
                ThingamabobEffect::StealPointTokenFromRicherPlayer => {
                    let my_tokens = self.players[player].point_tokens;
                    for (target, p) in self.players.iter().enumerate() {
                        if target != player && p.point_tokens > my_tokens {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::PlatonicIsolator {
                                    target_player: target,
                                },
                            });
                        }
                    }
                }
                ThingamabobEffect::RemoveCardsFromDeals {
                    max_cards,
                    max_deals,
                } => {
                    for removals in self.removal_combinations(max_cards, max_deals) {
                        actions.push(Action::PlayThingamabob {
                            card,
                            params: ThingamabobParams::RemoveFromDeals { removals },
                        });
                    }
                }
                ThingamabobEffect::AddHiddenHandCardToDeal => {
                    // the player picks which of their own cards to give away,
                    // so identical hand cards are one choice, not several
                    for hand_card in self.class_distinct(&self.players[player].hand) {
                        for deal in &self.deals {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::CryptozooticExpander {
                                    hand_card,
                                    target_deal: deal.seller,
                                },
                            });
                        }
                    }
                }
                ThingamabobEffect::RevealCardInDeal => {
                    for deal in &self.deals {
                        for hidden in deal.hidden_cards() {
                            actions.push(Action::PlayThingamabob {
                                card,
                                params: ThingamabobParams::SpectroelectricOptimeter {
                                    target_deal: deal.seller,
                                    target_card: hidden,
                                },
                            });
                        }
                    }
                }
            }
        }
        actions
    }

    /// All valid `(seller, card)` removal sets of size `0..=max_cards`,
    /// touching at most `max_deals` distinct deals.
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
                // combinations preserves order and all_targets is grouped
                // by deal, so distinct sellers are just the runs in combo,
                // no set allocation needed per candidate
                let distinct_deals = combo.windows(2).filter(|w| w[0].0 != w[1].0).count()
                    + usize::from(!combo.is_empty());
                if distinct_deals <= max_deals as usize {
                    results.push(combo);
                }
            }
        }
        results
    }

    // apply_action

    /// Applies one action for `player`, mutating game state and returning
    /// the resulting events, or an error if the action isn't legal right now.
    pub fn apply_action(
        &mut self,
        player: PlayerId,
        action: Action,
    ) -> Result<Vec<Event>, RulesError> {
        if self.is_game_over() {
            return Err(RulesError::GameAlreadyOver);
        }
        if self.active_player() != Some(player) {
            return Err(RulesError::NotActivePlayer(player));
        }

        let events = match action {
            Action::SubmitDeal { cards } => self.apply_submit_deal(player, cards),
            Action::TwoPlayerSubmitDeal {
                own_pile,
                other_pile,
            } => self.apply_two_player_submit_deal(player, own_pile, other_pile),
            Action::RevealCard { card } => self.apply_reveal_card(player, card),
            Action::BuyerPeek { target_seller } => self.apply_buyer_peek(player, target_seller),
            Action::PlayThingamabob { card, params } => {
                self.apply_play_thingamabob(player, card, params)
            }
            Action::PassThingamabobWindow => self.apply_pass_thingamabob(player),
            Action::ChooseDeal { seller } => self.apply_choose_deal(player, seller),
            Action::RespondToDeal { reverse } => self.apply_respond_to_deal(player, reverse),
            Action::ResolveNastyPenalty { taken_cards } => {
                self.apply_resolve_nasty_penalty(player, taken_cards)
            }
        }?;
        self.remember(&events);
        Ok(events)
    }

    /// Folds a micro-step's events into the episode memory the observation
    /// encoder exposes, see the `stats` module. A rules error leaves state
    /// untouched and so never reaches here.
    fn remember(&mut self, events: &[Event]) {
        for slot in self.event_memory.iter_mut() {
            *slot *= EVENT_MEMORY_DECAY;
        }
        for event in events {
            self.event_memory[event_kind_index(event)] += 1.0;
            crate::stats::record(&mut self.stats, event);
            if matches!(event, Event::TurnLeaderPassed { .. }) {
                self.turn_index += 1;
            }
        }
    }

    /// Buyer-mode only: a seller submits their three-card deal.
    fn apply_submit_deal(
        &mut self,
        player: PlayerId,
        cards: [CardId; 3],
    ) -> Result<Vec<Event>, RulesError> {
        let Phase::DealOffer(s) = &self.phase else {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        };
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

        let Phase::DealOffer(s) = &mut self.phase else {
            unreachable!()
        };
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

    /// Two-player mode only: the active player's whole deal-offer
    /// micro-turn, in one shot. Splits exactly three of their own hand
    /// cards between the two piles (any split summing to three), removes
    /// all three from hand, and creates a [`Deal`] per non-empty pile (none
    /// at all if a pile is empty, since there's nothing to award there
    /// either way).
    fn apply_two_player_submit_deal(
        &mut self,
        player: PlayerId,
        own_pile: Vec<CardId>,
        other_pile: Vec<CardId>,
    ) -> Result<Vec<Event>, RulesError> {
        let Phase::TwoPlayerDealOffer(s) = &self.phase else {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        };
        if s.step != TwoPlayerDealStep::AwaitingSplit {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        let all: Vec<CardId> = own_pile.iter().chain(other_pile.iter()).copied().collect();
        if all.len() != 3 {
            return Err(RulesError::DealMustBeThreeDistinctCards);
        }
        let unique: std::collections::HashSet<_> = all.iter().collect();
        if unique.len() != 3 {
            return Err(RulesError::DealMustBeThreeDistinctCards);
        }
        for &c in &all {
            if !self.players[player].hand.contains(&c) {
                return Err(RulesError::CardNotInHand(c));
            }
        }
        for &c in &all {
            self.players[player].remove_from_hand(c);
        }

        let other = 1 - player;
        if !own_pile.is_empty() {
            self.deals.push(Deal::from_pile(player, own_pile));
        }
        if !other_pile.is_empty() {
            self.deals.push(Deal::from_pile(other, other_pile));
        }

        let Phase::TwoPlayerDealOffer(s) = &mut self.phase else {
            unreachable!()
        };
        s.step = TwoPlayerDealStep::AwaitingReveal;
        Ok(vec![Event::DealSubmitted { player }])
    }

    /// Reveals one of a just-submitted deal's cards face up.
    fn apply_reveal_card(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<Vec<Event>, RulesError> {
        match &self.phase {
            Phase::DealOffer(s) if s.step == DealOfferStep::AwaitingReveal => {
                let deal = self
                    .deal_for_seller_mut(player)
                    .ok_or(RulesError::NoSuchDeal(player))?;
                if !deal.cards.iter().any(|c| c.card == card) {
                    return Err(RulesError::RevealCardNotInDeal);
                }
                deal.reveal(card);
                self.advance_deal_offer();
                let mut events = vec![Event::CardRevealed {
                    seller: player,
                    card,
                }];
                events.extend(self.maybe_finish_deal_offer());
                Ok(events)
            }
            Phase::TwoPlayerDealOffer(s) if s.step == TwoPlayerDealStep::AwaitingReveal => {
                let deal = self
                    .deals
                    .iter_mut()
                    .find(|d| d.cards.iter().any(|c| c.card == card))
                    .ok_or(RulesError::RevealCardNotInDeal)?;
                deal.reveal(card);
                self.start_thingamabob_window();
                Ok(vec![Event::CardRevealed {
                    seller: player,
                    card,
                }])
            }
            _ => Err(RulesError::WrongPhaseAction(self.phase.name())),
        }
    }

    /// Moves the buyer-mode deal-offer sequence to the next seller.
    fn advance_deal_offer(&mut self) {
        let Phase::DealOffer(s) = &mut self.phase else {
            unreachable!()
        };
        s.idx += 1;
        s.step = DealOfferStep::AwaitingSubmit;
    }

    /// Buyer-mode only. Two-player mode's deal-offer never queues multiple
    /// entries, so it moves straight to the thingamabob window from
    /// [`Self::apply_reveal_card`] instead of going through this.
    fn maybe_finish_deal_offer(&mut self) -> Vec<Event> {
        let Phase::DealOffer(s) = &self.phase else {
            unreachable!()
        };
        if !s.is_done() {
            return vec![];
        }
        self.phase = Phase::BuyerPeek;
        vec![]
    }

    /// Buyer-mode only: the buyer's extra peek at a target seller's hidden cards.
    fn apply_buyer_peek(
        &mut self,
        player: PlayerId,
        target_seller: PlayerId,
    ) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::BuyerPeek) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        debug_assert_eq!(player, self.turn_leader);
        let hidden: Vec<CardId> = self
            .deal_for_seller(target_seller)
            .ok_or(RulesError::NoSuchDeal(target_seller))?
            .hidden_cards()
            .collect();
        if hidden.is_empty() {
            return Err(RulesError::NoSuchDeal(target_seller));
        }
        // the rulebook doesn't specify which hidden card gets flipped,
        // resolved uniformly at random, a flagged assumption, see the readme
        let idx = self.rng.gen_index(hidden.len());
        let card = hidden[idx];
        self.deal_for_seller_mut(target_seller)
            .unwrap()
            .reveal(card);
        self.start_thingamabob_window();
        Ok(vec![Event::BuyerPeeked {
            target_seller,
            card,
        }])
    }

    /// A player passes during the thingamabob window; closes the window once everyone has passed in a row.
    fn apply_pass_thingamabob(&mut self, player: PlayerId) -> Result<Vec<Event>, RulesError> {
        let Phase::ThingamabobWindow(s) = &mut self.phase else {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        };
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

    /// Plays a thingamabob card from the collection during the response window.
    fn apply_play_thingamabob(
        &mut self,
        player: PlayerId,
        card: CardId,
        params: ThingamabobParams,
    ) -> Result<Vec<Event>, RulesError> {
        if !matches!(self.phase, Phase::ThingamabobWindow(_)) {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        }
        if !self.players[player].collection.contains(&card) {
            return Err(RulesError::CardNotInCollection(card));
        }
        let kind = match self.kind_of(card) {
            CardKind::Thingamabob(k) => k,
            _ => return Err(RulesError::NotAThingamabob(card)),
        };
        let effect = self.catalog.thingamabob_effect(kind);

        let mut events = self.resolve_thingamabob_effect(player, effect, &params)?;

        self.players[player].remove_from_collection(card);
        self.discard_pile.push(card);
        events.insert(0, Event::ThingamabobPlayed { player, card, kind });

        let Phase::ThingamabobWindow(s) = &mut self.phase else {
            unreachable!()
        };
        s.consecutive_passes = 0;
        Ok(events)
    }

    /// Applies one thingamabob's effect given its parameters, checking that the params match the effect.
    fn resolve_thingamabob_effect(
        &mut self,
        player: PlayerId,
        effect: ThingamabobEffect,
        params: &ThingamabobParams,
    ) -> Result<Vec<Event>, RulesError> {
        match (effect, params) {
            (
                ThingamabobEffect::StealPointTokenFromRicherPlayer,
                ThingamabobParams::PlatonicIsolator { target_player },
            ) => {
                let target_player = *target_player;
                if target_player == player {
                    return Err(RulesError::InvalidThingamabobParams);
                }
                if self.players[target_player].point_tokens <= self.players[player].point_tokens {
                    return Err(RulesError::IsolatorTargetNotRicher);
                }
                let taken = try_take_point_tokens(&mut self.players[target_player], 1);
                self.players[player].point_tokens += taken;
                Ok(vec![Event::PointTokenStolen {
                    from: target_player,
                    to: player,
                    amount: taken,
                }])
            }
            (
                ThingamabobEffect::RemoveCardsFromDeals {
                    max_cards,
                    max_deals,
                },
                ThingamabobParams::RemoveFromDeals { removals },
            ) => {
                if removals.len() > max_cards as usize {
                    return Err(RulesError::TooManyRemovals);
                }
                let distinct: std::collections::HashSet<_> =
                    removals.iter().map(|(s, _)| *s).collect();
                if distinct.len() > max_deals as usize {
                    return Err(RulesError::TooManyRemovals);
                }
                let card_set: std::collections::HashSet<_> =
                    removals.iter().map(|(_, c)| *c).collect();
                if card_set.len() != removals.len() {
                    return Err(RulesError::DuplicateRemovalTarget);
                }
                let mut removed = Vec::new();
                for (seller, card) in removals {
                    let deal = self
                        .deal_for_seller_mut(*seller)
                        .ok_or(RulesError::NoSuchDeal(*seller))?;
                    if !deal.remove_card(*card) {
                        return Err(RulesError::CardNotHiddenInDeal(*card));
                    }
                    self.discard_pile.push(*card);
                    removed.push(*card);
                }
                Ok(vec![Event::CardsDiscarded { cards: removed }])
            }
            (
                ThingamabobEffect::AddHiddenHandCardToDeal,
                ThingamabobParams::CryptozooticExpander {
                    hand_card,
                    target_deal,
                },
            ) => {
                if !self.players[player].hand.contains(hand_card) {
                    return Err(RulesError::CardNotInHand(*hand_card));
                }
                if self.deal_for_seller(*target_deal).is_none() {
                    return Err(RulesError::NoSuchDeal(*target_deal));
                }
                self.players[player].remove_from_hand(*hand_card);
                self.deal_for_seller_mut(*target_deal)
                    .unwrap()
                    .add_hidden_card(*hand_card);
                Ok(vec![])
            }
            (
                ThingamabobEffect::RevealCardInDeal,
                ThingamabobParams::SpectroelectricOptimeter {
                    target_deal,
                    target_card,
                },
            ) => {
                let deal = self
                    .deal_for_seller_mut(*target_deal)
                    .ok_or(RulesError::NoSuchDeal(*target_deal))?;
                if !deal.hidden_cards().any(|c| c == *target_card) {
                    return Err(RulesError::CardNotHiddenInDeal(*target_card));
                }
                deal.reveal(*target_card);
                Ok(vec![Event::CardRevealed {
                    seller: *target_deal,
                    card: *target_card,
                }])
            }
            _ => Err(RulesError::InvalidThingamabobParams),
        }
    }

    /// Buyer-mode only: the buyer commits to one seller's deal, awarding cards and passing the turn leader.
    fn apply_choose_deal(
        &mut self,
        player: PlayerId,
        seller: PlayerId,
    ) -> Result<Vec<Event>, RulesError> {
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
            let recipient = if deal.seller == seller {
                buyer
            } else {
                deal.seller
            };
            self.players[recipient]
                .collection
                .extend(cards.iter().copied());
            events.push(Event::CardsAwarded {
                player: recipient,
                cards,
            });
        }

        self.turn_leader = seller;
        events.push(Event::TurnLeaderPassed { new_leader: seller });
        events.extend(self.begin_trade_in());
        Ok(events)
    }

    /// Two-player mode only: the responder accepts or reverses the active player's offer.
    fn apply_respond_to_deal(
        &mut self,
        player: PlayerId,
        reverse: bool,
    ) -> Result<Vec<Event>, RulesError> {
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
            let recipient = if reverse {
                1 - deal.seller
            } else {
                deal.seller
            };
            self.players[recipient]
                .collection
                .extend(cards.iter().copied());
            events.push(Event::CardsAwarded {
                player: recipient,
                cards,
            });
        }

        self.turn_leader = responder;
        events.push(Event::TurnLeaderPassed {
            new_leader: responder,
        });
        events.extend(self.begin_trade_in());
        Ok(events)
    }

    // mandatory trade-in

    /// Starts the post-deal mandatory trade-in sequence.
    fn begin_trade_in(&mut self) -> Vec<Event> {
        self.advance_nasty_resolution()
    }

    /// Scans from scratch for the next completed nasty set requiring a
    /// choice, auto-resolving (discard plus effect) every set that doesn't
    /// need one along the way. Re-invoked after each
    /// [`Action::ResolveNastyPenalty`] so cascades (a stolen card completing
    /// a further set, possibly in the resolver's own collection) are
    /// caught. When nothing is left, moves straight on to creature
    /// trade-in, hand refill, and end of turn.
    fn advance_nasty_resolution(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        loop {
            let mut found = None;
            let configured_kinds: Vec<NastyKind> =
                self.catalog.nasty_set_size.keys().copied().collect();
            'scan: for player in 0..self.num_players {
                for kind in configured_kinds.iter().copied() {
                    let set_size = self.catalog.nasty_set_size(kind);
                    let count = self.players[player]
                        .collection
                        .iter()
                        .filter(|&&c| self.kind_of(c) == CardKind::Nasty(kind))
                        .count();
                    if count as u32 >= set_size {
                        found = Some((player, kind, set_size));
                        break 'scan;
                    }
                }
            }
            let Some((player, kind, set_size)) = found else {
                break;
            };

            let to_discard: Vec<CardId> = self.players[player]
                .collection
                .iter()
                .copied()
                .filter(|&c| self.kind_of(c) == CardKind::Nasty(kind))
                .take(set_size as usize)
                .collect();
            for &c in &to_discard {
                self.players[player].remove_from_collection(c);
            }
            self.discard_pile.extend(to_discard);
            events.push(Event::NastySetTradedIn { player, kind });

            match self.catalog.nasty_effect(kind) {
                NastyEffect::BuyerStealsOnePointToken => {
                    let beneficiary = self.nasty_beneficiary(player);
                    let taken = try_take_point_tokens(&mut self.players[player], 1);
                    self.players[beneficiary].point_tokens += taken;
                    if taken > 0 {
                        events.push(Event::PointTokenStolen {
                            from: player,
                            to: beneficiary,
                            amount: taken,
                        });
                    }
                }
                NastyEffect::BuyerMayStealUpToNCards(_) => {
                    self.phase = Phase::NastyResolution(NastyResolutionState {
                        pending: PendingNasty { player, kind },
                    });
                    return events;
                }
            }
        }
        events.extend(self.finish_creature_trade_in_and_refill());
        events
    }

    /// The beneficiary of a pending nasty trade-in chooses which cards to take.
    fn apply_resolve_nasty_penalty(
        &mut self,
        player: PlayerId,
        taken_cards: Vec<CardId>,
    ) -> Result<Vec<Event>, RulesError> {
        let Phase::NastyResolution(s) = &self.phase else {
            return Err(RulesError::WrongPhaseAction(self.phase.name()));
        };
        let pending = s.pending.clone();
        debug_assert_eq!(player, self.nasty_beneficiary(pending.player));
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
        self.players[player]
            .collection
            .extend(taken.iter().copied());
        let mut events = Vec::new();
        if !taken.is_empty() {
            events.push(Event::CardsStolen {
                from: pending.player,
                to: player,
                cards: taken,
            });
        }

        events.extend(self.advance_nasty_resolution());
        Ok(events)
    }

    /// Trades in every completed creature set for point tokens, refills every hand, and starts the next turn or ends the game.
    fn finish_creature_trade_in_and_refill(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let configured_tiers: Vec<Tier> = self.catalog.creature_set_size.keys().copied().collect();
        for player in 0..self.num_players {
            for tier in configured_tiers.iter().copied() {
                let set_size = self.catalog.creature_set_size(tier);
                loop {
                    let count = self.players[player]
                        .collection
                        .iter()
                        .filter(|&&c| self.kind_of(c) == CardKind::Creature(tier))
                        .count();
                    if (count as u32) < set_size {
                        break;
                    }
                    let to_discard: Vec<CardId> = self.players[player]
                        .collection
                        .iter()
                        .copied()
                        .filter(|&c| self.kind_of(c) == CardKind::Creature(tier))
                        .take(set_size as usize)
                        .collect();
                    for &c in &to_discard {
                        self.players[player].remove_from_collection(c);
                    }
                    self.discard_pile.extend(to_discard);
                    self.players[player].point_tokens += 1;
                    events.push(Event::CreatureSetTradedIn {
                        player,
                        tier,
                        tokens_gained: 1,
                    });
                }
            }
        }

        for player in 0..self.num_players {
            let mut drawn = 0usize;
            while self.players[player].hand.len() < HAND_SIZE {
                if self.draw_pile.is_empty() {
                    if self.discard_pile.is_empty() {
                        break; // total-card starvation edge case, nothing left to draw
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

    /// The winning player, if anyone has met the point threshold with a clear lead.
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

    /// Builds `player`'s filtered [`Observation`] of the current game state.
    pub fn observation_for(&self, player: PlayerId) -> Observation {
        let deals = self
            .deals
            .iter()
            .map(|d| ObservedDeal {
                seller: d.seller,
                revealed_cards: d
                    .cards
                    .iter()
                    .filter(|c| c.revealed)
                    .map(|c| c.card)
                    .collect(),
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

/// `n choose k`, saturating rather than overflowing on absurd inputs.
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
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

/// Recursive backtracking helper for [`combinations`].
fn combinations_helper<T: Clone>(
    items: &[T],
    k: usize,
    start: usize,
    current: &mut Vec<T>,
    results: &mut Vec<Vec<T>>,
) {
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
