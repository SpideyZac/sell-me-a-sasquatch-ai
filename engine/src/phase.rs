//! Turn phase state machine. One [`Phase`] value at a time lives in the game
//! state; `game.rs` drives transitions.

use crate::card::{NastyKind, PlayerId};

/// Where a buyer-mode deal offer is in its submit/reveal cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DealOfferStep {
    /// Waiting for the seller to submit three cards.
    AwaitingSubmit,
    /// Waiting for the seller to reveal one of those cards.
    AwaitingReveal,
}

/// One seller's slot in the buyer-mode deal-offer sequence.
#[derive(Debug, Clone)]
pub struct DealOfferEntry {
    /// The seller for this slot.
    pub player: PlayerId,
    /// Whether this seller still needs to reveal a card.
    pub needs_reveal: bool,
}

/// Buyer-mode only: tracks the full sequence of sellers making offers.
#[derive(Debug, Clone)]
pub struct DealOfferState {
    /// One entry per seller in offer order.
    pub entries: Vec<DealOfferEntry>,
    /// Index of the seller currently acting.
    pub idx: usize,
    /// Whether that seller is submitting or revealing.
    pub step: DealOfferStep,
}

impl DealOfferState {
    /// The seller currently acting.
    pub fn current(&self) -> &DealOfferEntry {
        &self.entries[self.idx]
    }

    /// Whether every seller has finished offering.
    pub fn is_done(&self) -> bool {
        self.idx >= self.entries.len()
    }
}

/// Two-player mode's deal-offer micro-turn: the active player splits three
/// of their own hand cards between "my pile" and "their pile" in one shot
/// ([`Self::AwaitingSplit`]), then reveals exactly one of those three cards
/// face up ([`Self::AwaitingReveal`]) before the thingamabob window opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPlayerDealStep {
    /// Waiting for the active player to split their three cards.
    AwaitingSplit,
    /// Waiting for the active player to reveal one card.
    AwaitingReveal,
}

/// Two-player mode only: tracks the split/reveal cycle for the active player's offer.
#[derive(Debug, Clone)]
pub struct TwoPlayerDealState {
    /// Which half of the cycle is in progress.
    pub step: TwoPlayerDealStep,
}

/// Tracks whose turn it is to play a thingamabob during the response window.
#[derive(Debug, Clone)]
pub struct ThingamabobWindowState {
    /// Turn order for this window.
    pub order: Vec<PlayerId>,
    /// Index into `order` of the player currently up.
    pub turn_idx: usize,
    /// How many players have passed in a row.
    pub consecutive_passes: usize,
}

impl ThingamabobWindowState {
    /// The player currently up in this window.
    pub fn current_player(&self) -> PlayerId {
        self.order[self.turn_idx % self.order.len()]
    }
}

/// A nasty set trade-in whose effect still needs to be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingNasty {
    /// Player who traded in the set.
    pub player: PlayerId,
    /// Which nasty kind was traded in.
    pub kind: NastyKind,
}

/// At most one pending resolution exists at a time. After the acting player
/// resolves it, the engine rescans from scratch (transferred cards can
/// complete further sets, including cascades into the resolver's own
/// collection) before either producing the next resolution state or moving
/// on to creature trade-in.
#[derive(Debug, Clone)]
pub struct NastyResolutionState {
    /// The trade-in awaiting resolution.
    pub pending: PendingNasty,
}

/// One phase of a turn.
#[derive(Debug, Clone)]
pub enum Phase {
    /// Buyer-mode only.
    DealOffer(DealOfferState),
    /// Two-player mode only.
    TwoPlayerDealOffer(TwoPlayerDealState),
    /// Buyer-mode only.
    BuyerPeek,
    /// Players may play thingamabobs in response to the current deal.
    ThingamabobWindow(ThingamabobWindowState),
    /// Buyer-mode only.
    BuyerChoosesDeal,
    /// Two-player mode only.
    RespondToDeal,
    /// A completed nasty set is waiting to be resolved.
    NastyResolution(NastyResolutionState),
    /// The game has ended.
    GameOver {
        /// The winning player.
        winner: PlayerId,
    },
}

impl Phase {
    /// Stable string name for this phase, used for logging and encoding.
    pub fn name(&self) -> &'static str {
        match self {
            Phase::DealOffer(s) => {
                if s.step == DealOfferStep::AwaitingSubmit {
                    "deal_offer_submit"
                } else {
                    "deal_offer_reveal"
                }
            }
            Phase::TwoPlayerDealOffer(s) => {
                if s.step == TwoPlayerDealStep::AwaitingSplit {
                    "deal_offer_submit"
                } else {
                    "deal_offer_reveal"
                }
            }
            Phase::BuyerPeek => "buyer_peek",
            Phase::ThingamabobWindow(_) => "thingamabob_window",
            Phase::BuyerChoosesDeal => "buyer_chooses_deal",
            Phase::RespondToDeal => "respond_to_deal",
            Phase::NastyResolution(_) => "nasty_resolution",
            Phase::GameOver { .. } => "game_over",
        }
    }
}
