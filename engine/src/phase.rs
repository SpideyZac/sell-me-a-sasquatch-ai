//! Turn phase state machine (§2.3, §2.7). One `Phase` value at a time lives
//! in `GameState`; `game.rs` drives transitions.

use crate::card::{NastyKind, PlayerId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DealOfferStep {
    AwaitingSubmit,
    AwaitingReveal,
}

/// One seller's (or, in 2-player mode, one player's) slot in the deal-offer
/// sequence. `needs_reveal` is false for the 2-player responder's deal -
/// see README's flagged assumption for §2.7's two-piles interpretation.
#[derive(Debug, Clone)]
pub struct DealOfferEntry {
    pub player: PlayerId,
    pub needs_reveal: bool,
}

#[derive(Debug, Clone)]
pub struct DealOfferState {
    pub entries: Vec<DealOfferEntry>,
    pub idx: usize,
    pub step: DealOfferStep,
}

impl DealOfferState {
    pub fn current(&self) -> &DealOfferEntry {
        &self.entries[self.idx]
    }

    pub fn is_done(&self) -> bool {
        self.idx >= self.entries.len()
    }
}

#[derive(Debug, Clone)]
pub struct ThingamabobWindowState {
    pub order: Vec<PlayerId>,
    pub turn_idx: usize,
    pub consecutive_passes: usize,
}

impl ThingamabobWindowState {
    pub fn current_player(&self) -> PlayerId {
        self.order[self.turn_idx % self.order.len()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingNasty {
    pub player: PlayerId,
    pub kind: NastyKind,
}

/// At most one pending resolution exists at a time: after the acting player
/// resolves it, the engine rescans from scratch (transferred cards can
/// complete further sets, including cascades into the resolver's own
/// Collection) before either producing the next `NastyResolutionState` or
/// moving on to Creature trade-in.
#[derive(Debug, Clone)]
pub struct NastyResolutionState {
    pub pending: PendingNasty,
}

#[derive(Debug, Clone)]
pub enum Phase {
    DealOffer(DealOfferState),
    /// Buyer-mode only.
    BuyerPeek,
    ThingamabobWindow(ThingamabobWindowState),
    /// Buyer-mode only.
    BuyerChoosesDeal,
    /// 2-player mode only.
    RespondToDeal,
    NastyResolution(NastyResolutionState),
    GameOver { winner: PlayerId },
}

impl Phase {
    pub fn name(&self) -> &'static str {
        match self {
            Phase::DealOffer(s) => {
                if s.step == DealOfferStep::AwaitingSubmit {
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
