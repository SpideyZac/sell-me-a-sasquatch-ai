//! Action space (§4). All actions carry only ids/indices, never full card
//! data, to keep the RL action space and the FFI boundary compact.

use crate::card::{CardId, PlayerId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThingamabobParams {
    /// Platonic Isolator: target must currently have strictly more Point
    /// Tokens than the acting player (enforced in `legal_actions`).
    PlatonicIsolator { target_player: PlayerId },
    /// Detrital Repositioner / Super Detrital Repositioner share this: the
    /// max cards/deals allowed is a property of the specific card played
    /// (looked up via `Catalog::thingamabob_effect`), not of this variant.
    RemoveFromDeals { removals: Vec<(PlayerId, CardId)> },
    CryptozooticExpander { hand_card: CardId, target_deal: PlayerId },
    SpectroelectricOptimeter { target_deal: PlayerId, target_card: CardId },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Buyer-mode only: a seller's deal-offer micro-turn.
    SubmitDeal { cards: [CardId; 3] },
    /// 2-player mode only (§2.7): the active player's whole deal-offer
    /// micro-turn - splits exactly 3 cards from their own hand between
    /// `own_pile` (theirs again on Accept) and `other_pile` (the opponent's
    /// on Accept); sizes can be any split summing to 3 (3/0, 2/1, 1/2, 0/3).
    TwoPlayerSubmitDeal { own_pile: Vec<CardId>, other_pile: Vec<CardId> },
    /// Choosing which of the 3 just-submitted cards to flip face up.
    RevealCard { card: CardId },
    /// Buyer-mode only: the Buyer's extra peek target (§2.3 step 3). Which
    /// of the target's still-hidden cards flips is resolved by the engine's
    /// RNG - see README's flagged rules assumption.
    BuyerPeek { target_seller: PlayerId },
    PlayThingamabob { card: CardId, params: ThingamabobParams },
    PassThingamabobWindow,
    /// Buyer-mode only: the Buyer's final commit (§2.3 step 5).
    ChooseDeal { seller: PlayerId },
    /// 2-player mode only (§2.7): the responder's Accept/Reverse decision.
    RespondToDeal { reverse: bool },
    /// New-Buyer's (2-player mode: responder's) choice of which card(s) to
    /// take for Poison Pill Bug / Loan Shark (§2.4). `taken_cards` must be a
    /// subset (size 0..=effect max) of the target player's Collection.
    ResolveNastyPenalty { taken_cards: Vec<CardId> },
}
