//! The action space. Actions only carry ids and indices, never full card
//! data, so they stay compact for both the RL action space and the FFI
//! boundary.

use crate::card::{CardId, PlayerId};

/// Parameters for a thingamabob play, one variant per card that needs extra
/// targeting info beyond the card id itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThingamabobParams {
    /// Platonic Isolator: the target must currently hold strictly more
    /// point tokens than the acting player, enforced in [`crate::game`]'s
    /// legal action generation.
    PlatonicIsolator {
        /// Player being targeted.
        target_player: PlayerId,
    },
    /// Shared by Detrital Repositioner and Super Detrital Repositioner. The
    /// max number of removals allowed comes from the catalog entry for the
    /// specific card played, not from this variant.
    RemoveFromDeals {
        /// Deal-card pairs to remove, one per removal.
        removals: Vec<(PlayerId, CardId)>,
    },
    /// Cryptozootic Expander.
    CryptozooticExpander {
        /// Card taken from the player's hand.
        hand_card: CardId,
        /// Deal the hand card is added to.
        target_deal: PlayerId,
    },
    /// Spectroelectric Optimeter.
    SpectroelectricOptimeter {
        /// Deal being swapped from.
        target_deal: PlayerId,
        /// Card being swapped in.
        target_card: CardId,
    },
}

/// A single move a player can make. Every action variant is a full legal
/// move, not a partial input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Buyer-mode only: a seller's deal-offer micro-turn.
    SubmitDeal {
        /// The three cards offered.
        cards: [CardId; 3],
    },
    /// Two-player mode only: the active player's whole deal-offer
    /// micro-turn. Splits exactly three cards from their own hand between
    /// `own_pile` (theirs again on accept) and `other_pile` (the opponent's
    /// on accept). The split can be any combination summing to three.
    TwoPlayerSubmitDeal {
        /// Cards that stay with the active player if accepted.
        own_pile: Vec<CardId>,
        /// Cards that go to the opponent if accepted.
        other_pile: Vec<CardId>,
    },
    /// Choosing which of the three just-submitted cards to flip face up.
    RevealCard {
        /// Card being revealed.
        card: CardId,
    },
    /// Buyer-mode only: the buyer's extra peek target. Which of the
    /// target's still-hidden cards flips is resolved by the engine's RNG,
    /// see the README's flagged rules assumption.
    BuyerPeek {
        /// Seller being peeked at.
        target_seller: PlayerId,
    },
    /// Playing a thingamabob card.
    PlayThingamabob {
        /// Card being played.
        card: CardId,
        /// Targeting info specific to the card played.
        params: ThingamabobParams,
    },
    /// Declining to play a thingamabob during the response window.
    PassThingamabobWindow,
    /// Buyer-mode only: the buyer's final commit.
    ChooseDeal {
        /// Seller whose deal is chosen.
        seller: PlayerId,
    },
    /// Two-player mode only: the responder's accept or reverse decision.
    RespondToDeal {
        /// True to reverse the deal instead of accepting it.
        reverse: bool,
    },
    /// The new buyer's (two-player mode: responder's) choice of which
    /// card(s) to take for Poison Pill Bug or Loan Shark. `taken_cards`
    /// must be a subset of the target player's collection, sized within
    /// the effect's allowed max.
    ResolveNastyPenalty {
        /// Cards taken from the target's collection.
        taken_cards: Vec<CardId>,
    },
}
