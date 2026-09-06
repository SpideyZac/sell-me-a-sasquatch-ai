//! In-progress deal state. A deal starts with exactly three cards but can
//! shrink via Detrital Repositioner or Super Detrital Repositioner, and grow
//! back via Cryptozootic Expander.

use crate::card::{CardId, PlayerId};

/// One card sitting in a deal, along with whether it's been flipped face up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DealCard {
    /// The card itself.
    pub card: CardId,
    /// Whether this card has been revealed.
    pub revealed: bool,
}

/// An offer a seller has put together, made up of face-down and face-up cards.
#[derive(Debug, Clone)]
pub struct Deal {
    /// The player who made this deal.
    pub seller: PlayerId,
    /// The cards in the deal.
    pub cards: Vec<DealCard>,
}

impl Deal {
    /// Builds a deal from exactly three cards, all hidden.
    pub fn new(seller: PlayerId, cards: [CardId; 3]) -> Self {
        Self {
            seller,
            cards: cards
                .into_iter()
                .map(|card| DealCard {
                    card,
                    revealed: false,
                })
                .collect(),
        }
    }

    /// Builds a deal from an arbitrary-length pile, all hidden. Used for
    /// two-player mode's split piles, which aren't fixed at three cards each.
    pub fn from_pile(seller: PlayerId, cards: Vec<CardId>) -> Self {
        Self {
            seller,
            cards: cards
                .into_iter()
                .map(|card| DealCard {
                    card,
                    revealed: false,
                })
                .collect(),
        }
    }

    /// Ids of the cards still hidden.
    pub fn hidden_cards(&self) -> impl Iterator<Item = CardId> + '_ {
        self.cards.iter().filter(|c| !c.revealed).map(|c| c.card)
    }

    /// Flips a card face up, returns false if the card isn't in this deal.
    pub fn reveal(&mut self, card: CardId) -> bool {
        if let Some(entry) = self.cards.iter_mut().find(|c| c.card == card) {
            entry.revealed = true;
            true
        } else {
            false
        }
    }

    /// Removes a card from the deal, returns false if it wasn't there.
    pub fn remove_card(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.cards.iter().position(|c| c.card == card) {
            self.cards.remove(pos);
            true
        } else {
            false
        }
    }

    /// Adds a new hidden card to the deal.
    pub fn add_hidden_card(&mut self, card: CardId) {
        self.cards.push(DealCard {
            card,
            revealed: false,
        });
    }

    /// Ids of every card currently in the deal, hidden or revealed.
    pub fn all_card_ids(&self) -> Vec<CardId> {
        self.cards.iter().map(|c| c.card).collect()
    }
}
