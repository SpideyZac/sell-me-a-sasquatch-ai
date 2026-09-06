//! In-progress deal state (§2.3 steps 1-5). A `Deal` starts with exactly 3
//! cards but can shrink via Detrital Repositioner / Super Detrital
//! Repositioner (§2.3 step 4), and grow back via Cryptozooptic Expander.

use crate::card::{CardId, PlayerId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DealCard {
    pub card: CardId,
    pub revealed: bool,
}

#[derive(Debug, Clone)]
pub struct Deal {
    pub seller: PlayerId,
    pub cards: Vec<DealCard>,
}

impl Deal {
    pub fn new(seller: PlayerId, cards: [CardId; 3]) -> Self {
        Self { seller, cards: cards.into_iter().map(|card| DealCard { card, revealed: false }).collect() }
    }

    pub fn hidden_cards(&self) -> impl Iterator<Item = CardId> + '_ {
        self.cards.iter().filter(|c| !c.revealed).map(|c| c.card)
    }

    pub fn reveal(&mut self, card: CardId) -> bool {
        if let Some(entry) = self.cards.iter_mut().find(|c| c.card == card) {
            entry.revealed = true;
            true
        } else {
            false
        }
    }

    pub fn remove_card(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.cards.iter().position(|c| c.card == card) {
            self.cards.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn add_hidden_card(&mut self, card: CardId) {
        self.cards.push(DealCard { card, revealed: false });
    }

    pub fn all_card_ids(&self) -> Vec<CardId> {
        self.cards.iter().map(|c| c.card).collect()
    }
}
