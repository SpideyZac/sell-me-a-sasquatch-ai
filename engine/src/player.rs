//! Per-player state: hand, collection, and point tokens.

use crate::card::CardId;

/// One player's hand, collection, and point token count.
#[derive(Debug, Clone, Default)]
pub struct Player {
    /// Cards currently in hand.
    pub hand: Vec<CardId>,
    /// Cards the player has collected.
    pub collection: Vec<CardId>,
    /// Point tokens currently held.
    pub point_tokens: u32,
}

impl Player {
    /// Builds an empty player.
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes a card from hand, returns false if it wasn't there.
    pub fn remove_from_hand(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.hand.iter().position(|&c| c == card) {
            self.hand.remove(pos);
            true
        } else {
            false
        }
    }

    /// Removes a card from the collection, returns false if it wasn't there.
    pub fn remove_from_collection(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.collection.iter().position(|&c| c == card) {
            self.collection.remove(pos);
            true
        } else {
            false
        }
    }
}

/// Takes up to `n` point tokens from `from`. Stealing against an empty
/// target always fizzles silently per the game's global rule; this is never
/// illegal, just a no-op. Returns the number actually removed. The caller
/// credits it to the taker; this is kept as two steps to sidestep aliasing
/// when both players live in the same `Vec<Player>`.
pub fn try_take_point_tokens(from: &mut Player, n: u32) -> u32 {
    let taken = n.min(from.point_tokens);
    from.point_tokens -= taken;
    taken
}

/// Removes up to `n` specific cards (by id) from `from`'s collection. Cards
/// not actually present are silently skipped as a fizzle rather than an
/// error, so this stays safe even if legality was checked against slightly
/// stale state. Returns the ids actually removed.
pub fn try_take_cards(from: &mut Player, cards: &[CardId], n: u32) -> Vec<CardId> {
    let mut taken = Vec::new();
    for &card in cards.iter().take(n as usize) {
        if from.remove_from_collection(card) {
            taken.push(card);
        }
    }
    taken
}
