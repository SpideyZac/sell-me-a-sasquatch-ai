//! Per-player state: hand, Collection, Point Tokens (§2.2/§3.2).

use crate::card::CardId;

#[derive(Debug, Clone, Default)]
pub struct Player {
    pub hand: Vec<CardId>,
    pub collection: Vec<CardId>,
    pub point_tokens: u32,
}

impl Player {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn remove_from_hand(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.hand.iter().position(|&c| c == card) {
            self.hand.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn remove_from_collection(&mut self, card: CardId) -> bool {
        if let Some(pos) = self.collection.iter().position(|&c| c == card) {
            self.collection.remove(pos);
            true
        } else {
            false
        }
    }
}

/// Take up to `n` Point Tokens from `from`. Per the §2.4 global rule,
/// stealing against an empty target always fizzles silently - this is never
/// illegal, just a no-op. Returns the number actually removed; the caller
/// credits it to the taker (kept as two steps to sidestep aliasing when both
/// players live in the same `Vec<Player>`).
pub fn try_take_point_tokens(from: &mut Player, n: u32) -> u32 {
    let taken = n.min(from.point_tokens);
    from.point_tokens -= taken;
    taken
}

/// Remove up to `n` specific cards (by id) from `from`'s Collection. Cards
/// not actually present are silently skipped (fizzle, not an error) so this
/// stays safe even if legality was checked against slightly stale state.
/// Returns the ids actually removed.
pub fn try_take_cards(from: &mut Player, cards: &[CardId], n: u32) -> Vec<CardId> {
    let mut taken = Vec::new();
    for &card in cards.iter().take(n as usize) {
        if from.remove_from_collection(card) {
            taken.push(card);
        }
    }
    taken
}
