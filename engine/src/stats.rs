//! Per-episode *memory* features.
//!
//! A single micro-step observation is a snapshot: it says what the tableau
//! looks like right now, but nothing about how the table got there. That
//! makes the environment a fairly deep POMDP - "this opponent has dumped
//! three Thingamabobs and stolen two tokens already" is exactly the kind of
//! thing a good player tracks and a memoryless policy cannot.
//!
//! Rather than pay for a recurrent policy (which does not compose with
//! action masking in sb3-contrib), the engine keeps the memory itself: every
//! `Event` the rules produce folds into a small per-player behavioral
//! summary plus a decayed histogram of recent event kinds. Both are then
//! encoded into the observation (see `encode.rs`), so an ordinary
//! feed-forward policy still sees a running history of the episode.

use crate::card::PlayerId;
use crate::game::Event;

/// One slot per `Event` variant.
pub const NUM_EVENT_KINDS: usize = 17;

/// Per-micro-step decay of `GameState::event_memory`. At 0.8 an event still
/// contributes ~10% of its weight ten micro-steps later, which is roughly
/// one full turn - long enough to carry "what happened this turn" without
/// smearing the whole episode into a constant.
pub const EVENT_MEMORY_DECAY: f32 = 0.8;

/// Number of floats `PlayerStats::write_features` emits.
pub const PLAYER_STAT_LEN: usize = 10;

/// Cumulative counts of what one seat has done/suffered this episode.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlayerStats {
    pub deals_submitted: u32,
    pub thingamabobs_played: u32,
    pub tokens_gained: u32,
    pub tokens_lost: u32,
    pub cards_gained: u32,
    pub cards_lost: u32,
    pub nasty_sets_traded: u32,
    pub creature_sets_traded: u32,
    pub turns_led: u32,
    pub deals_bought: u32,
}

/// Squash an unbounded count into `[0, 1)`, monotonically. Keeps every
/// memory feature on the same scale as the rest of the observation without
/// needing a running normalizer or a hand-picked cap per statistic.
#[inline]
fn squash(count: u32) -> f32 {
    let v = count as f32;
    v / (1.0 + v)
}

impl PlayerStats {
    /// Writes exactly `PLAYER_STAT_LEN` features into `out`.
    pub fn write_features(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), PLAYER_STAT_LEN);
        out[0] = squash(self.deals_submitted);
        out[1] = squash(self.thingamabobs_played);
        out[2] = squash(self.tokens_gained);
        out[3] = squash(self.tokens_lost);
        out[4] = squash(self.cards_gained);
        out[5] = squash(self.cards_lost);
        out[6] = squash(self.nasty_sets_traded);
        out[7] = squash(self.creature_sets_traded);
        out[8] = squash(self.turns_led);
        out[9] = squash(self.deals_bought);
    }
}

/// Stable slot for `event_memory`. Order is an implementation detail - only
/// stability across a build matters, since nothing outside this crate names
/// individual slots.
pub fn event_kind_index(event: &Event) -> usize {
    match event {
        Event::DealSubmitted { .. } => 0,
        Event::CardRevealed { .. } => 1,
        Event::BuyerPeeked { .. } => 2,
        Event::ThingamabobPlayed { .. } => 3,
        Event::ThingamabobWindowClosed => 4,
        Event::DealChosen { .. } => 5,
        Event::DealResponded { .. } => 6,
        Event::CardsAwarded { .. } => 7,
        Event::CardsDiscarded { .. } => 8,
        Event::NastySetTradedIn { .. } => 9,
        Event::PointTokenStolen { .. } => 10,
        Event::CardsStolen { .. } => 11,
        Event::CreatureSetTradedIn { .. } => 12,
        Event::HandRefilled { .. } => 13,
        Event::DrawPileReshuffledFromDiscard => 14,
        Event::TurnLeaderPassed { .. } => 15,
        Event::GameOver { .. } => 16,
    }
}

/// Folds one event into the per-player summaries. `stats` is indexed by
/// `PlayerId`; events naming a player out of range are ignored rather than
/// panicking (nothing in the engine produces one, but this keeps the
/// bookkeeping path total).
pub fn record(stats: &mut [PlayerStats], event: &Event) {
    let mut bump = |p: PlayerId, f: fn(&mut PlayerStats)| {
        if let Some(s) = stats.get_mut(p) {
            f(s);
        }
    };
    match event {
        Event::DealSubmitted { player } => bump(*player, |s| s.deals_submitted += 1),
        Event::ThingamabobPlayed { player, .. } => bump(*player, |s| s.thingamabobs_played += 1),
        Event::DealChosen { buyer, .. } => bump(*buyer, |s| s.deals_bought += 1),
        Event::NastySetTradedIn { player, .. } => bump(*player, |s| s.nasty_sets_traded += 1),
        Event::CreatureSetTradedIn { player, .. } => bump(*player, |s| s.creature_sets_traded += 1),
        Event::TurnLeaderPassed { new_leader } => bump(*new_leader, |s| s.turns_led += 1),
        Event::PointTokenStolen { from, to, amount } => {
            if let Some(s) = stats.get_mut(*from) {
                s.tokens_lost += amount;
            }
            if let Some(s) = stats.get_mut(*to) {
                s.tokens_gained += amount;
            }
        }
        Event::CardsStolen { from, to, cards } => {
            let n = cards.len() as u32;
            if let Some(s) = stats.get_mut(*from) {
                s.cards_lost += n;
            }
            if let Some(s) = stats.get_mut(*to) {
                s.cards_gained += n;
            }
        }
        Event::CardsAwarded { player, cards } => {
            let n = cards.len() as u32;
            if let Some(s) = stats.get_mut(*player) {
                s.cards_gained += n;
            }
        }
        Event::CardRevealed { .. }
        | Event::BuyerPeeked { .. }
        | Event::ThingamabobWindowClosed
        | Event::DealResponded { .. }
        | Event::CardsDiscarded { .. }
        | Event::HandRefilled { .. }
        | Event::DrawPileReshuffledFromDiscard
        | Event::GameOver { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_is_bounded_and_monotone() {
        let mut prev = -1.0;
        for n in [0u32, 1, 2, 5, 50, 10_000] {
            let v = squash(n);
            assert!((0.0..1.0).contains(&v), "{n} -> {v} out of range");
            assert!(v > prev, "not monotone at {n}");
            prev = v;
        }
    }

    #[test]
    fn steals_credit_both_sides() {
        let mut stats = vec![PlayerStats::default(); 3];
        record(&mut stats, &Event::PointTokenStolen { from: 2, to: 0, amount: 1 });
        record(&mut stats, &Event::CardsStolen { from: 2, to: 0, cards: vec![7, 8] });
        assert_eq!(stats[0].tokens_gained, 1);
        assert_eq!(stats[2].tokens_lost, 1);
        assert_eq!(stats[0].cards_gained, 2);
        assert_eq!(stats[2].cards_lost, 2);
    }

    #[test]
    fn event_kind_indices_are_unique_and_in_range() {
        let events = [
            Event::DealSubmitted { player: 0 },
            Event::CardRevealed { seller: 0, card: 0 },
            Event::BuyerPeeked { target_seller: 0, card: 0 },
            Event::ThingamabobWindowClosed,
            Event::DealChosen { buyer: 0, seller: 1 },
            Event::DealResponded { active: 0, reverse: false },
            Event::CardsAwarded { player: 0, cards: vec![] },
            Event::CardsDiscarded { cards: vec![] },
            Event::PointTokenStolen { from: 0, to: 1, amount: 1 },
            Event::CardsStolen { from: 0, to: 1, cards: vec![] },
            Event::HandRefilled { player: 0, drawn: 1 },
            Event::DrawPileReshuffledFromDiscard,
            Event::TurnLeaderPassed { new_leader: 0 },
            Event::GameOver { winner: 0 },
        ];
        let mut seen = vec![];
        for e in &events {
            let i = event_kind_index(e);
            assert!(i < NUM_EVENT_KINDS);
            assert!(!seen.contains(&i), "duplicate index {i}");
            seen.push(i);
        }
    }
}
