//! Fixed-shape observation encoding, done in Rust.
//!
//! The Python side used to rebuild this vector card by card, calling back
//! across the FFI boundary once per card just to learn its kind. Encoding
//! here instead means a micro-step costs exactly one boundary crossing and
//! one `memcpy` into a numpy buffer.
//!
//! Three properties matter more than the exact field list:
//!
//! * **Fixed shape for every table size.** Everything is padded out to
//!   [`MAX_PLAYERS`] / [`MAX_DEALS`] with explicit validity flags, so one
//!   policy trains and plays across 2- through 6-player games (2-player
//!   being the materially different variant) instead of one model per
//!   table size.
//! * **Ego-centric.** Seat `k` in the encoding is always "the player `k`
//!   seats after me", never absolute seat `k`. A policy therefore cannot
//!   learn seat-specific habits, and what it learns at one seat transfers to
//!   every other.
//! * **Class counts, not card slots.** Hands and collections are unordered
//!   sets; encoding them as per-class counts is permutation-invariant, and
//!   an order of magnitude smaller than one padded slot per card.

use crate::{
    action::{Action, ThingamabobParams},
    card::{CardId, CardKind, NastyKind, PlayerId, Tier, NUM_CARD_CLASSES},
    game::{GameMode, GameState},
    phase::Phase,
    stats::{NUM_EVENT_KINDS, PLAYER_STAT_LEN},
};

/// Largest supported table size.
pub const MAX_PLAYERS: usize = 6;
/// Smallest supported table size.
pub const MIN_PLAYERS: usize = 2;
/// Buyer mode runs at most `num_players - 1` deals and two-player mode
/// exactly two, so 6 is comfortable headroom on the real maximum of 5.
pub const MAX_DEALS: usize = 6;
/// Number of distinct turn phases.
pub const NUM_PHASES: usize = 8;

/// Per-episode random "persona" vector. The engine writes zeros here; the
/// training wrapper fills it with a fresh sample each episode.
///
/// Without it, a deterministic-given-state policy plays the same opening
/// every time from the same deal, which is both exploitable and a poor
/// explorer. Conditioning on a latent that is constant *within* an episode
/// but resampled *across* episodes lets one set of weights express a family
/// of coherent strategies and pick one per game, rather than re-rolling its
/// personality on every micro-turn.
pub const NOISE_LEN: usize = 8;

/// Width of the global feature block.
const GLOBAL_LEN: usize = NUM_PHASES + 2 + (MAX_PLAYERS - MIN_PLAYERS + 1) + 7 + MAX_PLAYERS;
/// Width of the observer's own hand block.
const HAND_LEN: usize = NUM_CARD_CLASSES + 1;
/// Width of one player's block.
const PER_PLAYER_LEN: usize = 1 + NUM_CARD_CLASSES + 6 + 4 + 3 + PLAYER_STAT_LEN;
/// Width of one deal's block.
const PER_DEAL_LEN: usize = 1 + MAX_PLAYERS + 1 + NUM_CARD_CLASSES + 2;
/// Width of the card-counting block.
const COUNTING_LEN: usize = 2 * NUM_CARD_CLASSES + 1;

/// Offset of the trailing noise block within the observation.
pub const NOISE_OFFSET: usize = GLOBAL_LEN
    + HAND_LEN
    + MAX_PLAYERS * PER_PLAYER_LEN
    + MAX_DEALS * PER_DEAL_LEN
    + COUNTING_LEN
    + NUM_EVENT_KINDS;

/// Total width of [`GameState::encode_observation`].
pub const OBS_LEN: usize = NOISE_OFFSET + NOISE_LEN;

/// Saturating squash of an unbounded count into `[0, 1)`.
#[inline]
fn squash(v: f32) -> f32 {
    v / (1.0 + v)
}

/// Writes `1.0` at `slot` within the `len`-wide one-hot block at `out[..len]`.
#[inline]
fn one_hot(out: &mut [f32], slot: usize) {
    if slot < out.len() {
        out[slot] = 1.0;
    }
}

/// Cursor over the output buffer: every block below takes the next `n`
/// floats and advances, so adding a field can never silently overlap the
/// next block.
struct Writer<'a> {
    /// The buffer being written into.
    buf: &'a mut [f32],
    /// Current write position.
    at: usize,
}

impl<'a> Writer<'a> {
    /// Takes the next `n` floats and advances past them.
    fn take(&mut self, n: usize) -> &mut [f32] {
        let start = self.at;
        self.at += n;
        &mut self.buf[start..self.at]
    }

    /// Writes one float and advances past it.
    fn put(&mut self, v: f32) {
        self.buf[self.at] = v;
        self.at += 1;
    }
}

impl GameState {
    /// Fills `out` (exactly `OBS_LEN` floats) with `player`'s view of the
    /// game. Leaves the trailing `NOISE_LEN` slots at zero for the caller.
    pub fn encode_observation(&self, player: PlayerId, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            OBS_LEN,
            "observation buffer must be OBS_LEN wide"
        );
        out.fill(0.0);
        let n = self.num_players;
        let deck_total: f32 = self.deck_counts.iter().sum::<u32>().max(1) as f32;
        let threshold = self.win_threshold.max(1) as f32;
        let active = self.active_player();
        let mut w = Writer { buf: out, at: 0 };

        // global block
        one_hot(w.take(NUM_PHASES), phase_index(&self.phase));
        one_hot(
            w.take(2),
            if self.mode == GameMode::TwoPlayer {
                1
            } else {
                0
            },
        );
        one_hot(w.take(MAX_PLAYERS - MIN_PLAYERS + 1), n - MIN_PLAYERS);
        w.put(self.win_threshold as f32 / MAX_PLAYERS as f32);
        w.put(self.draw_pile.len() as f32 / deck_total);
        w.put(self.discard_pile.len() as f32 / deck_total);
        w.put(squash(self.turn_index as f32 / 4.0));
        w.put(self.deals.len() as f32 / MAX_DEALS as f32);
        w.put(f32::from(active == Some(player)));
        w.put(f32::from(self.turn_leader == player));
        one_hot(w.take(MAX_PLAYERS), (self.turn_leader + n - player) % n);

        // observer's own hand
        let hand = &self.players[player].hand;
        let hand_counts = w.take(NUM_CARD_CLASSES);
        self.tally_into(hand, hand_counts);
        for slot in hand_counts.iter_mut() {
            *slot /= crate::game::HAND_SIZE as f32;
        }
        w.put(hand.len() as f32 / crate::game::HAND_SIZE as f32);

        // players, in ego order, slot 0 is always "me". `seen` doubles as
        // the card-counting accumulator: every class the observer can
        // legitimately account for lands in it
        let mut seen = [0f32; NUM_CARD_CLASSES];
        self.tally_into(hand, &mut seen);
        for offset in 0..MAX_PLAYERS {
            let block = w.take(PER_PLAYER_LEN);
            if offset >= n {
                continue; // padding seat, validity flag stays 0
            }
            let seat = (player + offset) % n;
            let p = &self.players[seat];
            let mut counts = [0f32; NUM_CARD_CLASSES];
            self.tally_into(&p.collection, &mut counts);
            for (class, &c) in counts.iter().enumerate() {
                seen[class] += c;
            }

            block[0] = 1.0;
            for (class, &c) in counts.iter().enumerate() {
                block[1 + class] = c / 8.0;
            }
            let mut cursor = 1 + NUM_CARD_CLASSES;
            block[cursor] = squash(p.collection.len() as f32 / 8.0);
            block[cursor + 1] = p.point_tokens as f32 / threshold;
            block[cursor + 2] =
                ((self.win_threshold as f32 - p.point_tokens as f32) / threshold).max(0.0);
            block[cursor + 3] = f32::from(seat == self.turn_leader);
            block[cursor + 4] = f32::from(active == Some(seat));
            // lead margin against the best other seat, the quantity that
            // actually has to go positive to win
            let best_other = self
                .players
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != seat)
                .map(|(_, q)| q.point_tokens)
                .max()
                .unwrap_or(0);
            block[cursor + 5] = (p.point_tokens as f32 - best_other as f32) / threshold;
            cursor += 6;

            // how close this seat is to completing each set: creature sets
            // score a point token, nasty sets fire a penalty
            for (i, tier) in Tier::ALL.iter().enumerate() {
                let have = counts[CardKind::Creature(*tier).class_index()];
                let size = self.catalog.creature_set_size(*tier).max(1) as f32;
                block[cursor + i] = (have / size).min(1.0);
            }
            cursor += 4;
            for (i, kind) in NastyKind::ALL.iter().enumerate() {
                let have = counts[CardKind::Nasty(*kind).class_index()];
                let size = self.catalog.nasty_set_size(*kind).max(1) as f32;
                block[cursor + i] = (have / size).min(1.0);
            }
            cursor += 3;

            self.stats[seat].write_features(&mut block[cursor..cursor + PLAYER_STAT_LEN]);
        }

        // deals
        for slot in 0..MAX_DEALS {
            let block = w.take(PER_DEAL_LEN);
            let Some(deal) = self.deals.get(slot) else {
                continue;
            };
            let mut counts = [0f32; NUM_CARD_CLASSES];
            let mut num_revealed = 0f32;
            for c in deal.cards.iter().filter(|c| c.revealed) {
                counts[self.class_of(c.card)] += 1.0;
                seen[self.class_of(c.card)] += 1.0;
                num_revealed += 1.0;
            }

            block[0] = 1.0;
            one_hot(
                &mut block[1..1 + MAX_PLAYERS],
                (deal.seller + n - player) % n,
            );
            let mut cursor = 1 + MAX_PLAYERS;
            block[cursor] = f32::from(deal.seller == player);
            cursor += 1;
            for (class, &c) in counts.iter().enumerate() {
                block[cursor + class] = c / 4.0;
            }
            cursor += NUM_CARD_CLASSES;
            block[cursor] = num_revealed / 4.0;
            block[cursor + 1] = (deal.cards.len() as f32 - num_revealed) / 4.0;
        }

        // card counting. discards are face up at a real table, so they
        // count as seen; the remainder is genuinely unknown (draw pile,
        // other hands, hidden deal cards) and is what a card counter
        // actually tracks
        let mut discard = [0f32; NUM_CARD_CLASSES];
        self.tally_into(&self.discard_pile, &mut discard);
        for (class, &c) in discard.iter().enumerate() {
            seen[class] += c;
        }
        let out_discard = w.take(NUM_CARD_CLASSES);
        for (class, slot) in out_discard.iter_mut().enumerate() {
            *slot = discard[class] / self.deck_counts[class].max(1) as f32;
        }
        let mut unseen = [0f32; NUM_CARD_CLASSES];
        let mut unseen_total = 0f32;
        for class in 0..NUM_CARD_CLASSES {
            let remaining = (self.deck_counts[class] as f32 - seen[class]).max(0.0);
            unseen_total += remaining;
            unseen[class] = remaining / self.deck_counts[class].max(1) as f32;
        }
        w.take(NUM_CARD_CLASSES).copy_from_slice(&unseen);
        w.put(unseen_total / deck_total);

        // episode memory
        let memory = w.take(NUM_EVENT_KINDS);
        for (slot, &v) in memory.iter_mut().zip(self.event_memory.iter()) {
            *slot = squash(v);
        }

        debug_assert_eq!(w.at, NOISE_OFFSET);
    }

    /// Adds one count per card in `cards` to its class slot in `counts`.
    fn tally_into(&self, cards: &[CardId], counts: &mut [f32]) {
        for &c in cards {
            counts[self.class_of(c)] += 1.0;
        }
    }
}

/// Stable slot for a phase in the one-hot phase block.
fn phase_index(phase: &Phase) -> usize {
    match phase.name() {
        "deal_offer_submit" => 0,
        "deal_offer_reveal" => 1,
        "buyer_peek" => 2,
        "thingamabob_window" => 3,
        "buyer_chooses_deal" => 4,
        "respond_to_deal" => 5,
        "nasty_resolution" => 6,
        _ => 7, // game_over
    }
}

// action features

/// Width of one action's feature row, see [`GameState::encode_actions`].
pub const ACTION_FEAT_LEN: usize = 43;

/// Offset of the action-type one-hot block.
const AF_TYPE: usize = 0;
/// Number of distinct action types.
const AF_NUM_TYPES: usize = 9;
/// Offset of the committed-cards class-count block.
const AF_COMMITTED: usize = AF_TYPE + AF_NUM_TYPES;
/// Offset of the target-cards class-count block.
const AF_TARGET_CARDS: usize = AF_COMMITTED + NUM_CARD_CLASSES;
/// Offset of the target-seat one-hot block.
const AF_TARGET_SEAT: usize = AF_TARGET_CARDS + NUM_CARD_CLASSES;
/// Offset of the target-is-self flag.
const AF_TARGET_IS_SELF: usize = AF_TARGET_SEAT + MAX_PLAYERS;
/// Offset of the hidden-target count.
const AF_HIDDEN_TARGETS: usize = AF_TARGET_IS_SELF + 1;
/// Offset of the reverse-decision flag.
const AF_REVERSE: usize = AF_HIDDEN_TARGETS + 1;
/// Offset of the deals-touched count.
const AF_DEALS_TOUCHED: usize = AF_REVERSE + 1;

impl GameState {
    /// Describes what each currently-legal action does, one
    /// [`ACTION_FEAT_LEN`]-wide row per action.
    ///
    /// The action space is ordinal; index `i` means "the i-th entry of
    /// [`GameState::legal_actions`] right now", which keeps the space
    /// fixed-width without a combinatorial encoding of the nested [`Action`]
    /// type, but on its own leaves the policy guessing, since the same
    /// index means a different move in every state. These rows close that
    /// gap. A policy scores each candidate from its own description (see
    /// the pointer-style policy in `python/sell_me_a_sasquatch/policy.py`)
    /// instead of having to memorize the enumeration order the engine
    /// happens to use.
    ///
    /// Privacy is preserved exactly as in [`GameState::legal_actions`]: a
    /// targeted card contributes its class only when it is already face
    /// up. Still-hidden targets contribute to a count, never to a class.
    pub fn encode_actions(&self, player: PlayerId, actions: &[Action], out: &mut [f32]) {
        assert!(
            out.len().is_multiple_of(ACTION_FEAT_LEN),
            "action buffer must be a whole number of rows"
        );
        out.fill(0.0);
        let rows = out.len() / ACTION_FEAT_LEN;
        let n = self.num_players;
        let victim = match &self.phase {
            Phase::NastyResolution(s) => Some(s.pending.player),
            _ => None,
        };

        for (i, action) in actions.iter().take(rows).enumerate() {
            let row = &mut out[i * ACTION_FEAT_LEN..(i + 1) * ACTION_FEAT_LEN];
            let mut target_seat = player;
            match action {
                Action::SubmitDeal { cards } => {
                    row[AF_TYPE] = 1.0;
                    for &c in cards {
                        row[AF_COMMITTED + self.class_of(c)] += 1.0 / 3.0;
                    }
                }
                Action::TwoPlayerSubmitDeal {
                    own_pile,
                    other_pile,
                } => {
                    row[AF_TYPE + 1] = 1.0;
                    for &c in own_pile {
                        row[AF_COMMITTED + self.class_of(c)] += 1.0 / 3.0;
                    }
                    // the opposite pile is the offer to the other seat's
                    // own cards, but the decision that matters is which
                    // classes land on which side
                    for &c in other_pile {
                        row[AF_TARGET_CARDS + self.class_of(c)] += 1.0 / 3.0;
                    }
                    target_seat = 1 - player;
                }
                Action::RevealCard { card } => {
                    row[AF_TYPE + 2] = 1.0;
                    row[AF_COMMITTED + self.class_of(*card)] += 1.0;
                }
                Action::BuyerPeek { target_seller } => {
                    row[AF_TYPE + 3] = 1.0;
                    target_seat = *target_seller;
                    row[AF_HIDDEN_TARGETS] = self
                        .deal_for_seller(*target_seller)
                        .map(|d| d.cards.iter().filter(|c| !c.revealed).count() as f32 / 3.0)
                        .unwrap_or(0.0);
                }
                Action::PlayThingamabob { card, params } => {
                    row[AF_TYPE + 4] = 1.0;
                    row[AF_COMMITTED + self.class_of(*card)] += 1.0;
                    match params {
                        ThingamabobParams::PlatonicIsolator { target_player } => {
                            target_seat = *target_player;
                        }
                        ThingamabobParams::RemoveFromDeals { removals } => {
                            let mut sellers: Vec<PlayerId> =
                                removals.iter().map(|(s, _)| *s).collect();
                            sellers.sort_unstable();
                            sellers.dedup();
                            row[AF_DEALS_TOUCHED] = sellers.len() as f32 / 2.0;
                            if let Some(&first) = sellers.first() {
                                target_seat = first;
                            }
                            for (seller, card) in removals {
                                if self.deal_card_is_revealed(*seller, *card) {
                                    row[AF_TARGET_CARDS + self.class_of(*card)] += 1.0 / 2.0;
                                } else {
                                    row[AF_HIDDEN_TARGETS] += 1.0 / 2.0;
                                }
                            }
                        }
                        ThingamabobParams::CryptozooticExpander {
                            hand_card,
                            target_deal,
                        } => {
                            row[AF_COMMITTED + self.class_of(*hand_card)] += 1.0;
                            target_seat = *target_deal;
                        }
                        ThingamabobParams::SpectroelectricOptimeter { target_deal, .. } => {
                            target_seat = *target_deal;
                            row[AF_HIDDEN_TARGETS] = 1.0 / 3.0;
                        }
                    }
                }
                Action::PassThingamabobWindow => {
                    row[AF_TYPE + 5] = 1.0;
                }
                Action::ChooseDeal { seller } => {
                    row[AF_TYPE + 6] = 1.0;
                    target_seat = *seller;
                    if let Some(deal) = self.deal_for_seller(*seller) {
                        let mut hidden = 0f32;
                        for c in &deal.cards {
                            if c.revealed {
                                row[AF_TARGET_CARDS + self.class_of(c.card)] += 1.0 / 3.0;
                            } else {
                                hidden += 1.0;
                            }
                        }
                        row[AF_HIDDEN_TARGETS] = hidden / 3.0;
                    }
                }
                Action::RespondToDeal { reverse } => {
                    row[AF_TYPE + 7] = 1.0;
                    row[AF_REVERSE] = f32::from(*reverse);
                    target_seat = 1 - player;
                }
                Action::ResolveNastyPenalty { taken_cards } => {
                    row[AF_TYPE + 8] = 1.0;
                    // collections are public, so every steal target is known
                    for &c in taken_cards {
                        row[AF_TARGET_CARDS + self.class_of(c)] += 1.0 / 2.0;
                    }
                    if let Some(v) = victim {
                        target_seat = v;
                    }
                }
            }

            one_hot(
                &mut row[AF_TARGET_SEAT..AF_TARGET_SEAT + MAX_PLAYERS],
                (target_seat + n - player) % n,
            );
            row[AF_TARGET_IS_SELF] = f32::from(target_seat == player);
        }
    }

    /// Whether the given card in the given seller's deal is face up.
    fn deal_card_is_revealed(&self, seller: PlayerId, card: CardId) -> bool {
        self.deal_for_seller(seller)
            .is_some_and(|d| d.cards.iter().any(|c| c.card == card && c.revealed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::DeckConfig;

    const DECK: &str = include_str!("../../configs/deck.toml");

    fn game(num_players: usize, seed: u64) -> GameState {
        GameState::new(num_players, DeckConfig::from_toml_str(DECK).unwrap(), seed).unwrap()
    }

    #[test]
    fn encoding_is_finite_bounded_and_fixed_width_for_every_table_size() {
        for n in MIN_PLAYERS..=MAX_PLAYERS {
            let mut g = game(n, n as u64);
            let mut buf = vec![0f32; OBS_LEN];
            let mut counter = 1u64;
            for _ in 0..400 {
                if g.is_game_over() {
                    break;
                }
                for p in 0..n {
                    g.encode_observation(p, &mut buf);
                    assert!(
                        buf.iter().all(|v| v.is_finite()),
                        "non-finite feature for {n} players"
                    );
                    assert!(
                        buf.iter().all(|v| (-4.0..=4.0).contains(v)),
                        "unnormalized feature for {n} players"
                    );
                }
                let p = g.active_player().unwrap();
                let actions = g.legal_actions(p);
                counter = counter.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = actions[(counter as usize) % actions.len()].clone();
                g.apply_action(p, a).unwrap();
            }
        }
    }

    #[test]
    fn unseen_counts_never_include_what_the_observer_can_see() {
        let g = game(4, 7);
        let mut buf = vec![0f32; OBS_LEN];
        g.encode_observation(0, &mut buf);
        // On turn 1 every card is either in a hand or in the draw pile, so
        // the observer accounts for exactly their own 5 hand cards.
        let unseen_total = buf[NOISE_OFFSET - NUM_EVENT_KINDS - 1];
        let deck_total: f32 = g.deck_counts().iter().sum::<u32>() as f32;
        assert!((unseen_total - (deck_total - 5.0) / deck_total).abs() < 1e-6);
    }

    #[test]
    fn ego_centric_encoding_puts_the_observer_in_slot_zero() {
        let g = game(5, 11);
        let mut a = vec![0f32; OBS_LEN];
        let mut b = vec![0f32; OBS_LEN];
        g.encode_observation(0, &mut a);
        g.encode_observation(3, &mut b);
        let players_at = GLOBAL_LEN + HAND_LEN;
        // Slot 0 of each view is that observer's own seat, so both views
        // agree it is valid, and the encodings differ elsewhere.
        assert_eq!(a[players_at], 1.0);
        assert_eq!(b[players_at], 1.0);
        assert_ne!(a, b);
    }

    #[test]
    fn action_rows_describe_each_legal_action_and_hide_face_down_cards() {
        let mut g = game(4, 21);
        let mut counter = 5u64;
        for _ in 0..120 {
            if g.is_game_over() {
                break;
            }
            let p = g.active_player().unwrap();
            let actions = g.legal_actions(p);
            let mut buf = vec![0f32; actions.len().max(1) * ACTION_FEAT_LEN];
            g.encode_actions(p, &actions, &mut buf);
            for (i, _) in actions.iter().enumerate() {
                let row = &buf[i * ACTION_FEAT_LEN..(i + 1) * ACTION_FEAT_LEN];
                let type_hot: f32 = row[AF_TYPE..AF_TYPE + AF_NUM_TYPES].iter().sum();
                assert_eq!(
                    type_hot, 1.0,
                    "every action row names exactly one action type"
                );
                let seat_hot: f32 = row[AF_TARGET_SEAT..AF_TARGET_SEAT + MAX_PLAYERS]
                    .iter()
                    .sum();
                assert_eq!(
                    seat_hot, 1.0,
                    "every action row names exactly one target seat"
                );
                assert!(row
                    .iter()
                    .all(|v| v.is_finite() && (-4.0..=4.0).contains(v)));
            }
            counter = counter.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = actions[(counter as usize) % actions.len()].clone();
            g.apply_action(p, a).unwrap();
        }
    }

    /// Peeking is the one move whose whole point is that the target is
    /// unknown - so its row must carry a hidden-card *count* and no classes.
    #[test]
    fn buyer_peek_rows_never_leak_the_hidden_card_class() {
        let mut g = game(4, 33);
        while g.current_phase() != "buyer_peek" {
            let p = g.active_player().unwrap();
            let actions = g.legal_actions(p);
            g.apply_action(p, actions[0].clone()).unwrap();
        }
        let p = g.active_player().unwrap();
        let actions = g.legal_actions(p);
        let mut buf = vec![0f32; actions.len() * ACTION_FEAT_LEN];
        g.encode_actions(p, &actions, &mut buf);
        for i in 0..actions.len() {
            let row = &buf[i * ACTION_FEAT_LEN..(i + 1) * ACTION_FEAT_LEN];
            assert_eq!(
                row[AF_TARGET_CARDS..AF_TARGET_CARDS + NUM_CARD_CLASSES]
                    .iter()
                    .sum::<f32>(),
                0.0
            );
            assert!(row[AF_HIDDEN_TARGETS] > 0.0);
        }
    }

    #[test]
    fn noise_slots_are_left_for_the_caller() {
        let g = game(3, 3);
        let mut buf = vec![9f32; OBS_LEN];
        g.encode_observation(1, &mut buf);
        assert!(buf[NOISE_OFFSET..].iter().all(|&v| v == 0.0));
    }
}
