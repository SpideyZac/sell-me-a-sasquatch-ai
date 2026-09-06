//! Deck config loading (§2.5) - parses `deck.toml` into a `Catalog` (rules
//! data: set sizes, effects) and a `Vec<Card>` (concrete card instances).

use crate::card::{Card, CardKind, NastyEffect, NastyKind, Tier, ThingamabobEffect, ThingamabobKind};
use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Deserialize)]
struct RawCreatureTier {
    tier: String,
    set_size: u32,
    copies: u32,
}

#[derive(Debug, Deserialize)]
struct RawNasty {
    name: String,
    set_size: u32,
    copies: u32,
    effect: String,
}

#[derive(Debug, Deserialize)]
struct RawThingamabob {
    name: String,
    copies: u32,
    effect: String,
}

#[derive(Debug, Deserialize)]
struct RawDeckConfig {
    creature_tiers: Vec<RawCreatureTier>,
    nasties: Vec<RawNasty>,
    thingamabobs: Vec<RawThingamabob>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeckConfigError {
    #[error("failed to parse deck config toml: {0}")]
    Toml(String),
    #[error("unknown creature tier name: {0}")]
    UnknownTier(String),
    #[error("unknown nasty name: {0}")]
    UnknownNasty(String),
    #[error("unknown nasty effect: {0}")]
    UnknownNastyEffect(String),
    #[error("unknown thingamabob name: {0}")]
    UnknownThingamabob(String),
    #[error("unknown thingamabob effect: {0}")]
    UnknownThingamabobEffect(String),
    #[error("duplicate creature tier entry: {0}")]
    DuplicateTier(String),
    #[error("duplicate nasty entry: {0}")]
    DuplicateNasty(String),
    #[error("duplicate thingamabob entry: {0}")]
    DuplicateThingamabob(String),
}

/// Rules data derived from the deck config: set sizes and effect bindings.
/// This is what game logic consults - it never hardcodes set sizes.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub creature_set_size: BTreeMap<Tier, u32>,
    pub nasty_set_size: BTreeMap<NastyKind, u32>,
    pub nasty_effect: BTreeMap<NastyKind, NastyEffect>,
    pub thingamabob_effect: BTreeMap<ThingamabobKind, ThingamabobEffect>,
}

impl Catalog {
    pub fn creature_set_size(&self, tier: Tier) -> u32 {
        self.creature_set_size[&tier]
    }

    pub fn nasty_set_size(&self, kind: NastyKind) -> u32 {
        self.nasty_set_size[&kind]
    }

    pub fn nasty_effect(&self, kind: NastyKind) -> NastyEffect {
        self.nasty_effect[&kind]
    }

    pub fn thingamabob_effect(&self, kind: ThingamabobKind) -> ThingamabobEffect {
        self.thingamabob_effect[&kind]
    }
}

/// A fully parsed deck config: rules catalog plus the concrete cards to deal.
#[derive(Debug, Clone)]
pub struct DeckConfig {
    pub catalog: Catalog,
    pub cards: Vec<Card>,
}

impl DeckConfig {
    pub fn total_cards(&self) -> usize {
        self.cards.len()
    }

    pub fn from_toml_str(s: &str) -> Result<DeckConfig, DeckConfigError> {
        let raw: RawDeckConfig = toml::from_str(s).map_err(|e| DeckConfigError::Toml(e.to_string()))?;
        Self::from_raw(raw)
    }

    pub fn from_file(path: &std::path::Path) -> Result<DeckConfig, DeckConfigError> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| DeckConfigError::Toml(format!("could not read {}: {e}", path.display())))?;
        Self::from_toml_str(&s)
    }

    fn from_raw(raw: RawDeckConfig) -> Result<DeckConfig, DeckConfigError> {
        let mut creature_set_size = BTreeMap::new();
        let mut cards = Vec::new();
        let mut next_id: u32 = 0;

        for entry in &raw.creature_tiers {
            let tier = Tier::parse(&entry.tier).ok_or_else(|| DeckConfigError::UnknownTier(entry.tier.clone()))?;
            if creature_set_size.insert(tier, entry.set_size).is_some() {
                return Err(DeckConfigError::DuplicateTier(entry.tier.clone()));
            }
            for i in 0..entry.copies {
                cards.push(Card {
                    id: next_id,
                    kind: CardKind::Creature(tier),
                    name: creature_flavor_name(tier, i),
                });
                next_id += 1;
            }
        }

        let mut nasty_set_size = BTreeMap::new();
        let mut nasty_effect = BTreeMap::new();
        for entry in &raw.nasties {
            let kind = NastyKind::parse(&entry.name).ok_or_else(|| DeckConfigError::UnknownNasty(entry.name.clone()))?;
            let effect =
                NastyEffect::parse(&entry.effect).ok_or_else(|| DeckConfigError::UnknownNastyEffect(entry.effect.clone()))?;
            if nasty_set_size.insert(kind, entry.set_size).is_some() {
                return Err(DeckConfigError::DuplicateNasty(entry.name.clone()));
            }
            nasty_effect.insert(kind, effect);
            for _ in 0..entry.copies {
                cards.push(Card { id: next_id, kind: CardKind::Nasty(kind), name: kind.name().to_string() });
                next_id += 1;
            }
        }

        let mut thingamabob_effect = BTreeMap::new();
        for entry in &raw.thingamabobs {
            let kind = ThingamabobKind::parse(&entry.name)
                .ok_or_else(|| DeckConfigError::UnknownThingamabob(entry.name.clone()))?;
            let effect = ThingamabobEffect::parse(&entry.effect)
                .ok_or_else(|| DeckConfigError::UnknownThingamabobEffect(entry.effect.clone()))?;
            if thingamabob_effect.insert(kind, effect).is_some() {
                return Err(DeckConfigError::DuplicateThingamabob(entry.name.clone()));
            }
            for _ in 0..entry.copies {
                cards.push(Card { id: next_id, kind: CardKind::Thingamabob(kind), name: kind.name().to_string() });
                next_id += 1;
            }
        }

        let catalog = Catalog { creature_set_size, nasty_set_size, nasty_effect, thingamabob_effect };
        Ok(DeckConfig { catalog, cards })
    }
}

/// Display-only flavor names, cycled arbitrarily per tier (§2.4 note - never
/// affects legality/set-completion logic).
fn creature_flavor_name(tier: Tier, index: u32) -> String {
    let names: &[&str] = match tier {
        Tier::Giant => &["Mega Mammoth", "Colossal Kraken", "Titan Sasquatch", "Great Griffin"],
        Tier::Big => &["Bullbear", "Direwolf", "Thunder Yeti", "Rock Troll", "Cave Bear"],
        Tier::Medium => &["Jackalope", "Chupacabra", "Wendigo", "Moth Man", "Skunk Ape"],
        Tier::Tiny => &["Gremlin", "Pixie Newt", "Bog Sprite", "Fen Imp", "Marsh Gnome"],
    };
    names[index as usize % names.len()].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIRMED_DECK_TOML: &str = include_str!("../../configs/deck.toml");

    #[test]
    fn confirmed_deck_totals_exactly_120() {
        let deck = DeckConfig::from_toml_str(CONFIRMED_DECK_TOML).unwrap();
        assert_eq!(deck.total_cards(), 120);
    }

    /// The confirmed real deck (§2.5) is fixed physical-game data and is
    /// *not* required to have set sizes divide evenly into copy counts (e.g.
    /// 25 Medium creatures / set_size 3 leaves a remainder - that's fine,
    /// not every card need be part of a completed set). The "set_size
    /// divides evenly into a plausible deck" sanity check instead applies to
    /// a smaller synthetic config, of the kind used for fast unit tests.
    const PLAUSIBLE_TEST_DECK_TOML: &str = r#"
        [[creature_tiers]]
        tier = "Giant"
        set_size = 1
        copies = 4

        [[creature_tiers]]
        tier = "Big"
        set_size = 2
        copies = 16

        [[creature_tiers]]
        tier = "Medium"
        set_size = 3
        copies = 24

        [[creature_tiers]]
        tier = "Tiny"
        set_size = 4
        copies = 20

        [[nasties]]
        name = "Poison Pill Bug"
        set_size = 2
        copies = 10
        effect = "buyer_may_steal_up_to_1_card"

        [[nasties]]
        name = "Loan Shark"
        set_size = 2
        copies = 10
        effect = "buyer_may_steal_up_to_2_cards"

        [[nasties]]
        name = "Trojan Horse"
        set_size = 3
        copies = 12
        effect = "buyer_steals_1_point_token"

        [[thingamabobs]]
        name = "Platonic Isolator"
        copies = 3
        effect = "steal_point_token_from_richer_player"

        [[thingamabobs]]
        name = "Detrital Repositioner"
        copies = 6
        effect = "remove_1_card_from_1_deal"

        [[thingamabobs]]
        name = "Super Detrital Repositioner"
        copies = 3
        effect = "remove_up_to_2_cards_from_1_or_2_deals"

        [[thingamabobs]]
        name = "Cryptozooptic Expander"
        copies = 6
        effect = "add_hidden_hand_card_to_deal"

        [[thingamabobs]]
        name = "Spectroelectric Optimeter"
        copies = 5
        effect = "reveal_1_card_in_a_deal"
    "#;

    #[test]
    fn plausible_test_deck_set_sizes_divide_evenly() {
        let deck = DeckConfig::from_toml_str(PLAUSIBLE_TEST_DECK_TOML).unwrap();
        for tier in Tier::ALL {
            let copies = deck.cards.iter().filter(|c| c.kind == CardKind::Creature(tier)).count() as u32;
            let set_size = deck.catalog.creature_set_size(tier);
            assert!(set_size > 0);
            assert!(copies.is_multiple_of(set_size), "{tier} copies {copies} not divisible by set_size {set_size}");
        }
        for kind in NastyKind::ALL {
            let copies = deck.cards.iter().filter(|c| c.kind == CardKind::Nasty(kind)).count() as u32;
            let set_size = deck.catalog.nasty_set_size(kind);
            assert!(set_size > 0);
            assert!(copies.is_multiple_of(set_size), "{} copies {copies} not divisible by set_size {set_size}", kind.name());
        }
    }

    #[test]
    fn confirmed_deck_subtotals_match_spec() {
        let deck = DeckConfig::from_toml_str(CONFIRMED_DECK_TOML).unwrap();
        let creature_count = deck.cards.iter().filter(|c| matches!(c.kind, CardKind::Creature(_))).count();
        let nasty_count = deck.cards.iter().filter(|c| matches!(c.kind, CardKind::Nasty(_))).count();
        let thingamabob_count = deck.cards.iter().filter(|c| matches!(c.kind, CardKind::Thingamabob(_))).count();
        assert_eq!(creature_count, 65);
        assert_eq!(nasty_count, 32);
        assert_eq!(thingamabob_count, 23);
    }

    #[test]
    fn rejects_unknown_names() {
        let bad = r#"
            creature_tiers = []
            thingamabobs = []

            [[nasties]]
            name = "Not A Real Nasty"
            set_size = 2
            copies = 1
            effect = "buyer_may_steal_up_to_1_card"
        "#;
        let err = DeckConfig::from_toml_str(bad).unwrap_err();
        assert_eq!(err, DeckConfigError::UnknownNasty("Not A Real Nasty".to_string()));
    }
}
