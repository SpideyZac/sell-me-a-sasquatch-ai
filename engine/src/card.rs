//! Card identity and type catalog. See PROMPT.md §2.4/§2.5.
//!
//! Individual named Creature cards carry no mechanical behavior (§2.4 note) -
//! only `Tier` matters for legality/set-completion. Names are display-only
//! flavor. Nasties and Thingamabobs *do* have per-kind identity because sets
//! (Nasties) and effects (both) are defined per specific card, not per tier.

use std::fmt;

pub type CardId = u32;
pub type PlayerId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    Giant,
    Big,
    Medium,
    Tiny,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Giant, Tier::Big, Tier::Medium, Tier::Tiny];

    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "Giant" => Some(Tier::Giant),
            "Big" => Some(Tier::Big),
            "Medium" => Some(Tier::Medium),
            "Tiny" => Some(Tier::Tiny),
            _ => None,
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Tier::Giant => "Giant",
            Tier::Big => "Big",
            Tier::Medium => "Medium",
            Tier::Tiny => "Tiny",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NastyKind {
    PoisonPillBug,
    LoanShark,
    TrojanHorse,
}

impl NastyKind {
    pub const ALL: [NastyKind; 3] = [NastyKind::PoisonPillBug, NastyKind::LoanShark, NastyKind::TrojanHorse];

    pub fn parse(s: &str) -> Option<NastyKind> {
        match s {
            "Poison Pill Bug" => Some(NastyKind::PoisonPillBug),
            "Loan Shark" => Some(NastyKind::LoanShark),
            "Trojan Horse" => Some(NastyKind::TrojanHorse),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NastyKind::PoisonPillBug => "Poison Pill Bug",
            NastyKind::LoanShark => "Loan Shark",
            NastyKind::TrojanHorse => "Trojan Horse",
        }
    }
}

/// Behavior triggered when a Nasty set is traded in (§2.4). Parsed from the
/// deck config's `effect` string so behavior stays data-driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NastyEffect {
    /// Buyer may steal up to N cards (0..=N, buyer's choice) from the
    /// trading-in player's Collection. Fizzles/caps silently if fewer are
    /// available (§2.4 global rule).
    BuyerMayStealUpToNCards(u8),
    /// Buyer automatically steals 1 Point Token (no choice). Fizzles
    /// silently if the player has 0 tokens.
    BuyerStealsOnePointToken,
}

impl NastyEffect {
    pub fn parse(s: &str) -> Option<NastyEffect> {
        match s {
            "buyer_may_steal_up_to_1_card" => Some(NastyEffect::BuyerMayStealUpToNCards(1)),
            "buyer_may_steal_up_to_2_cards" => Some(NastyEffect::BuyerMayStealUpToNCards(2)),
            "buyer_steals_1_point_token" => Some(NastyEffect::BuyerStealsOnePointToken),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ThingamabobKind {
    PlatonicIsolator,
    DetritalRepositioner,
    SuperDetritalRepositioner,
    CryptozooticExpander,
    SpectroelectricOptimeter,
}

impl ThingamabobKind {
    pub const ALL: [ThingamabobKind; 5] = [
        ThingamabobKind::PlatonicIsolator,
        ThingamabobKind::DetritalRepositioner,
        ThingamabobKind::SuperDetritalRepositioner,
        ThingamabobKind::CryptozooticExpander,
        ThingamabobKind::SpectroelectricOptimeter,
    ];

    pub fn parse(s: &str) -> Option<ThingamabobKind> {
        match s {
            "Platonic Isolator" => Some(ThingamabobKind::PlatonicIsolator),
            "Detrital Repositioner" => Some(ThingamabobKind::DetritalRepositioner),
            "Super Detrital Repositioner" => Some(ThingamabobKind::SuperDetritalRepositioner),
            "Cryptozooptic Expander" => Some(ThingamabobKind::CryptozooticExpander),
            "Spectroelectric Optimeter" => Some(ThingamabobKind::SpectroelectricOptimeter),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ThingamabobKind::PlatonicIsolator => "Platonic Isolator",
            ThingamabobKind::DetritalRepositioner => "Detrital Repositioner",
            ThingamabobKind::SuperDetritalRepositioner => "Super Detrital Repositioner",
            ThingamabobKind::CryptozooticExpander => "Cryptozooptic Expander",
            ThingamabobKind::SpectroelectricOptimeter => "Spectroelectric Optimeter",
        }
    }
}

/// Behavior triggered when a Thingamabob is played (§2.3 step 4 / §2.4
/// table). Parsed from the deck config's `effect` string. Note that the two
/// "remove cards from deals" cards share one parameterized effect variant -
/// they are still distinct cards/counts in the catalog, just unified logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThingamabobEffect {
    StealPointTokenFromRicherPlayer,
    RemoveCardsFromDeals { max_cards: u8, max_deals: u8 },
    AddHiddenHandCardToDeal,
    RevealCardInDeal,
}

impl ThingamabobEffect {
    pub fn parse(s: &str) -> Option<ThingamabobEffect> {
        match s {
            "steal_point_token_from_richer_player" => Some(ThingamabobEffect::StealPointTokenFromRicherPlayer),
            "remove_1_card_from_1_deal" => Some(ThingamabobEffect::RemoveCardsFromDeals { max_cards: 1, max_deals: 1 }),
            "remove_up_to_2_cards_from_1_or_2_deals" => {
                Some(ThingamabobEffect::RemoveCardsFromDeals { max_cards: 2, max_deals: 2 })
            }
            "add_hidden_hand_card_to_deal" => Some(ThingamabobEffect::AddHiddenHandCardToDeal),
            "reveal_1_card_in_a_deal" => Some(ThingamabobEffect::RevealCardInDeal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardKind {
    Creature(Tier),
    Nasty(NastyKind),
    Thingamabob(ThingamabobKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Card {
    pub id: CardId,
    pub kind: CardKind,
    /// Display-only flavor name. Never affects legality or set-completion
    /// logic (§2.4 note on Creatures) - for Nasties/Thingamabobs this is
    /// just the canonical card name.
    pub name: String,
}

/// Number of distinct *card classes* - the coarse bucket a policy actually
/// reasons about (tier for Creatures, kind for Nasties/Thingamabobs).
/// Individual card ids are meaningless to a learner (they're an arbitrary
/// per-episode shuffle artifact), so every observation feature is expressed
/// as counts over these classes.
pub const NUM_CARD_CLASSES: usize = 12;

/// Stable class ordering. Index here == index everywhere else (observation
/// encoder, Python `spaces.CARD_CLASS_NAMES`, deck-composition vectors).
pub const CARD_CLASS_NAMES: [&str; NUM_CARD_CLASSES] = [
    "Creature:Giant",
    "Creature:Big",
    "Creature:Medium",
    "Creature:Tiny",
    "Nasty:Poison Pill Bug",
    "Nasty:Loan Shark",
    "Nasty:Trojan Horse",
    "Thingamabob:Platonic Isolator",
    "Thingamabob:Detrital Repositioner",
    "Thingamabob:Super Detrital Repositioner",
    "Thingamabob:Cryptozooptic Expander",
    "Thingamabob:Spectroelectric Optimeter",
];

impl CardKind {
    /// This kind's index into a `NUM_CARD_CLASSES`-wide count vector.
    pub const fn class_index(self) -> usize {
        match self {
            CardKind::Creature(Tier::Giant) => 0,
            CardKind::Creature(Tier::Big) => 1,
            CardKind::Creature(Tier::Medium) => 2,
            CardKind::Creature(Tier::Tiny) => 3,
            CardKind::Nasty(NastyKind::PoisonPillBug) => 4,
            CardKind::Nasty(NastyKind::LoanShark) => 5,
            CardKind::Nasty(NastyKind::TrojanHorse) => 6,
            CardKind::Thingamabob(ThingamabobKind::PlatonicIsolator) => 7,
            CardKind::Thingamabob(ThingamabobKind::DetritalRepositioner) => 8,
            CardKind::Thingamabob(ThingamabobKind::SuperDetritalRepositioner) => 9,
            CardKind::Thingamabob(ThingamabobKind::CryptozooticExpander) => 10,
            CardKind::Thingamabob(ThingamabobKind::SpectroelectricOptimeter) => 11,
        }
    }

    pub fn name(self) -> String {
        match self {
            CardKind::Creature(tier) => format!("Creature:{tier}"),
            CardKind::Nasty(kind) => format!("Nasty:{}", kind.name()),
            CardKind::Thingamabob(kind) => format!("Thingamabob:{}", kind.name()),
        }
    }

    /// Inverse of `name` - parses e.g. `"Creature:Tiny"` back into a kind.
    pub fn parse(s: &str) -> Option<CardKind> {
        let (prefix, rest) = s.split_once(':')?;
        match prefix {
            "Creature" => Tier::parse(rest).map(CardKind::Creature),
            "Nasty" => NastyKind::parse(rest).map(CardKind::Nasty),
            "Thingamabob" => ThingamabobKind::parse(rest).map(CardKind::Thingamabob),
            _ => None,
        }
    }
}

#[cfg(test)]
mod class_tests {
    use super::*;

    #[test]
    fn class_index_matches_name_table_and_round_trips() {
        let mut seen = [false; NUM_CARD_CLASSES];
        let all: Vec<CardKind> = Tier::ALL
            .iter()
            .map(|&t| CardKind::Creature(t))
            .chain(NastyKind::ALL.iter().map(|&k| CardKind::Nasty(k)))
            .chain(ThingamabobKind::ALL.iter().map(|&k| CardKind::Thingamabob(k)))
            .collect();
        assert_eq!(all.len(), NUM_CARD_CLASSES);
        for kind in all {
            let i = kind.class_index();
            assert!(!seen[i], "duplicate class index {i}");
            seen[i] = true;
            assert_eq!(CARD_CLASS_NAMES[i], kind.name());
            assert_eq!(CardKind::parse(&kind.name()), Some(kind));
        }
    }
}
