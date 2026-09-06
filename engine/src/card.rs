//! Card identity and the type catalog.
//!
//! Individual named creature cards carry no mechanical behavior, only
//! [`Tier`] matters for legality and set completion; names are display-only
//! flavor. Nasties and thingamabobs do have per-kind identity because sets
//! (nasties) and effects (both) are defined per specific card, not per tier.

use std::fmt;

/// Unique id for a single physical card in the deck.
pub type CardId = u32;
/// Index of a player within the game's player list.
pub type PlayerId = usize;

/// Size class of a creature card, the only thing about a creature that
/// affects legality or set completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Largest creature tier.
    Giant,
    /// Second largest creature tier.
    Big,
    /// Second smallest creature tier.
    Medium,
    /// Smallest creature tier.
    Tiny,
}

impl Tier {
    /// Every tier, in display order.
    pub const ALL: [Tier; 4] = [Tier::Giant, Tier::Big, Tier::Medium, Tier::Tiny];

    /// Parses a tier from its config name, returns `None` if unrecognized.
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

/// Which nasty card a nasty is, since nasty behavior is defined per card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NastyKind {
    /// Poison Pill Bug.
    PoisonPillBug,
    /// Loan Shark.
    LoanShark,
    /// Trojan Horse.
    TrojanHorse,
}

impl NastyKind {
    /// Every nasty kind.
    pub const ALL: [NastyKind; 3] = [
        NastyKind::PoisonPillBug,
        NastyKind::LoanShark,
        NastyKind::TrojanHorse,
    ];

    /// Parses a nasty kind from its config name, returns `None` if unrecognized.
    pub fn parse(s: &str) -> Option<NastyKind> {
        match s {
            "Poison Pill Bug" => Some(NastyKind::PoisonPillBug),
            "Loan Shark" => Some(NastyKind::LoanShark),
            "Trojan Horse" => Some(NastyKind::TrojanHorse),
            _ => None,
        }
    }

    /// The card's display name.
    pub fn name(&self) -> &'static str {
        match self {
            NastyKind::PoisonPillBug => "Poison Pill Bug",
            NastyKind::LoanShark => "Loan Shark",
            NastyKind::TrojanHorse => "Trojan Horse",
        }
    }
}

/// What happens when a full nasty set is traded in, parsed from the deck
/// config's `effect` string so behavior stays data-driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NastyEffect {
    /// Buyer may steal up to N cards, their choice, from the trading-in
    /// player's collection. Fizzles or caps silently if fewer are available.
    BuyerMayStealUpToNCards(u8),
    /// Buyer automatically steals one point token, no choice involved.
    /// Fizzles silently if the player has none.
    BuyerStealsOnePointToken,
}

impl NastyEffect {
    /// Parses a nasty effect from its config string, returns `None` if unrecognized.
    pub fn parse(s: &str) -> Option<NastyEffect> {
        match s {
            "buyer_may_steal_up_to_1_card" => Some(NastyEffect::BuyerMayStealUpToNCards(1)),
            "buyer_may_steal_up_to_2_cards" => Some(NastyEffect::BuyerMayStealUpToNCards(2)),
            "buyer_steals_1_point_token" => Some(NastyEffect::BuyerStealsOnePointToken),
            _ => None,
        }
    }
}

/// Which thingamabob card a thingamabob is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ThingamabobKind {
    /// Platonic Isolator.
    PlatonicIsolator,
    /// Detrital Repositioner.
    DetritalRepositioner,
    /// Super Detrital Repositioner.
    SuperDetritalRepositioner,
    /// Cryptozootic Expander.
    CryptozooticExpander,
    /// Spectroelectric Optimeter.
    SpectroelectricOptimeter,
}

impl ThingamabobKind {
    /// Every thingamabob kind.
    pub const ALL: [ThingamabobKind; 5] = [
        ThingamabobKind::PlatonicIsolator,
        ThingamabobKind::DetritalRepositioner,
        ThingamabobKind::SuperDetritalRepositioner,
        ThingamabobKind::CryptozooticExpander,
        ThingamabobKind::SpectroelectricOptimeter,
    ];

    /// Parses a thingamabob kind from its config name, returns `None` if unrecognized.
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

    /// The card's display name.
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

/// What happens when a thingamabob is played, parsed from the deck config's
/// `effect` string. The two "remove cards from deals" cards share one
/// parameterized variant here even though they're distinct cards in the
/// catalog, since the logic is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThingamabobEffect {
    /// Steal a point token from a richer player.
    StealPointTokenFromRicherPlayer,
    /// Remove cards from active deals.
    RemoveCardsFromDeals {
        /// Max number of cards that can be removed total.
        max_cards: u8,
        /// Max number of distinct deals that can be touched.
        max_deals: u8,
    },
    /// Add a hidden card from hand into a deal.
    AddHiddenHandCardToDeal,
    /// Reveal a hidden card in a deal.
    RevealCardInDeal,
}

impl ThingamabobEffect {
    /// Parses a thingamabob effect from its config string, returns `None` if unrecognized.
    pub fn parse(s: &str) -> Option<ThingamabobEffect> {
        match s {
            "steal_point_token_from_richer_player" => {
                Some(ThingamabobEffect::StealPointTokenFromRicherPlayer)
            }
            "remove_1_card_from_1_deal" => Some(ThingamabobEffect::RemoveCardsFromDeals {
                max_cards: 1,
                max_deals: 1,
            }),
            "remove_up_to_2_cards_from_1_or_2_deals" => {
                Some(ThingamabobEffect::RemoveCardsFromDeals {
                    max_cards: 2,
                    max_deals: 2,
                })
            }
            "add_hidden_hand_card_to_deal" => Some(ThingamabobEffect::AddHiddenHandCardToDeal),
            "reveal_1_card_in_a_deal" => Some(ThingamabobEffect::RevealCardInDeal),
            _ => None,
        }
    }
}

/// The mechanical type of a card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardKind {
    /// A creature of the given tier.
    Creature(Tier),
    /// A nasty of the given kind.
    Nasty(NastyKind),
    /// A thingamabob of the given kind.
    Thingamabob(ThingamabobKind),
}

/// A single physical card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Card {
    /// This card's unique id.
    pub id: CardId,
    /// This card's mechanical type.
    pub kind: CardKind,
    /// Display-only flavor name, never affects legality or set completion.
    /// For nasties and thingamabobs this is just the card's canonical name.
    pub name: String,
}

/// Number of distinct card classes, the coarse bucket a policy actually
/// reasons about (tier for creatures, kind for nasties and thingamabobs).
/// Individual card ids are meaningless to a learner since they're an
/// arbitrary per-episode shuffle artifact, so every observation feature is
/// expressed as counts over these classes.
pub const NUM_CARD_CLASSES: usize = 12;

/// Stable class ordering, index here matches index everywhere else
/// (observation encoder, Python's `spaces.CARD_CLASS_NAMES`, deck-composition
/// vectors).
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
    /// This kind's index into a [`NUM_CARD_CLASSES`]-wide count vector.
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

    /// This kind's class name, e.g. `"Creature:Tiny"`.
    pub fn name(self) -> String {
        match self {
            CardKind::Creature(tier) => format!("Creature:{tier}"),
            CardKind::Nasty(kind) => format!("Nasty:{}", kind.name()),
            CardKind::Thingamabob(kind) => format!("Thingamabob:{}", kind.name()),
        }
    }

    /// Inverse of [`Self::name`]; parses e.g. `"Creature:Tiny"` back into a kind.
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
            .chain(
                ThingamabobKind::ALL
                    .iter()
                    .map(|&k| CardKind::Thingamabob(k)),
            )
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
