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
