//! PyO3 bindings (§3.3) exposing `sasquatch_engine::game::GameState` as
//! `PyGame`. Kept to the FFI boundary crossed once per micro-step (§3.5) -
//! no per-card calls back into Rust.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sasquatch_engine::action::{Action, ThingamabobParams};
use sasquatch_engine::card::{CardId, CardKind, NastyKind, PlayerId, Tier, ThingamabobKind};
use sasquatch_engine::deck::DeckConfig;
use sasquatch_engine::game::{Event, GameState, Observation, ObservedDeal};

fn card_kind_to_string(kind: CardKind) -> String {
    match kind {
        CardKind::Creature(tier) => format!("Creature:{tier}"),
        CardKind::Nasty(kind) => format!("Nasty:{}", kind.name()),
        CardKind::Thingamabob(kind) => format!("Thingamabob:{}", kind.name()),
    }
}

/// Inverse of `card_kind_to_string` - parses e.g. `"Creature:Tiny"` back into
/// a `CardKind`, for `PyGame::pin_kind` (live-tracking a physical game).
fn parse_card_kind(s: &str) -> Option<CardKind> {
    let (prefix, rest) = s.split_once(':')?;
    match prefix {
        "Creature" => Tier::parse(rest).map(CardKind::Creature),
        "Nasty" => NastyKind::parse(rest).map(CardKind::Nasty),
        "Thingamabob" => ThingamabobKind::parse(rest).map(CardKind::Thingamabob),
        _ => None,
    }
}

// PyAction

#[pyclass(name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    pub(crate) inner: Action,
}

impl PyAction {
    fn wrap(inner: Action) -> Self {
        PyAction { inner }
    }
}

#[pymethods]
impl PyAction {
    #[staticmethod]
    fn submit_deal(cards: [CardId; 3]) -> Self {
        Self::wrap(Action::SubmitDeal { cards })
    }

    #[staticmethod]
    fn reveal_card(card: CardId) -> Self {
        Self::wrap(Action::RevealCard { card })
    }

    #[staticmethod]
    fn buyer_peek(target_seller: PlayerId) -> Self {
        Self::wrap(Action::BuyerPeek { target_seller })
    }

    #[staticmethod]
    fn pass_thingamabob_window() -> Self {
        Self::wrap(Action::PassThingamabobWindow)
    }

    #[staticmethod]
    fn choose_deal(seller: PlayerId) -> Self {
        Self::wrap(Action::ChooseDeal { seller })
    }

    #[staticmethod]
    fn respond_to_deal(reverse: bool) -> Self {
        Self::wrap(Action::RespondToDeal { reverse })
    }

    #[staticmethod]
    fn resolve_nasty_penalty(taken_cards: Vec<CardId>) -> Self {
        Self::wrap(Action::ResolveNastyPenalty { taken_cards })
    }

    #[staticmethod]
    fn play_platonic_isolator(card: CardId, target_player: PlayerId) -> Self {
        Self::wrap(Action::PlayThingamabob { card, params: ThingamabobParams::PlatonicIsolator { target_player } })
    }

    #[staticmethod]
    fn play_remove_from_deals(card: CardId, removals: Vec<(PlayerId, CardId)>) -> Self {
        Self::wrap(Action::PlayThingamabob { card, params: ThingamabobParams::RemoveFromDeals { removals } })
    }

    #[staticmethod]
    fn play_cryptozootic_expander(card: CardId, hand_card: CardId, target_deal: PlayerId) -> Self {
        Self::wrap(Action::PlayThingamabob { card, params: ThingamabobParams::CryptozooticExpander { hand_card, target_deal } })
    }

    #[staticmethod]
    fn play_spectroelectric_optimeter(card: CardId, target_deal: PlayerId, target_card: CardId) -> Self {
        Self::wrap(Action::PlayThingamabob { card, params: ThingamabobParams::SpectroelectricOptimeter { target_deal, target_card } })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        action_to_pydict(py, &self.inner)
    }
}

fn action_to_pydict<'py>(py: Python<'py>, action: &Action) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match action {
        Action::SubmitDeal { cards } => {
            d.set_item("type", "submit_deal")?;
            d.set_item("cards", cards.to_vec())?;
        }
        Action::RevealCard { card } => {
            d.set_item("type", "reveal_card")?;
            d.set_item("card", card)?;
        }
        Action::BuyerPeek { target_seller } => {
            d.set_item("type", "buyer_peek")?;
            d.set_item("target_seller", target_seller)?;
        }
        Action::PlayThingamabob { card, params } => {
            d.set_item("type", "play_thingamabob")?;
            d.set_item("card", card)?;
            match params {
                ThingamabobParams::PlatonicIsolator { target_player } => {
                    d.set_item("effect", "platonic_isolator")?;
                    d.set_item("target_player", target_player)?;
                }
                ThingamabobParams::RemoveFromDeals { removals } => {
                    d.set_item("effect", "remove_from_deals")?;
                    d.set_item("removals", removals.clone())?;
                }
                ThingamabobParams::CryptozooticExpander { hand_card, target_deal } => {
                    d.set_item("effect", "cryptozootic_expander")?;
                    d.set_item("hand_card", hand_card)?;
                    d.set_item("target_deal", target_deal)?;
                }
                ThingamabobParams::SpectroelectricOptimeter { target_deal, target_card } => {
                    d.set_item("effect", "spectroelectric_optimeter")?;
                    d.set_item("target_deal", target_deal)?;
                    d.set_item("target_card", target_card)?;
                }
            }
        }
        Action::PassThingamabobWindow => {
            d.set_item("type", "pass_thingamabob_window")?;
        }
        Action::ChooseDeal { seller } => {
            d.set_item("type", "choose_deal")?;
            d.set_item("seller", seller)?;
        }
        Action::RespondToDeal { reverse } => {
            d.set_item("type", "respond_to_deal")?;
            d.set_item("reverse", reverse)?;
        }
        Action::ResolveNastyPenalty { taken_cards } => {
            d.set_item("type", "resolve_nasty_penalty")?;
            d.set_item("taken_cards", taken_cards.clone())?;
        }
    }
    Ok(d)
}

fn event_to_pydict<'py>(py: Python<'py>, event: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match event {
        Event::DealSubmitted { player } => {
            d.set_item("type", "deal_submitted")?;
            d.set_item("player", player)?;
        }
        Event::CardRevealed { seller, card } => {
            d.set_item("type", "card_revealed")?;
            d.set_item("seller", seller)?;
            d.set_item("card", card)?;
        }
        Event::BuyerPeeked { target_seller, card } => {
            d.set_item("type", "buyer_peeked")?;
            d.set_item("target_seller", target_seller)?;
            d.set_item("card", card)?;
        }
        Event::ThingamabobPlayed { player, card, kind } => {
            d.set_item("type", "thingamabob_played")?;
            d.set_item("player", player)?;
            d.set_item("card", card)?;
            d.set_item("kind", kind.name())?;
        }
        Event::ThingamabobWindowClosed => {
            d.set_item("type", "thingamabob_window_closed")?;
        }
        Event::DealChosen { buyer, seller } => {
            d.set_item("type", "deal_chosen")?;
            d.set_item("buyer", buyer)?;
            d.set_item("seller", seller)?;
        }
        Event::DealResponded { active, reverse } => {
            d.set_item("type", "deal_responded")?;
            d.set_item("active", active)?;
            d.set_item("reverse", reverse)?;
        }
        Event::CardsAwarded { player, cards } => {
            d.set_item("type", "cards_awarded")?;
            d.set_item("player", player)?;
            d.set_item("cards", cards.clone())?;
        }
        Event::CardsDiscarded { cards } => {
            d.set_item("type", "cards_discarded")?;
            d.set_item("cards", cards.clone())?;
        }
        Event::NastySetTradedIn { player, kind } => {
            d.set_item("type", "nasty_set_traded_in")?;
            d.set_item("player", player)?;
            d.set_item("kind", kind.name())?;
        }
        Event::PointTokenStolen { from, to, amount } => {
            d.set_item("type", "point_token_stolen")?;
            d.set_item("from", from)?;
            d.set_item("to", to)?;
            d.set_item("amount", amount)?;
        }
        Event::CardsStolen { from, to, cards } => {
            d.set_item("type", "cards_stolen")?;
            d.set_item("from", from)?;
            d.set_item("to", to)?;
            d.set_item("cards", cards.clone())?;
        }
        Event::CreatureSetTradedIn { player, tier, tokens_gained } => {
            d.set_item("type", "creature_set_traded_in")?;
            d.set_item("player", player)?;
            d.set_item("tier", tier.to_string())?;
            d.set_item("tokens_gained", tokens_gained)?;
        }
        Event::HandRefilled { player, drawn } => {
            d.set_item("type", "hand_refilled")?;
            d.set_item("player", player)?;
            d.set_item("drawn", drawn)?;
        }
        Event::DrawPileReshuffledFromDiscard => {
            d.set_item("type", "draw_pile_reshuffled_from_discard")?;
        }
        Event::TurnLeaderPassed { new_leader } => {
            d.set_item("type", "turn_leader_passed")?;
            d.set_item("new_leader", new_leader)?;
        }
        Event::GameOver { winner } => {
            d.set_item("type", "game_over")?;
            d.set_item("winner", winner)?;
        }
    }
    Ok(d)
}

// PyObservation

#[pyclass(name = "ObservedDeal", get_all)]
#[derive(Clone)]
pub struct PyObservedDeal {
    pub seller: PlayerId,
    pub revealed_cards: Vec<CardId>,
    pub num_hidden: usize,
}

impl From<&ObservedDeal> for PyObservedDeal {
    fn from(d: &ObservedDeal) -> Self {
        PyObservedDeal { seller: d.seller, revealed_cards: d.revealed_cards.clone(), num_hidden: d.num_hidden }
    }
}

#[pymethods]
impl PyObservedDeal {
    fn __repr__(&self) -> String {
        format!("ObservedDeal(seller={}, revealed_cards={:?}, num_hidden={})", self.seller, self.revealed_cards, self.num_hidden)
    }
}

#[pyclass(name = "Observation", get_all)]
pub struct PyObservation {
    pub player: PlayerId,
    pub phase: String,
    pub turn_leader: PlayerId,
    pub own_hand: Vec<CardId>,
    pub collections: Vec<Vec<CardId>>,
    pub point_tokens: Vec<u32>,
    pub deals: Vec<PyObservedDeal>,
    pub draw_pile_len: usize,
    pub discard_pile_len: usize,
    pub winner: Option<PlayerId>,
}

impl From<Observation> for PyObservation {
    fn from(o: Observation) -> Self {
        PyObservation {
            player: o.player,
            phase: o.phase.to_string(),
            turn_leader: o.turn_leader,
            own_hand: o.own_hand,
            collections: o.collections,
            point_tokens: o.point_tokens,
            deals: o.deals.iter().map(PyObservedDeal::from).collect(),
            draw_pile_len: o.draw_pile_len,
            discard_pile_len: o.discard_pile_len,
            winner: o.winner,
        }
    }
}

#[pymethods]
impl PyObservation {
    fn __repr__(&self) -> String {
        format!(
            "Observation(player={}, phase={:?}, own_hand={:?}, point_tokens={:?})",
            self.player, self.phase, self.own_hand, self.point_tokens
        )
    }
}

// PyStepResult

#[pyclass(name = "StepResult")]
pub struct PyStepResult {
    events: Vec<Event>,
    #[pyo3(get)]
    done: bool,
    #[pyo3(get)]
    winner: Option<PlayerId>,
}

#[pymethods]
impl PyStepResult {
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.events.iter().map(|e| event_to_pydict(py, e)).collect()
    }

    fn __repr__(&self) -> String {
        format!("StepResult(events={}, done={}, winner={:?})", self.events.len(), self.done, self.winner)
    }
}

// PyGame

#[pyclass(name = "Game")]
pub struct PyGame {
    inner: GameState,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new(num_players: usize, deck_config_path: &str, seed: u64) -> PyResult<Self> {
        let deck = DeckConfig::from_file(std::path::Path::new(deck_config_path))
            .map_err(|e| PyValueError::new_err(format!("failed to load deck config: {e}")))?;
        let inner = GameState::new(num_players, deck, seed).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyGame { inner })
    }

    fn legal_actions(&self, player: PlayerId) -> Vec<PyAction> {
        self.inner.legal_actions(player).into_iter().map(PyAction::wrap).collect()
    }

    fn step(&mut self, player: PlayerId, action: &PyAction) -> PyResult<PyStepResult> {
        let events = self
            .inner
            .apply_action(player, action.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyStepResult { events, done: self.inner.is_game_over(), winner: self.inner.winner() })
    }

    fn observation(&self, player: PlayerId) -> PyObservation {
        PyObservation::from(self.inner.observation_for(player))
    }

    fn current_phase(&self) -> String {
        self.inner.current_phase().to_string()
    }

    fn active_players(&self) -> Vec<PlayerId> {
        self.inner.active_players()
    }

    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    fn win_threshold(&self) -> u32 {
        self.inner.win_threshold()
    }

    fn turn_leader(&self) -> PlayerId {
        self.inner.turn_leader()
    }

    fn is_game_over(&self) -> bool {
        self.inner.is_game_over()
    }

    fn winner(&self) -> Option<PlayerId> {
        self.inner.winner()
    }

    fn card_kind(&self, card: CardId) -> Option<String> {
        self.inner.card_kind(card).map(card_kind_to_string)
    }

    fn card_name(&self, card: CardId) -> Option<String> {
        self.inner.card_name(card).map(str::to_string)
    }

    /// Full internal hand contents, bypassing observation privacy - only
    /// meaningful for live-tracking, where the caller (not another in-game
    /// player) is the sole source of truth for what's really in play.
    fn player_hand(&self, player: PlayerId) -> Vec<CardId> {
        self.inner.player_hand(player).to_vec()
    }

    fn player_collection(&self, player: PlayerId) -> Vec<CardId> {
        self.inner.player_collection(player).to_vec()
    }

    fn player_point_tokens(&self, player: PlayerId) -> u32 {
        self.inner.player_point_tokens(player)
    }

    /// Still-hidden card ids in `seller`'s active deal (empty if none).
    fn hidden_cards_in_deal(&self, seller: PlayerId) -> Vec<CardId> {
        self.inner.hidden_cards_in_deal(seller)
    }

    fn is_pinned(&self, card: CardId) -> bool {
        self.inner.is_pinned(card)
    }

    /// Remaining not-yet-pinned supply per card class (e.g. `"Creature:Tiny"`
    /// -> how many more could still be truthfully `pin_kind`-ed).
    fn kind_supply(&self) -> std::collections::HashMap<String, u32> {
        self.inner.kind_supply().iter().map(|(k, v)| (card_kind_to_string(*k), *v)).collect()
    }

    /// For live-tracking a physical game: overwrites `card`'s kind to match
    /// what was actually revealed at the table (see `GameState::pin_kind`).
    fn pin_kind(&mut self, card: CardId, kind: &str) -> PyResult<()> {
        let kind = parse_card_kind(kind).ok_or_else(|| PyValueError::new_err(format!("unknown card kind: {kind}")))?;
        self.inner.pin_kind(card, kind).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Exposed to Python as `sell_me_a_sasquatch._native` (see
/// `python/pyproject.toml`'s `[tool.maturin] module-name`).
#[pymodule(name = "_native")]
fn sasquatch_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyObservedDeal>()?;
    m.add_class::<PyStepResult>()?;
    Ok(())
}
