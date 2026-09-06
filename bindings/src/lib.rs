//! PyO3 bindings exposing [`sasquatch_engine::game::GameState`] as
//! [`PyGame`].
//!
//! The boundary is crossed exactly once per micro-step on the training
//! path: [`PyGame::encode_observation`] writes the whole fixed-shape
//! observation straight into a caller-owned numpy buffer, and
//! [`PyGame::step_index`] applies the i-th legal action without ever
//! materializing a Python object per action. The richer object API
//! (`observation`, `legal_actions`, `step`) is still here for the web app
//! and for live-tracking a physical game, where one call per human decision
//! costs nothing.

use numpy::{PyReadwriteArray1, PyReadwriteArray2};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use sasquatch_engine::{
    action::{Action, ThingamabobParams},
    card::{CardId, CardKind, PlayerId, CARD_CLASS_NAMES},
    deck::DeckConfig,
    encode::{ACTION_FEAT_LEN, MAX_DEALS, MAX_PLAYERS, NOISE_LEN, NOISE_OFFSET, OBS_LEN},
    game::{Event, GameState, Observation, ObservedDeal},
};

/// A card kind's Python-facing name string.
fn card_kind_to_string(kind: CardKind) -> String {
    kind.name()
}

// PyAction

/// Python-facing wrapper around an [`Action`].
#[pyclass(name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    pub(crate) inner: Action,
}

impl PyAction {
    /// Wraps an [`Action`] for Python.
    fn wrap(inner: Action) -> Self {
        PyAction { inner }
    }
}

#[pymethods]
impl PyAction {
    /// Builds a [`Action::SubmitDeal`].
    #[staticmethod]
    fn submit_deal(cards: [CardId; 3]) -> Self {
        Self::wrap(Action::SubmitDeal { cards })
    }

    /// Two-player mode only: builds a [`Action::TwoPlayerSubmitDeal`].
    /// Splits three of the active player's own hand cards between
    /// `own_pile` (theirs again on accept) and `other_pile` (the
    /// opponent's on accept); sizes can be any split summing to three.
    #[staticmethod]
    fn two_player_submit_deal(own_pile: Vec<CardId>, other_pile: Vec<CardId>) -> Self {
        Self::wrap(Action::TwoPlayerSubmitDeal {
            own_pile,
            other_pile,
        })
    }

    /// Builds a [`Action::RevealCard`].
    #[staticmethod]
    fn reveal_card(card: CardId) -> Self {
        Self::wrap(Action::RevealCard { card })
    }

    /// Builds a [`Action::BuyerPeek`].
    #[staticmethod]
    fn buyer_peek(target_seller: PlayerId) -> Self {
        Self::wrap(Action::BuyerPeek { target_seller })
    }

    /// Builds a [`Action::PassThingamabobWindow`].
    #[staticmethod]
    fn pass_thingamabob_window() -> Self {
        Self::wrap(Action::PassThingamabobWindow)
    }

    /// Builds a [`Action::ChooseDeal`].
    #[staticmethod]
    fn choose_deal(seller: PlayerId) -> Self {
        Self::wrap(Action::ChooseDeal { seller })
    }

    /// Builds a [`Action::RespondToDeal`].
    #[staticmethod]
    fn respond_to_deal(reverse: bool) -> Self {
        Self::wrap(Action::RespondToDeal { reverse })
    }

    /// Builds a [`Action::ResolveNastyPenalty`].
    #[staticmethod]
    fn resolve_nasty_penalty(taken_cards: Vec<CardId>) -> Self {
        Self::wrap(Action::ResolveNastyPenalty { taken_cards })
    }

    /// Builds a Platonic Isolator play.
    #[staticmethod]
    fn play_platonic_isolator(card: CardId, target_player: PlayerId) -> Self {
        Self::wrap(Action::PlayThingamabob {
            card,
            params: ThingamabobParams::PlatonicIsolator { target_player },
        })
    }

    /// Builds a Detrital Repositioner or Super Detrital Repositioner play.
    #[staticmethod]
    fn play_remove_from_deals(card: CardId, removals: Vec<(PlayerId, CardId)>) -> Self {
        Self::wrap(Action::PlayThingamabob {
            card,
            params: ThingamabobParams::RemoveFromDeals { removals },
        })
    }

    /// Builds a Cryptozootic Expander play.
    #[staticmethod]
    fn play_cryptozootic_expander(card: CardId, hand_card: CardId, target_deal: PlayerId) -> Self {
        Self::wrap(Action::PlayThingamabob {
            card,
            params: ThingamabobParams::CryptozooticExpander {
                hand_card,
                target_deal,
            },
        })
    }

    /// Builds a Spectroelectric Optimeter play.
    #[staticmethod]
    fn play_spectroelectric_optimeter(
        card: CardId,
        target_deal: PlayerId,
        target_card: CardId,
    ) -> Self {
        Self::wrap(Action::PlayThingamabob {
            card,
            params: ThingamabobParams::SpectroelectricOptimeter {
                target_deal,
                target_card,
            },
        })
    }

    /// Debug representation.
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Converts this action to a Python dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        action_to_pydict(py, &self.inner)
    }
}

/// Converts an [`Action`] into a Python dict with a `"type"` discriminator.
fn action_to_pydict<'py>(py: Python<'py>, action: &Action) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match action {
        Action::SubmitDeal { cards } => {
            d.set_item("type", "submit_deal")?;
            d.set_item("cards", cards.to_vec())?;
        }
        Action::TwoPlayerSubmitDeal {
            own_pile,
            other_pile,
        } => {
            d.set_item("type", "two_player_submit_deal")?;
            d.set_item("own_pile", own_pile.clone())?;
            d.set_item("other_pile", other_pile.clone())?;
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
                ThingamabobParams::CryptozooticExpander {
                    hand_card,
                    target_deal,
                } => {
                    d.set_item("effect", "cryptozootic_expander")?;
                    d.set_item("hand_card", hand_card)?;
                    d.set_item("target_deal", target_deal)?;
                }
                ThingamabobParams::SpectroelectricOptimeter {
                    target_deal,
                    target_card,
                } => {
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

/// Converts an [`Event`] into a Python dict with a `"type"` discriminator.
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
        Event::BuyerPeeked {
            target_seller,
            card,
        } => {
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
        Event::CreatureSetTradedIn {
            player,
            tier,
            tokens_gained,
        } => {
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

/// Python-facing wrapper around an [`ObservedDeal`].
#[pyclass(name = "ObservedDeal", get_all)]
#[derive(Clone)]
pub struct PyObservedDeal {
    /// The seller who made this deal.
    pub seller: PlayerId,
    /// Cards currently revealed in the deal.
    pub revealed_cards: Vec<CardId>,
    /// Number of cards still hidden.
    pub num_hidden: usize,
}

impl From<&ObservedDeal> for PyObservedDeal {
    fn from(d: &ObservedDeal) -> Self {
        PyObservedDeal {
            seller: d.seller,
            revealed_cards: d.revealed_cards.clone(),
            num_hidden: d.num_hidden,
        }
    }
}

#[pymethods]
impl PyObservedDeal {
    /// Debug representation.
    fn __repr__(&self) -> String {
        format!(
            "ObservedDeal(seller={}, revealed_cards={:?}, num_hidden={})",
            self.seller, self.revealed_cards, self.num_hidden
        )
    }
}

/// Python-facing wrapper around an [`Observation`].
#[pyclass(name = "Observation", get_all)]
pub struct PyObservation {
    /// The player this observation is for.
    pub player: PlayerId,
    /// Current phase name.
    pub phase: String,
    /// Current turn leader.
    pub turn_leader: PlayerId,
    /// This player's own hand.
    pub own_hand: Vec<CardId>,
    /// Every player's collection, public information.
    pub collections: Vec<Vec<CardId>>,
    /// Every player's point token count.
    pub point_tokens: Vec<u32>,
    /// Active deals, with hidden cards collapsed.
    pub deals: Vec<PyObservedDeal>,
    /// Number of cards left in the draw pile.
    pub draw_pile_len: usize,
    /// Number of cards in the discard pile.
    pub discard_pile_len: usize,
    /// The winner, if the game has ended.
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
    /// Debug representation.
    fn __repr__(&self) -> String {
        format!(
            "Observation(player={}, phase={:?}, own_hand={:?}, point_tokens={:?})",
            self.player, self.phase, self.own_hand, self.point_tokens
        )
    }
}

// PyStepResult

/// Result of applying one action: the resulting events plus whether the game ended.
#[pyclass(name = "StepResult")]
pub struct PyStepResult {
    /// Events produced by the action.
    events: Vec<Event>,
    /// Whether the game has now ended.
    #[pyo3(get)]
    done: bool,
    /// The winner, if the game has now ended.
    #[pyo3(get)]
    winner: Option<PlayerId>,
}

#[pymethods]
impl PyStepResult {
    /// The events produced by the action, as Python dicts.
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.events.iter().map(|e| event_to_pydict(py, e)).collect()
    }

    /// Debug representation.
    fn __repr__(&self) -> String {
        format!(
            "StepResult(events={}, done={}, winner={:?})",
            self.events.len(),
            self.done,
            self.winner
        )
    }
}

// PyDeck

/// A parsed `deck.toml`, kept alive across episodes.
///
/// Re-reading and re-parsing the deck config on every `reset()` used to
/// cost a file read plus a full TOML parse per episode, easily more than
/// the episode itself once the engine got fast. Parse once, clone the
/// in-memory config per game.
#[pyclass(name = "Deck")]
#[derive(Clone)]
pub struct PyDeck {
    /// The parsed deck config.
    inner: DeckConfig,
    /// Path this deck was loaded from, if any.
    #[pyo3(get)]
    path: Option<String>,
}

#[pymethods]
impl PyDeck {
    /// Loads a deck config from a TOML file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = DeckConfig::from_file(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("failed to load deck config: {e}")))?;
        Ok(PyDeck {
            inner,
            path: Some(path.to_string()),
        })
    }

    /// Parses a deck config from a TOML string.
    #[staticmethod]
    fn from_toml(text: &str) -> PyResult<Self> {
        let inner = DeckConfig::from_toml_str(text)
            .map_err(|e| PyValueError::new_err(format!("failed to parse deck config: {e}")))?;
        Ok(PyDeck { inner, path: None })
    }

    /// Total number of physical cards in the deck.
    fn total_cards(&self) -> usize {
        self.inner.total_cards()
    }

    /// Exact width the ordinal action space needs for a table of
    /// `num_players` playing this deck, see [`GameState::max_legal_actions`].
    fn max_legal_actions(&self, num_players: usize) -> PyResult<usize> {
        let game = GameState::new(num_players, self.inner.clone(), 0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(game.max_legal_actions())
    }

    /// Debug representation.
    fn __repr__(&self) -> String {
        format!(
            "Deck(cards={}, path={:?})",
            self.inner.total_cards(),
            self.path
        )
    }
}

// PyGame

/// Python-facing wrapper around a [`GameState`].
#[pyclass(name = "Game")]
pub struct PyGame {
    /// The underlying game state.
    inner: GameState,
    /// Memoized [`GameState::legal_actions`] for the current state. A step
    /// needs the action list twice, once to size the mask and once to
    /// resolve the chosen index, and enumerating it is the engine's most
    /// expensive operation.
    legal_cache: Option<(PlayerId, Vec<Action>)>,
}

impl PyGame {
    /// The (possibly cached) legal actions for `player` right now.
    fn legal_for(&mut self, player: PlayerId) -> &[Action] {
        if !matches!(&self.legal_cache, Some((p, _)) if *p == player) {
            self.legal_cache = Some((player, self.inner.legal_actions(player)));
        }
        &self.legal_cache.as_ref().expect("just populated").1
    }
}

#[pymethods]
impl PyGame {
    /// `deck` is either a [`PyDeck`] (preferred, parsed once and reused for
    /// every episode) or a path to a `deck.toml`, which parses it afresh.
    #[new]
    #[pyo3(signature = (num_players, deck, seed, first_player=None))]
    fn new(
        num_players: usize,
        deck: &Bound<'_, PyAny>,
        seed: u64,
        first_player: Option<PlayerId>,
    ) -> PyResult<Self> {
        let config = if let Ok(d) = deck.extract::<PyDeck>() {
            d.inner
        } else {
            let path: String = deck.extract().map_err(|_| {
                PyValueError::new_err("deck must be a Deck or a path to a deck.toml")
            })?;
            DeckConfig::from_file(std::path::Path::new(&path))
                .map_err(|e| PyValueError::new_err(format!("failed to load deck config: {e}")))?
        };
        let inner = GameState::new_with_starting_leader(num_players, config, seed, first_player)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyGame {
            inner,
            legal_cache: None,
        })
    }

    /// Every legal action for `player` right now.
    fn legal_actions(&mut self, player: PlayerId) -> Vec<PyAction> {
        self.legal_for(player)
            .iter()
            .cloned()
            .map(PyAction::wrap)
            .collect()
    }

    /// How many actions are legal for `player` right now. The ordinal
    /// action space's mask is exactly `[1] * this + [0] * (width - this)`,
    /// so the training loop never needs the actions themselves.
    fn legal_action_count(&mut self, player: PlayerId) -> usize {
        self.legal_for(player).len()
    }

    /// Applies one action for `player`, returning the resulting events and game status.
    fn step(&mut self, player: PlayerId, action: &PyAction) -> PyResult<PyStepResult> {
        let events = self
            .inner
            .apply_action(player, action.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.legal_cache = None;
        Ok(PyStepResult {
            events,
            done: self.inner.is_game_over(),
            winner: self.inner.winner(),
        })
    }

    /// Applies the `index`-th currently-legal action, the ordinal action
    /// space's step, with no Python `Action` object built along the way.
    /// Returns `(done, winner)`; use [`Self::step`] when the events matter.
    fn step_index(&mut self, player: PlayerId, index: usize) -> PyResult<(bool, Option<PlayerId>)> {
        let action = self
            .legal_for(player)
            .get(index)
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "action index {index} out of range for player {player}"
                ))
            })?
            .clone();
        self.inner
            .apply_action(player, action)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.legal_cache = None;
        Ok((self.inner.is_game_over(), self.inner.winner()))
    }

    /// Writes `player`'s fixed-shape observation into `out` (a contiguous
    /// float32 array of length [`OBS_LEN`]), in place. Returns the number
    /// of currently-legal actions, since the caller needs it for the
    /// action mask on the very same step.
    fn encode_observation(
        &mut self,
        player: PlayerId,
        mut out: PyReadwriteArray1<f32>,
    ) -> PyResult<usize> {
        let slice = out
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("observation buffer must be contiguous float32"))?;
        if slice.len() != OBS_LEN {
            return Err(PyValueError::new_err(format!(
                "observation buffer must have length {OBS_LEN}, got {}",
                slice.len()
            )));
        }
        self.inner.encode_observation(player, slice);
        Ok(self.legal_for(player).len())
    }

    /// The training loop's single boundary crossing: fills `obs` with
    /// `player`'s state vector and `actions` with one feature row per
    /// currently-legal action, and returns how many rows are live (the
    /// rest are zeroed, and masked out by the caller). See
    /// [`GameState::encode_observation`] and [`GameState::encode_actions`].
    fn encode(
        &mut self,
        player: PlayerId,
        mut obs: PyReadwriteArray1<f32>,
        mut actions: PyReadwriteArray2<f32>,
    ) -> PyResult<usize> {
        let obs_slice = obs
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("observation buffer must be contiguous float32"))?;
        if obs_slice.len() != OBS_LEN {
            return Err(PyValueError::new_err(format!(
                "observation buffer must have length {OBS_LEN}, got {}",
                obs_slice.len()
            )));
        }
        let act_slice = actions.as_slice_mut().map_err(|_| {
            PyValueError::new_err("action-feature buffer must be C-contiguous float32")
        })?;
        if act_slice.len() % ACTION_FEAT_LEN != 0 {
            return Err(PyValueError::new_err(format!(
                "action-feature buffer must be (n, {ACTION_FEAT_LEN})"
            )));
        }
        self.inner.encode_observation(player, obs_slice);
        // move the cache out so the action list and self.inner are
        // provably disjoint borrows, no per-step clone of the action list
        let mut cache = self.legal_cache.take();
        if !matches!(&cache, Some((p, _)) if *p == player) {
            cache = Some((player, self.inner.legal_actions(player)));
        }
        let legal = &cache.as_ref().expect("just populated").1;
        let count = legal.len();
        self.inner.encode_actions(player, legal, act_slice);
        self.legal_cache = cache;
        Ok(count)
    }

    /// `player`'s filtered observation of the current game state.
    fn observation(&self, player: PlayerId) -> PyObservation {
        PyObservation::from(self.inner.observation_for(player))
    }

    /// Stable string name of the current phase.
    fn current_phase(&self) -> String {
        self.inner.current_phase().to_string()
    }

    /// The one seat to act right now, or `None` if the game is over.
    fn active_player(&self) -> Option<PlayerId> {
        self.inner.active_player()
    }

    /// PettingZoo-shaped view of [`Self::active_player`].
    fn active_players(&self) -> Vec<PlayerId> {
        self.inner.active_players()
    }

    /// Number of players at the table.
    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    /// Point tokens needed to win.
    fn win_threshold(&self) -> u32 {
        self.inner.win_threshold()
    }

    /// Exact upper bound on the number of legal actions for this deck and table size.
    fn max_legal_actions(&self) -> usize {
        self.inner.max_legal_actions()
    }

    /// Turns completed so far.
    fn turn_index(&self) -> u32 {
        self.inner.turn_index()
    }

    /// Current turn leader.
    fn turn_leader(&self) -> PlayerId {
        self.inner.turn_leader()
    }

    /// Whether the game has ended.
    fn is_game_over(&self) -> bool {
        self.inner.is_game_over()
    }

    /// The winning player, if the game has ended.
    fn winner(&self) -> Option<PlayerId> {
        self.inner.winner()
    }

    /// Every seat's point tokens. Public information, and cheap; reward
    /// shaping reads it on every single step, so it must not go through
    /// the full [`Self::observation`] (which clones every collection).
    fn point_tokens(&self) -> Vec<u32> {
        (0..self.inner.num_players())
            .map(|p| self.inner.player_point_tokens(p))
            .collect()
    }

    /// A card's mechanical kind, by name.
    fn card_kind(&self, card: CardId) -> Option<String> {
        self.inner.card_kind(card).map(card_kind_to_string)
    }

    /// A card's display name.
    fn card_name(&self, card: CardId) -> Option<String> {
        self.inner.card_name(card).map(str::to_string)
    }

    /// Full internal hand contents, bypassing observation privacy. Only
    /// meaningful for live-tracking, where the caller (not another in-game
    /// player) is the sole source of truth for what's really in play.
    fn player_hand(&self, player: PlayerId) -> Vec<CardId> {
        self.inner.player_hand(player).to_vec()
    }

    /// A player's current collection.
    fn player_collection(&self, player: PlayerId) -> Vec<CardId> {
        self.inner.player_collection(player).to_vec()
    }

    /// A player's current point token count.
    fn player_point_tokens(&self, player: PlayerId) -> u32 {
        self.inner.player_point_tokens(player)
    }

    /// Still-hidden card ids in `seller`'s active deal, empty if none.
    fn hidden_cards_in_deal(&self, seller: PlayerId) -> Vec<CardId> {
        self.inner.hidden_cards_in_deal(seller)
    }

    /// Whether a card has already been pinned via [`Self::pin_kind`].
    fn is_pinned(&self, card: CardId) -> bool {
        self.inner.is_pinned(card)
    }

    /// Remaining not-yet-pinned supply per card class (e.g.
    /// `"Creature:Tiny"` maps to how many more could still be truthfully
    /// pinned via [`Self::pin_kind`]).
    fn kind_supply(&self) -> std::collections::HashMap<String, u32> {
        self.inner
            .kind_supply()
            .into_iter()
            .map(|(k, v)| (card_kind_to_string(k), v))
            .collect()
    }

    /// For live-tracking a physical game: overwrites `card`'s kind to
    /// match what was actually revealed at the table, see
    /// [`GameState::pin_kind`].
    fn pin_kind(&mut self, card: CardId, kind: &str) -> PyResult<()> {
        let parsed = CardKind::parse(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown card kind: {kind}")))?;
        self.legal_cache = None;
        self.inner
            .pin_kind(card, parsed)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Exposed to Python as `sell_me_a_sasquatch._native`, see
/// `python/pyproject.toml`'s `[tool.maturin] module-name`.
#[pymodule(name = "_native")]
fn sasquatch_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyDeck>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyObservedDeal>()?;
    m.add_class::<PyStepResult>()?;
    // observation layout, so the python spaces are derived from the
    // encoder rather than re-declared alongside it and left to drift
    m.add("OBS_LEN", OBS_LEN)?;
    m.add("NOISE_OFFSET", NOISE_OFFSET)?;
    m.add("NOISE_LEN", NOISE_LEN)?;
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    m.add("MAX_DEALS", MAX_DEALS)?;
    m.add("ACTION_FEAT_LEN", ACTION_FEAT_LEN)?;
    m.add("CARD_CLASS_NAMES", CARD_CLASS_NAMES.to_vec())?;
    Ok(())
}
