//! Core game engine for Sell Me a Sasquatch: rules, state, and RL-facing
//! encoding, with no I/O or bindings of its own.

/// Player actions.
pub mod action;
/// Card identity and the type catalog.
pub mod card;
/// In-progress deal state.
pub mod deal;
/// Deck config loading.
pub mod deck;
/// Observation encoding for RL.
pub mod encode;
/// Game state and rules.
pub mod game;
/// Turn phase state machine.
pub mod phase;
/// Per-player state.
pub mod player;
/// Seedable RNG wrapper.
pub mod rng;
/// Game statistics tracking.
pub mod stats;
