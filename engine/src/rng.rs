//! Seedable RNG wrapper. All randomness in the engine (shuffling, and the
//! §2.3 step 3 forced-random hidden-card reveal) flows through this single
//! type so `(seed, action_sequence)` fully determines an episode.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub struct GameRng {
    inner: ChaCha8Rng,
}

impl GameRng {
    pub fn from_seed(seed: u64) -> Self {
        Self { inner: ChaCha8Rng::seed_from_u64(seed) }
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        slice.shuffle(&mut self.inner);
    }

    /// Uniform index in `0..len`. Panics if `len == 0`.
    pub fn gen_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "gen_index called with len == 0");
        self.inner.random_range(0..len)
    }
}
