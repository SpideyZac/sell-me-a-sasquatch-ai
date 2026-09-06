//! Win threshold and tie handling: a player wins once they've reached the
//! threshold and hold strictly more tokens than every other player,
//! checked at each end of turn.

mod common;

use common::*;
use sasquatch_engine::card::{CardKind, Tier};

/// After every simulated end-of-turn, recomputes the expected winner from
/// each player's Point Token count using the spec's exact rule and checks
/// it against `GameState::winner()`. Giant creatures (set_size 1) are used
/// to award tokens turn-by-turn in a controlled, incremental way.
#[test]
fn win_is_declared_exactly_when_a_player_meets_threshold_and_strictly_leads() {
    let mut game = new_test_game(4, 42);
    let threshold = game.win_threshold();

    for _ in 0..60 {
        if game.is_game_over() {
            break;
        }
        let Some(_) =
            stash_cards_of_kind_in_collection(&mut game, CardKind::Creature(Tier::Giant), 1)
        else {
            continue;
        };

        let tokens: Vec<u32> = (0..4).map(|p| game.player_point_tokens(p)).collect();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|&p| std::cmp::Reverse(tokens[p]));
        let expected_winner =
            if tokens[order[0]] >= threshold && tokens[order[0]] > tokens[order[1]] {
                Some(order[0])
            } else {
                None
            };

        assert_eq!(
            game.winner(),
            expected_winner,
            "tokens={tokens:?} threshold={threshold}"
        );
        if expected_winner.is_some() {
            assert!(game.is_game_over());
            return;
        }
    }
}

/// A tie at/above the threshold must NOT end the game - play must continue
/// until exactly one player strictly leads. Constructed directly against
/// the public API's observable token counts (see the loop above for the
/// general property); this test specifically asserts the game is still
/// running whenever two players are tied at or above threshold.
#[test]
fn tie_at_or_above_threshold_does_not_end_the_game() {
    let mut game = new_test_game(4, 99);
    let threshold = game.win_threshold();
    for _ in 0..60 {
        if game.is_game_over() {
            break;
        }
        stash_cards_of_kind_in_collection(&mut game, CardKind::Creature(Tier::Giant), 1);
        let tokens: Vec<u32> = (0..4).map(|p| game.player_point_tokens(p)).collect();
        let max = *tokens.iter().max().unwrap();
        let leaders = tokens.iter().filter(|&&t| t == max).count();
        if max >= threshold && leaders > 1 {
            assert!(
                !game.is_game_over(),
                "tied leaders at/above threshold must not end the game"
            );
        }
    }
}
