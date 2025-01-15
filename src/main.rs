mod ataxx;
mod mcts;
mod networks;
mod tree;
mod uai;

pub use ataxx::{Board, GameState, Move};
pub use mcts::{Limits, MctsParams, Searcher};
pub use tree::Tree;

fn main() {
    uai::run();
}
