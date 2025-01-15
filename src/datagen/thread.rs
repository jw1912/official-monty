use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use crate::{
    ataxx::{Board, GameState},
    mcts::{Limits, MctsParams, Searcher},
    tree::Tree,
};

use super::{
    dest::Destination,
    format::{MontyAtaxxFormat, SearchData},
};

pub struct DatagenThread<'a> {
    params: MctsParams,
    dest: Arc<Mutex<Destination>>,
    stop: &'a AtomicBool,
    node_limit: usize,
}

impl<'a> DatagenThread<'a> {
    pub fn new(
        params: MctsParams,
        stop: &'a AtomicBool,
        dest: Arc<Mutex<Destination>>,
        node_limit: usize,
    ) -> Self {
        Self {
            params,
            dest,
            stop,
            node_limit,
        }
    }

    pub fn run(&mut self) {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            self.run_game();
        }
    }

    fn run_game(&mut self) {
        let mut pos = Board::default();

        let limits = Limits {
            max_depth: 64,
            max_nodes: self.node_limit,
            max_time: None,
        };

        let mut result = 0.5;

        let mut tree = Tree::new_mb(8, 1);
        let mut temp = 0.8;

        let mut game = MontyAtaxxFormat::default();

        // play out game
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return;
            }

            let abort = AtomicBool::new(false);
            tree.try_use_subtree(&pos, &None);
            let searcher = Searcher::new(pos, &tree, &self.params, &abort);

            let (bm, score) = searcher.search(1, limits, false, &mut 0, Some(temp));

            temp *= 0.9;
            if temp <= 0.2 {
                temp = 0.0;
            }

            let mut root_count = 0;
            pos.map_legal_moves(|_| root_count += 1);

            let dist = if root_count == 0 {
                None
            } else {
                let mut dist = Vec::new();

                let actions = { *tree[tree.root_node()].actions() };

                for action in 0..tree[tree.root_node()].num_actions() {
                    let node = &tree[actions + action];
                    dist.push((node.parent_move(), node.visits() as u32));
                }

                assert_eq!(root_count, dist.len());

                Some(dist)
            };

            let search_data = SearchData::new(bm, score, dist);

            game.push(search_data);

            pos.make(bm);

            let game_state = pos.game_state();
            match game_state {
                GameState::Ongoing => {}
                GameState::Draw => break,
                GameState::Lost(_) => {
                    if pos.stm() == 1 {
                        result = 1.0;
                    } else {
                        result = 0.0;
                    }
                    break;
                }
                GameState::Won(_) => {
                    if pos.stm() == 1 {
                        result = 0.0;
                    } else {
                        result = 1.0;
                    }
                    break;
                }
            }

            tree.clear(1);
        }

        game.result = result;

        if self.stop.load(Ordering::Relaxed) {
            return;
        }

        let mut dest = self.dest.lock().unwrap();
        dest.push(&game, self.stop);
    }
}
