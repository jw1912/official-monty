mod helpers;
mod iteration;
mod params;

pub use helpers::SearchHelpers;
pub use params::MctsParams;

use crate::{
    chess::{GameState, Move},
    networks::{PolicyNetwork, ValueNetwork},
    tree::{NodePtr, Tree},
};

use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
    time::Instant,
};

pub static REPORT_ITERS: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy)]
pub struct Limits {
    pub max_time: Option<u128>,
    pub opt_time: Option<u128>,
    pub max_depth: usize,
    pub max_nodes: usize,
}

#[derive(Default)]
pub struct SearchStats {
    pub total_nodes: AtomicUsize,
    pub total_iters: AtomicUsize,
    pub main_iters: AtomicUsize,
    pub avg_depth: AtomicUsize,
    pub seldepth: AtomicUsize,
}

pub struct Searcher<'a> {
    tree: &'a Tree,
    params: &'a MctsParams,
    policy: &'a PolicyNetwork,
    value: &'a ValueNetwork,
    abort: &'a AtomicBool,
}

impl<'a> Searcher<'a> {
    pub fn new(
        tree: &'a Tree,
        params: &'a MctsParams,
        policy: &'a PolicyNetwork,
        value: &'a ValueNetwork,
        abort: &'a AtomicBool,
    ) -> Self {
        Self {
            tree,
            params,
            policy,
            value,
            abort,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn playout_until_full_main(
        &self,
        limits: &Limits,
        timer: &Instant,
        #[cfg(not(feature = "uci-minimal"))] timer_last_output: &mut Instant,
        search_stats: &SearchStats,
        best_move: &mut Move,
        best_move_changes: &mut i32,
        previous_score: &mut f32,
        #[cfg(not(feature = "uci-minimal"))] uci_output: bool,
    ) {
        if self.playout_until_full_internal(search_stats, true, || {
            self.check_limits(
                limits,
                timer,
                #[cfg(not(feature = "uci-minimal"))]
                timer_last_output,
                search_stats,
                best_move,
                best_move_changes,
                previous_score,
                #[cfg(not(feature = "uci-minimal"))]
                uci_output,
            )
        }) {
            self.abort.store(true, Ordering::Relaxed);
        }
    }

    fn playout_until_full_worker(&self, search_stats: &SearchStats) {
        let _ = self.playout_until_full_internal(search_stats, false, || false);
    }

    fn playout_until_full_internal<F>(
        &self,
        search_stats: &SearchStats,
        main_thread: bool,
        mut stop: F,
    ) -> bool
    where
        F: FnMut() -> bool,
    {
        loop {
            let mut pos = self.tree.root_position().clone();
            let mut this_depth = 0;

            if iteration::perform_one(self, &mut pos, self.tree.root_node(), &mut this_depth)
                .is_none()
            {
                return false;
            }

            search_stats.total_iters.fetch_add(1, Ordering::Relaxed);
            search_stats
                .total_nodes
                .fetch_add(this_depth, Ordering::Relaxed);
            search_stats
                .seldepth
                .fetch_max(this_depth - 1, Ordering::Relaxed);
            if main_thread {
                search_stats.main_iters.fetch_add(1, Ordering::Relaxed);
            }

            // proven checkmate
            if self.tree[self.tree.root_node()].is_terminal() {
                return true;
            }

            // stop signal sent
            if self.abort.load(Ordering::Relaxed) {
                return true;
            }

            if stop() {
                return true;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_limits(
        &self,
        limits: &Limits,
        timer: &Instant,
        #[cfg(not(feature = "uci-minimal"))] timer_last_output: &mut Instant,
        search_stats: &SearchStats,
        best_move: &mut Move,
        best_move_changes: &mut i32,
        previous_score: &mut f32,
        #[cfg(not(feature = "uci-minimal"))] uci_output: bool,
    ) -> bool {
        let iters = search_stats.main_iters.load(Ordering::Relaxed);

        if search_stats.total_iters.load(Ordering::Relaxed) >= limits.max_nodes {
            return true;
        }

        if iters % 128 == 0 {
            if let Some(time) = limits.max_time {
                if timer.elapsed().as_millis() >= time {
                    return true;
                }
            }

            let (_, new_best_move, _) = self.get_best_action(self.tree.root_node());
            if new_best_move != *best_move {
                *best_move = new_best_move;
                *best_move_changes += 1;
            }
        }

        if iters % 4096 == 0 {
            if let Some(time) = limits.opt_time {
                let (should_stop, score) = SearchHelpers::soft_time_cutoff(
                    self,
                    timer,
                    *previous_score,
                    *best_move_changes,
                    iters,
                    time,
                );

                if should_stop {
                    return true;
                }

                if iters % 16384 == 0 {
                    *best_move_changes = 0;
                }

                *previous_score = if *previous_score == f32::NEG_INFINITY {
                    score
                } else {
                    (score + 2.0 * *previous_score) / 3.0
                };
            }
        }

        // define "depth" as the average depth of selection
        let total_depth = search_stats.total_nodes.load(Ordering::Relaxed)
            - search_stats.total_iters.load(Ordering::Relaxed);
        let new_depth = total_depth / search_stats.total_iters.load(Ordering::Relaxed);
        if new_depth > search_stats.avg_depth.load(Ordering::Relaxed) {
            search_stats.avg_depth.store(new_depth, Ordering::Relaxed);
            if new_depth >= limits.max_depth {
                return true;
            }

            #[cfg(not(feature = "uci-minimal"))]
            if uci_output {
                self.search_report(
                    new_depth,
                    search_stats.seldepth.load(Ordering::Relaxed),
                    timer,
                    search_stats.total_nodes.load(Ordering::Relaxed),
                    search_stats.total_iters.load(Ordering::Relaxed),
                );

                *timer_last_output = Instant::now();
            }
        }

        #[cfg(not(feature = "uci-minimal"))]
        if uci_output && iters % 8192 == 0 && timer_last_output.elapsed().as_secs() >= 15 {
            self.search_report(
                search_stats.avg_depth.load(Ordering::Relaxed),
                search_stats.seldepth.load(Ordering::Relaxed),
                timer,
                search_stats.total_nodes.load(Ordering::Relaxed),
                search_stats.total_iters.load(Ordering::Relaxed),
            );

            *timer_last_output = Instant::now();
        }

        false
    }

    pub fn search(
        &self,
        threads: usize,
        limits: Limits,
        uci_output: bool,
        update_nodes: &mut usize,
    ) -> (Move, f32) {
        let timer = Instant::now();
        #[cfg(not(feature = "uci-minimal"))]
        let mut timer_last_output = Instant::now();

        let pos = self.tree.root_position();
        let node = self.tree.root_node();

        // the root node is added to an empty tree, **and not counted** towards the
        // total node count, in order for `go nodes 1` to give the expected result
        if self.tree.is_empty() {
            let ptr = self.tree.push_new_node().unwrap();

            assert_eq!(node, ptr);

            self.tree[ptr].clear();
            self.tree.expand_node(ptr, pos, self.params, self.policy, 1);

            let root_eval = pos.get_value_wdl(self.value, self.params);
            self.tree[ptr].update(1.0 - root_eval);
        }
        // relabel preexisting root policies with root PST value
        else if self.tree[node].has_children() {
            self.tree
                .relabel_policy(node, pos, self.params, self.policy, 1);

            let first_child_ptr = self.tree[node].actions();

            for action in 0..self.tree[node].num_actions() {
                let ptr = first_child_ptr + action;

                if ptr.is_null() || !self.tree[ptr].has_children() {
                    continue;
                }

                let mut child = pos.clone();
                child.make_move(self.tree[ptr].parent_move());
                self.tree
                    .relabel_policy(ptr, &child, self.params, self.policy, 2);
            }
        }

        let search_stats = SearchStats::default();

        let mut best_move = Move::NULL;
        let mut best_move_changes = 0;
        let mut previous_score = f32::NEG_INFINITY;

        // search loop
        while !self.abort.load(Ordering::Relaxed) {
            thread::scope(|s| {
                s.spawn(|| {
                    self.playout_until_full_main(
                        &limits,
                        &timer,
                        #[cfg(not(feature = "uci-minimal"))]
                        &mut timer_last_output,
                        &search_stats,
                        &mut best_move,
                        &mut best_move_changes,
                        &mut previous_score,
                        #[cfg(not(feature = "uci-minimal"))]
                        uci_output,
                    );
                });

                for _ in 0..threads - 1 {
                    s.spawn(|| self.playout_until_full_worker(&search_stats));
                }
            });

            if !self.abort.load(Ordering::Relaxed) {
                self.tree.flip(true, threads);
            }
        }

        *update_nodes += search_stats.total_nodes.load(Ordering::Relaxed);

        if uci_output {
            self.search_report(
                search_stats.avg_depth.load(Ordering::Relaxed).max(1),
                search_stats.seldepth.load(Ordering::Relaxed),
                &timer,
                search_stats.total_nodes.load(Ordering::Relaxed),
                search_stats.total_iters.load(Ordering::Relaxed),
            );
        }

        let (_, mov, q) = self.get_best_action(self.tree.root_node());
        (mov, q)
    }

    fn search_report(
        &self,
        depth: usize,
        seldepth: usize,
        timer: &Instant,
        nodes: usize,
        iters: usize,
    ) {
        print!("info depth {depth} seldepth {seldepth} ");
        let (pv_line, score) = self.get_pv(depth);

        if score > 1.0 {
            print!("score mate {} ", pv_line.len().div_ceil(2));
        } else if score < 0.0 {
            print!("score mate -{} ", pv_line.len() / 2);
        } else {
            let cp = Searcher::get_cp(score);
            print!("score cp {cp:.0} ");
        }

        let nodes = if REPORT_ITERS.load(Ordering::Relaxed) {
            iters
        } else {
            nodes
        };
        let elapsed = timer.elapsed();
        let nps = nodes as f32 / elapsed.as_secs_f32();
        let ms = elapsed.as_millis();

        print!("time {ms} nodes {nodes} nps {nps:.0} pv");

        for mov in pv_line {
            print!(" {}", self.tree.root_position().conv_mov_to_str(mov));
        }

        println!();
    }

    fn get_pv(&self, mut depth: usize) -> (Vec<Move>, f32) {
        let mate = self.tree[self.tree.root_node()].is_terminal();

        let (mut ptr, mut mov, q) = self.get_best_action(self.tree.root_node());

        let score = if !ptr.is_null() {
            match self.tree[ptr].state() {
                GameState::Lost(_) => 1.1,
                GameState::Won(_) => -0.1,
                GameState::Draw => 0.5,
                GameState::Ongoing => q,
            }
        } else {
            q
        };

        let mut pv = Vec::new();
        let half = self.tree.half() > 0;

        while (mate || depth > 0) && !ptr.is_null() && ptr.half() == half {
            pv.push(mov);
            let idx = self.get_best_child(ptr);

            if idx == usize::MAX {
                break;
            }

            (ptr, mov, _) = self.get_best_action(ptr);
            depth = depth.saturating_sub(1);
        }

        (pv, score)
    }

    fn get_best_action(&self, node: NodePtr) -> (NodePtr, Move, f32) {
        let idx = self.get_best_child(node);
        let ptr = self.tree[node].actions() + idx;
        let child = &self.tree[ptr];
        (ptr, child.parent_move(), child.q())
    }

    fn get_best_child(&self, node: NodePtr) -> usize {
        self.tree.get_best_child_by_key(node, |child| {
            if child.visits() == 0 {
                f32::NEG_INFINITY
            } else {
                match child.state() {
                    GameState::Lost(n) => 1.0 + f32::from(n),
                    GameState::Won(n) => f32::from(n) - 256.0,
                    GameState::Draw => 0.5,
                    GameState::Ongoing => child.q(),
                }
            }
        })
    }

    fn get_cp(score: f32) -> f32 {
        let clamped_score = score.clamp(0.0, 1.0);
        let deviation = (clamped_score - 0.5).abs();
        let sign = (clamped_score - 0.5).signum();
        if deviation > 0.107 {
            (100.0 + 2923.0 * (deviation - 0.107)) * sign
        } else {
            let adjusted_score = 0.5 + (clamped_score - 0.5).powi(3) * 100.0;
            -200.0 * (1.0 / adjusted_score - 1.0).ln()
        }
    }

    pub fn display_moves(&self) {
        let first_child_ptr = self.tree[self.tree.root_node()].actions();
        for action in 0..self.tree[self.tree.root_node()].num_actions() {
            let child = &self.tree[first_child_ptr + action];
            let mov = self
                .tree
                .root_position()
                .conv_mov_to_str(child.parent_move());
            let q = child.q() * 100.0;
            println!(
                "{mov} -> {q:.2}% V({}) S({})",
                child.visits(),
                child.state()
            );
        }
    }
}
