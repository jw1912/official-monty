use crate::{
    chess::{ChessState, GameState},
    tree::{Node, NodePtr},
};

use super::{SearchHelpers, Searcher};

pub fn perform_one(
    searcher: &Searcher,
    pos: &mut ChessState,
    ptr: NodePtr,
    depth: &mut usize,
) -> Option<f32> {
    *depth += 1;

    let hash = pos.hash();
    let tree = searcher.tree;
    let node = &tree[ptr];

    let mut u = if node.is_terminal() || node.visits() == 0 {
        if node.visits() == 0 {
            node.set_state(pos.game_state());
        }

        // probe hash table to use in place of network
        if node.state() == GameState::Ongoing {
            if let Some(entry) = tree.probe_hash(hash) {
                entry.q()
            } else {
                get_utility(searcher, ptr, pos, hash)
            }
        } else {
            get_utility(searcher, ptr, pos, hash)
        }
    } else {
        // expand node on the second visit
        if node.is_not_expanded() {
            tree.expand_node(ptr, pos, searcher.params, searcher.policy, *depth)?;
        }

        // this node has now been accessed so we need to move its
        // children across if they are in the other tree half
        tree.fetch_children(ptr)?;

        // select action to take via PUCT
        let action = pick_action(searcher, ptr, node);

        let first_child_ptr = { *node.actions() };
        let child_ptr = first_child_ptr + action;

        let mov = tree[child_ptr].parent_move();

        pos.make_move(mov);

        tree[child_ptr].inc_threads();

        // acquire lock to avoid issues with desynced setting of
        // game state between threads when threads > 1
        let lock = if tree[child_ptr].visits() == 0 {
            Some(node.actions_mut())
        } else {
            None
        };

        // descend further
        let maybe_u = perform_one(searcher, pos, child_ptr, depth);

        drop(lock);

        tree[child_ptr].dec_threads();

        let u = maybe_u?;

        tree.propogate_proven_mates(ptr, tree[child_ptr].state());

        u
    };

    // node scores are stored from the perspective
    // **of the parent**, as they are usually only
    // accessed from the parent's POV
    u = 1.0 - u;
    node.update(u);

    Some(u)
}

fn get_utility(searcher: &Searcher, ptr: NodePtr, pos: &ChessState, hash: u64) -> f32 {
    match searcher.tree[ptr].state() {
        GameState::Ongoing => {
            let q = pos.get_value_wdl(searcher.value, searcher.params);
            searcher.tree.push_hash(hash, q);
            q
        },
        GameState::Draw => 0.5,
        GameState::Lost(_) => 0.0,
        GameState::Won(_) => 1.0,
    }
}

fn pick_action(searcher: &Searcher, ptr: NodePtr, node: &Node) -> usize {
    let is_root = ptr == searcher.tree.root_node();

    let cpuct = SearchHelpers::get_cpuct(searcher.params, node, is_root);
    let fpu = SearchHelpers::get_fpu(node);
    let expl_scale = SearchHelpers::get_explore_scaling(searcher.params, node);

    let expl = cpuct * expl_scale;

    searcher.tree.get_best_child_by_key(ptr, |child| {
        let mut q = SearchHelpers::get_action_value(child, fpu);

        // virtual loss
        let threads = f64::from(child.threads());
        if threads > 0.0 {
            let visits = f64::from(child.visits());
            let q2 = f64::from(q) * visits / (visits + threads);
            q = q2 as f32;
        }

        let u = expl * child.policy() / (1 + child.visits()) as f32;

        q + u
    })
}
