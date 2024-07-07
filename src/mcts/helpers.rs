use crate::{mcts::MctsParams, tree::{Edge, Node}};

pub struct SearchHelpers;

impl SearchHelpers {
    pub fn get_cpuct(params: &MctsParams, parent: &Edge, is_root: bool) -> f32 {
        // baseline CPUCT value
        let mut cpuct = if is_root {
            params.root_cpuct()
        } else {
            params.cpuct()
        };

        // scale CPUCT as visits increase
        let scale = params.cpuct_visits_scale() * 128.0;
        cpuct *= 1.0 + ((parent.visits() as f32 + scale) / scale).ln();

        // scale CPUCT with variance of Q
        if parent.visits() > 1 {
            let frac = parent.var().sqrt() / params.cpuct_var_scale();
            cpuct *= 1.0 + params.cpuct_var_weight() * (frac - 1.0);
        }

        cpuct
    }

    pub fn get_explore_scaling(params: &MctsParams, parent: &Edge) -> f32 {
        (params.expl_tau() * (parent.visits().max(1) as f32).ln()).exp()
    }

    pub fn get_fpu(params: &MctsParams, parent: &Edge, node: &Node) -> f32 {
        let parent_q = 1.0 - parent.q();
        let adj_weight = Self::get_visited_policy(node);
        parent_q + params.fpu_offset() + params.fpu_factor() * adj_weight
    }

    pub fn get_action_value(action: &Edge, fpu: f32) -> f32 {
        if action.visits() == 0 {
            fpu
        } else {
            action.q()
        }
    }

    pub fn get_visited_policy(node: &Node) -> f32 {
        let mut sum = 0.0;

        for action in node.actions() {
            if action.visits() > 0 {
                sum += action.policy()
            }
        }

        sum
    }
}
