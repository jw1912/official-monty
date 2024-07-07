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

    pub fn get_fpu(parent: &Edge) -> f32 {
        1.0 - parent.q()
    }

    pub fn get_action_value(action: &Edge, fpu: f32) -> f32 {
        if action.visits() == 0 {
            fpu
        } else {
            action.q()
        }
    }

    pub fn get_action_policy(params: &MctsParams, action: &Edge, boost_util: f32) -> f32 {
        if action.visits() > 0 && action.q() >= boost_util {
            action.policy().max(params.policy_boost())
        } else {
            action.policy()
        }
    }

    pub fn get_boost_q(node: &Node) -> f32 {
        const NUM_TOP: usize = 3;

        let mut top_qs = [-1.0; NUM_TOP];

        for action in node.actions() {
            for i in 0..NUM_TOP {
                if action.visits() > 0 && action.q() > top_qs[i] {
                  for j in (i + 1..NUM_TOP).rev() {
                    top_qs[j] = top_qs[j - 1];
                  }
                  top_qs[i] = action.q();
                  break;
                }
            }
        }

        let q = top_qs[NUM_TOP - 1];

        if q < 0.0 { 1.0 } else { q }
    }
}
