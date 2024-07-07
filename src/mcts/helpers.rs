use crate::{mcts::MctsParams, tree::Edge};

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

        // increase CPUCT if the outcome seems certain
        if parent.visits() > 0 {
            let q = (parent.q() - 0.5).abs();

            if q >= params.cpuct_desperation_cutoff() {
                let prior = 32.0 * params.cpuct_desperation_prior();
                let p = parent.visits() as f32;

                cpuct *= 1.0 + params.cpuct_desperation_multiplier() * p / (prior + p);
            }
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
}
