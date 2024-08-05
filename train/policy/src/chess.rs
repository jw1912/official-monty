use datagen::{PolicyData, Rand};
use goober::{FeedForwardNetwork, OutputLayer, SparseVector, Vector};
use monty::{Board, Move, PolicyNetwork, SubNet};

use crate::TrainablePolicy;

impl TrainablePolicy for PolicyNetwork {
    type Data = PolicyData;

    fn update(
        policy: &mut Self,
        grad: &Self,
        adj: f32,
        lr: f32,
        momentum: &mut Self,
        velocity: &mut Self,
    ) {
        for (i, subnet) in policy.from_subnets.iter_mut().enumerate() {
            subnet.adam(
                &grad.from_subnets[i],
                &mut momentum.from_subnets[i],
                &mut velocity.from_subnets[i],
                adj,
                lr,
            );
        }

        for (i, subnet) in policy.to_subnets.iter_mut().enumerate() {
            subnet.adam(
                &grad.to_subnets[i],
                &mut momentum.to_subnets[i],
                &mut velocity.to_subnets[i],
                adj,
                lr,
            );
        }

        policy
            .hce
            .adam(&grad.hce, &mut momentum.hce, &mut velocity.hce, adj, lr);
    }

    fn update_single_grad(pos: &Self::Data, policy: &Self, grad: &mut Self, error: &mut f32) {
        let board = Board::from(pos.pos);

        let mut feats = SparseVector::with_capacity(32);
        board.map_policy_features(|feat| feats.push(feat));

        let mut policies = Vec::with_capacity(pos.num);
        let mut total = 0.0;
        let mut total_visits = 0;
        let mut max = -1000.0;

        let flip = board.flip_val();

        for &(mov, visits) in &pos.moves[..pos.num] {
            let mov = <Move as From<u16>>::from(mov);

            let from = usize::from(mov.src() ^ flip);
            let to = usize::from(mov.to() ^ flip);

            let from_out = policy.from_subnets[from].out_with_layers(&feats);
            let to_out = policy.to_subnets[to].out_with_layers(&feats);
            let hce_feats = PolicyNetwork::get_hce_feats(&board, &mov);
            let hce_out = policy.hce.out_with_layers(&hce_feats);
            let score =
                from_out.output_layer().dot(&to_out.output_layer()) + hce_out.output_layer()[0];

            if score > max {
                max = score;
            }

            total_visits += visits;
            policies.push((from_out, to_out, hce_out, mov, visits, score));
        }

        for (_, _, _, _, _, score) in policies.iter_mut() {
            *score = (*score - max).exp();
            total += *score;
        }

        for (from_out, to_out, hce_out, mov, visits, score) in policies {
            let from = usize::from(mov.src() ^ flip);
            let to = usize::from(mov.to() ^ flip);
            let hce_feats = PolicyNetwork::get_hce_feats(&board, &mov);

            let ratio = score / total;

            let expected = visits as f32 / total_visits as f32;
            let err = ratio - expected;

            *error -= expected * ratio.ln();

            let factor = err;

            policy.from_subnets[from].backprop(
                &feats,
                &mut grad.from_subnets[from],
                factor * to_out.output_layer(),
                &from_out,
            );

            policy.to_subnets[to].backprop(
                &feats,
                &mut grad.to_subnets[to],
                factor * from_out.output_layer(),
                &to_out,
            );

            policy.hce.backprop(
                &hce_feats,
                &mut grad.hce,
                Vector::from_raw([factor]),
                &hce_out,
            );
        }
    }

    fn rand_init() -> Box<Self> {
        let mut policy = Self::boxed_and_zeroed();

        let mut rng = Rand::with_seed();

        for subnet in policy.from_subnets.iter_mut() {
            *subnet = SubNet::from_fn(|| rng.rand_f32(0.2));
        }

        for subnet in policy.to_subnets.iter_mut() {
            *subnet = SubNet::from_fn(|| rng.rand_f32(0.2));
        }

        policy
    }

    fn add_without_explicit_lifetime(&mut self, rhs: &Self) {
        for (i, j) in self.from_subnets.iter_mut().zip(rhs.from_subnets.iter()) {
            *i += j;
        }

        for (i, j) in self.to_subnets.iter_mut().zip(rhs.to_subnets.iter()) {
            *i += j;
        }

        self.hce += &rhs.hce;
    }
}
