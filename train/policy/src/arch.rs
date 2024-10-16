use datagen::{PolicyData, Rand};
use goober::{activation, layer, FeedForwardNetwork, Matrix, OutputLayer, SparseVector, Vector};
use monty::{Board, Move};

use std::io::Write;

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: layer::SparseConnected<activation::ReLU, 768, 16>,
}

impl SubNet {
    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let matrix = Matrix::from_fn(|_, _| f());
        let vector = Vector::from_fn(|_| f());

        Self {
            ft: layer::SparseConnected::from_raw(matrix, vector),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub subnets: [SubNet; 128],
    pub good_see_subnet: SubNet,
    pub hce: layer::DenseConnected<activation::Identity, 4, 1>,
}

impl PolicyNetwork {
    pub fn get_hce_feats(_: &Board, mov: &Move) -> Vector<4> {
        let mut feats = Vector::zeroed();

        if mov.is_promo() {
            feats[mov.promo_pc() - 3] = 1.0;
        }

        feats
    }

    pub fn update(
        policy: &mut Self,
        grad: &Self,
        adj: f32,
        lr: f32,
        momentum: &mut Self,
        velocity: &mut Self,
    ) {
        for (i, subnet) in policy.subnets.iter_mut().enumerate() {
            subnet.adam(
                &grad.subnets[i],
                &mut momentum.subnets[i],
                &mut velocity.subnets[i],
                adj,
                lr,
            );
        }

        policy.good_see_subnet.adam(
            &grad.good_see_subnet,
            &mut momentum.good_see_subnet,
            &mut velocity.good_see_subnet,
            adj,
            lr,
        );

        policy
            .hce
            .adam(&grad.hce, &mut momentum.hce, &mut velocity.hce, adj, lr);
    }

    pub fn update_single_grad(pos: &PolicyData, policy: &Self, grad: &mut Self, error: &mut f32) {
        let board = Board::from(pos.pos);

        let mut feats = SparseVector::with_capacity(32);
        board.map_policy_features(|feat| feats.push(feat));

        let mut policies = Vec::with_capacity(pos.num);
        let mut total = 0.0;
        let mut total_visits = 0;
        let mut max = -1000.0;

        let flip = board.flip_val();

        let good_see_out = policy.good_see_subnet.out_with_layers(&feats);

        for &(mov, visits) in &pos.moves[..pos.num] {
            let mov = <Move as From<u16>>::from(mov);

            let from = usize::from(mov.src() ^ flip);
            let to = usize::from(mov.to() ^ flip) + 64;
            let good_see = usize::from(board.see(&mov, -108));

            let from_out = policy.subnets[from].out_with_layers(&feats);
            let to_out = policy.subnets[to].out_with_layers(&feats);

            let to_output = if good_see > 0 {
                to_out.output_layer() + good_see_out.output_layer()
            } else {
                to_out.output_layer()
            };

            let hce_feats = PolicyNetwork::get_hce_feats(&board, &mov);
            let hce_out = policy.hce.out_with_layers(&hce_feats);
            let score =
                from_out.output_layer().dot(&to_output) + hce_out.output_layer()[0];

            if score > max {
                max = score;
            }

            total_visits += visits;
            policies.push((from_out, to_out, hce_out, mov, visits, score, good_see));
        }

        for (_, _, _, _, _, score, _) in policies.iter_mut() {
            *score = (*score - max).exp();
            total += *score;
        }

        for (from_out, to_out, hce_out, mov, visits, score, good_see) in policies {
            let from = usize::from(mov.src() ^ flip);
            let to = usize::from(mov.to() ^ flip) + 64;
            let hce_feats = PolicyNetwork::get_hce_feats(&board, &mov);

            let ratio = score / total;

            let expected = visits as f32 / total_visits as f32;
            let err = ratio - expected;

            *error -= expected * ratio.ln();

            let factor = err;

            let to_output = if good_see > 0 {
                to_out.output_layer() + good_see_out.output_layer()
            } else {
                to_out.output_layer()
            };

            policy.subnets[from].backprop(
                &feats,
                &mut grad.subnets[from],
                factor * to_output,
                &from_out,
            );

            let from_grad = factor * from_out.output_layer();

            policy.subnets[to].backprop(
                &feats,
                &mut grad.subnets[to],
                from_grad,
                &to_out,
            );

            if good_see > 0 {
                policy.good_see_subnet.backprop(
                    &feats,
                    &mut grad.good_see_subnet,
                    from_grad,
                    &to_out,
                );
            }

            policy.hce.backprop(
                &hce_feats,
                &mut grad.hce,
                Vector::from_raw([factor]),
                &hce_out,
            );
        }
    }

    pub fn rand_init() -> Box<Self> {
        let mut policy = Self::boxed_and_zeroed();

        let mut rng = Rand::with_seed();
        for subnet in policy.subnets.iter_mut() {
            *subnet = SubNet::from_fn(|| rng.rand_f32(0.2));
        }

        policy.good_see_subnet = SubNet::from_fn(|| rng.rand_f32(0.2));

        policy
    }

    pub fn add_without_explicit_lifetime(&mut self, rhs: &Self) {
        for (i, j) in self.subnets.iter_mut().zip(rhs.subnets.iter()) {
            *i += j;
        }

        self.good_see_subnet += &rhs.good_see_subnet;

        self.hce += &rhs.hce;
    }

    pub fn boxed_and_zeroed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn write_to_bin(&self, path: &str) {
        let size_of = std::mem::size_of::<Self>();

        let mut file = std::fs::File::create(path).unwrap();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, size_of);
            file.write_all(slice).unwrap();
        }
    }
}
