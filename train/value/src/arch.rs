mod linear_comb;
mod one_hot_layer;
mod sparse_softmax;

use std::io::Write;

use goober::{activation::{Identity, ReLU}, layer::DenseConnected, FeedForwardNetwork, Matrix, OutputLayer, Vector};
use linear_comb::LinearComb;
use montyformat::chess::{Piece, Position};
use one_hot_layer::OneHotLayer;
use sparse_softmax::SparseSoftmax;

use crate::rand::Rand;

const DK: usize = 32;
const DV: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Network {
    wq: [OneHotLayer<12, DK>; 64],
    wk: [OneHotLayer<12, DK>; 64],
    wv: [OneHotLayer<12, DV>; 64],
    l1: DenseConnected<ReLU, {DV * 64}, 16>,
    l2: DenseConnected<Identity, 16, 1>,
}

impl Network {
    pub fn update(
        &mut self,
        grad: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        adj: f32,
        lr: f32,
    ) {
        for sq in 0..64 {
            self.wq[sq].adam(&grad.wq[sq], &mut momentum.wq[sq], &mut velocity.wq[sq], adj, lr);
            self.wv[sq].adam(&grad.wv[sq], &mut momentum.wv[sq], &mut velocity.wv[sq], adj, lr);
            self.wk[sq].adam(&grad.wk[sq], &mut momentum.wk[sq], &mut velocity.wk[sq], adj, lr);
        }

        self.l1.adam(&grad.l1, &mut momentum.l1, &mut velocity.l1, adj, lr);
        self.l2.adam(&grad.l2, &mut momentum.l2, &mut velocity.l2, adj, lr);
    }

    pub fn update_single_grad(&self, (pos, mut target): &(Position, f32), grad: &mut Self, error: &mut f32) {
        let mut active = Vec::new();

        if pos.stm() > 0 {
            target = 1.0 - target;
        }

        let flip = if pos.stm() > 0 { 56 } else { 0 };

        for (stm, &side) in [pos.stm(), 1 - pos.stm()].iter().enumerate() {
            for piece in Piece::PAWN..=Piece::KING {
                let mut bb = pos.piece(side) & pos.piece(piece);

                while bb > 0 {
                    let sq = bb.trailing_zeros() as usize;

                    active.push((sq ^ flip, 6 * stm + piece - 2));

                    bb &= bb - 1;
                }
            }
        }

        let mut logits = [[0.0; 64]; 64];

        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for &(sq, pc) in &active {
            queries.push(self.wq[sq].out_with_layers(&pc));
            keys.push(self.wk[sq].out_with_layers(&pc));
            values.push(self.wv[sq].out_with_layers(&pc));
        }

        let mut heads = Vec::new();
        let mut concat = Vector::zeroed();

        for (i, &(sq1, _)) in active.iter().enumerate() {
            let mut max = 0f32;            

            for (j, &(sq2, _)) in active.iter().enumerate() {
                logits[sq1][sq2] = queries[i].output_layer().dot(&keys[j].output_layer());
                max = max.max(logits[sq1][sq2]);
            }

            let mut total = 0.0;

            for &(sq2, _) in &active {
                logits[sq1][sq2] = (logits[sq1][sq2] - max).exp();
                total += logits[sq1][sq2];
            }

            for &(sq2, _) in &active {
                logits[sq1][sq2] /= total;
            }

            let head = LinearComb::fwd(&active, &logits[sq1], &values);

            for j in 0..DV {
                concat[sq1 * DV + j] = head[j];
            }

            heads.push(head);
        }

        let activated = concat.activate::<ReLU>();

        let l1 = self.l1.out_with_layers(&activated);
        let l2 = self.l2.out_with_layers(&l1.output_layer());

        let predicted = 1.0 / (1.0 + (-l2.output_layer()[0]).exp());
        let grd = (predicted - target) * predicted * (1.0 - predicted);

        *error += (predicted - target).powi(2);

        let l2_err = self.l2.backprop(&l1.output_layer(), &mut grad.l2, Vector::from_raw([grd]), &l2);
        let l1_err = self.l1.backprop(&activated, &mut grad.l1, l2_err, &l1);
        let concat_err = l1_err * concat.derivative::<ReLU>();

        for (i, &(sq1, pc1)) in active.iter().enumerate() {
            let mut head_err = Vector::zeroed();
            for j in 0..DV {
                head_err[j] = concat_err[sq1 * DV + j];
            }

            let logit_sum_err = LinearComb::backprop(&active, &values, head_err);
            let sm_err = SparseSoftmax::backprop(&active, &logits[sq1], &logit_sum_err);

            for (j, &(sq2, pc2)) in active.iter().enumerate() {
                let this_err = sm_err[sq2];

                self.wq[sq1].backprop(&pc1, &mut grad.wq[sq1], this_err * keys[j].output_layer(), &queries[i]);
                self.wk[sq2].backprop(&pc2, &mut grad.wk[sq2], this_err * queries[i].output_layer(), &keys[j]);
                self.wv[sq2].backprop(&pc2, &mut grad.wv[sq2], logits[sq1][sq2] * head_err, &values[j]);
            }
        }
    }

    pub fn rand_init() -> Box<Self> {
        let mut net = Self::boxed_and_zeroed();

        let mut rng = Rand::with_seed();
        let max = 0.2;
        
        for sq in 0..64 {
            net.wq[sq] = OneHotLayer::from_fn(|| rng.rand_f32(max));
            net.wv[sq] = OneHotLayer::from_fn(|| rng.rand_f32(max));
            net.wk[sq] = OneHotLayer::from_fn(|| rng.rand_f32(max));
        }

        net.l1 = DenseConnected::from_raw(
            Matrix::from_fn(|_, _| rng.rand_f32(max)),
            Vector::from_fn(|_| rng.rand_f32(max)),
        );

        net.l2 = DenseConnected::from_raw(
            Matrix::from_fn(|_, _| rng.rand_f32(max)),
            Vector::from_fn(|_| rng.rand_f32(max)),
        );

        net
    }

    pub fn add_without_explicit_lifetime(&mut self, rhs: &Self) {
        for sq in 0..64 {
            self.wq[sq] += &rhs.wq[sq];
            self.wv[sq] += &rhs.wv[sq];
            self.wk[sq] += &rhs.wk[sq];
        }

        self.l1 += &rhs.l1;
        self.l2 += &rhs.l2;
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