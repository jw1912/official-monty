mod linear_comb;
mod bitboard_layer;
mod softmax;

use std::io::Write;

use goober::{activation::{Identity, ReLU}, layer::DenseConnected, FeedForwardNetwork, Matrix, OutputLayer, Vector};
use linear_comb::LinearComb;
use montyformat::chess::{Piece, Position};
use bitboard_layer::{BitboardLayer, BitboardLayerLayers};
use softmax::Softmax;

use crate::rand::Rand;


pub const TOKENS: usize = 12;
const DK: usize = 32;
const DV: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
struct OutputHead {
    l1: DenseConnected<ReLU, {DV * TOKENS}, 16>,
    l2: DenseConnected<Identity, 16, 1>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Network {
    wq: [BitboardLayer<DK>; TOKENS],
    wk: [BitboardLayer<DK>; TOKENS],
    wv: [BitboardLayer<DV>; TOKENS],
    out: OutputHead,
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
        for sq in 0..TOKENS {
            self.wq[sq].adam(&grad.wq[sq], &mut momentum.wq[sq], &mut velocity.wq[sq], adj, lr);
            self.wv[sq].adam(&grad.wv[sq], &mut momentum.wv[sq], &mut velocity.wv[sq], adj, lr);
            self.wk[sq].adam(&grad.wk[sq], &mut momentum.wk[sq], &mut velocity.wk[sq], adj, lr);
        }

        self.out.adam(&grad.out, &mut momentum.out, &mut velocity.out, adj, lr);
    }

    pub fn update_single_grad(&self, (pos, mut target): &(Position, f32), grad: &mut Self, error: &mut f32, print: bool) {
        let mut bitboards = [[0; 4]; TOKENS];

        if pos.stm() > 0 {
            target = 1.0 - target;
        }

        let threats = pos.threats_by(1 - pos.stm());
        let defences = pos.threats_by(pos.stm());

        let flip = if pos.stm() > 0 { 56 } else { 0 };

        for (stm, &side) in [pos.stm(), 1 - pos.stm()].iter().enumerate() {
            for piece in Piece::PAWN..=Piece::KING {
                let mut input_bbs = [0; 4];

                let mut bb = pos.piece(side) & pos.piece(piece);
                while bb > 0 {
                    let sq = bb.trailing_zeros() as usize;

                    let bit = 1 << sq;
                    let state = usize::from(bit & threats > 0) + 2 * usize::from(bit & defences > 0);

                    input_bbs[state] ^= 1 << (sq ^ flip);

                    bb &= bb - 1;
                }

                bitboards[6 * stm + piece - 2] = input_bbs;
            }
        }

        let mut logits = [[0.0; TOKENS]; TOKENS];

        let mut queries = [BitboardLayerLayers::zeroed(); TOKENS];
        let mut keys = [BitboardLayerLayers::zeroed(); TOKENS];
        let mut values = [BitboardLayerLayers::zeroed(); TOKENS];

        for pc in 0..TOKENS {
            queries[pc] = self.wq[pc].out_with_layers(&bitboards[pc]);
            keys[pc] = self.wk[pc].out_with_layers(&bitboards[pc]);
            values[pc] = self.wv[pc].out_with_layers(&bitboards[pc]);
        }

        let mut heads = [Vector::zeroed(); TOKENS];
        let mut concat = Vector::zeroed();

        for pc1 in 0..TOKENS {
            let mut max = 0f32;            

            for (pc2, key) in keys.iter().enumerate() {
                logits[pc1][pc2] = queries[pc1].output_layer().dot(&key.output_layer());
                max = max.max(logits[pc1][pc2]);
            }

            let mut total = 0.0;

            for pc2 in 0..TOKENS {
                logits[pc1][pc2] = (logits[pc1][pc2] - max).exp();
                total += logits[pc1][pc2];
            }

            for pc2 in 0..TOKENS {
                logits[pc1][pc2] /= total;
            }

            let head = LinearComb::fwd(&logits[pc1], &values);

            for j in 0..DV {
                concat[pc1 * DV + j] = head[j];
            }

            heads[pc1] = head;
        }

        let activated = concat.activate::<ReLU>();
        let out = self.out.out_with_layers(&activated);

        if print {
            println!("EVAL: {}", 400.0 * out.output_layer()[0]);
        }

        let predicted = 1.0 / (1.0 + (-out.output_layer()[0]).exp());
        let grd = (predicted - target) * predicted * (1.0 - predicted);

        *error += (predicted - target).powi(2);

        let activated_err = self.out.backprop(&activated, &mut grad.out, Vector::from_raw([grd]), &out);
        let concat_err = activated_err * concat.derivative::<ReLU>();

        for pc1 in 0..TOKENS {
            let mut head_err = Vector::zeroed();
            for j in 0..DV {
                head_err[j] = concat_err[pc1 * DV + j];
            }

            let logit_sum_err = LinearComb::backprop(&values, head_err);

            let sm_err = Softmax::backprop(&logits[pc1], &logit_sum_err);

            for pc2 in 0..TOKENS {
                let this_err = sm_err[pc2];

                self.wq[pc1].backprop(&bitboards[pc1], &mut grad.wq[pc1], this_err * keys[pc2].output_layer(), &queries[pc1]);
                self.wk[pc2].backprop(&bitboards[pc2], &mut grad.wk[pc2], this_err * queries[pc1].output_layer(), &keys[pc2]);
                self.wv[pc2].backprop(&bitboards[pc2], &mut grad.wv[pc2], logits[pc1][pc2] * head_err, &values[pc2]);
            }
        }
    }

    pub fn rand_init() -> Box<Self> {
        let mut net = Self::boxed_and_zeroed();

        let mut rng = Rand::with_seed();
        let max = 0.2;
        
        for sq in 0..TOKENS {
            net.wq[sq] = BitboardLayer::from_fn(|| rng.rand_f32(max));
            net.wv[sq] = BitboardLayer::from_fn(|| rng.rand_f32(max));
            net.wk[sq] = BitboardLayer::from_fn(|| rng.rand_f32(max));
        }

        net.out.l1 = DenseConnected::from_raw(
            Matrix::from_fn(|_, _| rng.rand_f32(max)),
            Vector::from_fn(|_| rng.rand_f32(max)),
        );

        net.out.l2 = DenseConnected::from_raw(
            Matrix::from_fn(|_, _| rng.rand_f32(max)),
            Vector::from_fn(|_| rng.rand_f32(max)),
        );

        net
    }

    pub fn add_without_explicit_lifetime(&mut self, rhs: &Self) {
        for sq in 0..TOKENS {
            self.wq[sq] += &rhs.wq[sq];
            self.wv[sq] += &rhs.wv[sq];
            self.wk[sq] += &rhs.wk[sq];
        }

        self.out += &rhs.out;
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