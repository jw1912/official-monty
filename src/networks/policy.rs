use crate::{
    boxed_and_zeroed,
    chess::{Attacks, Board, Move},
};

use super::{
    accumulator::Accumulator, activation::SCReLU, layer::{Layer, TransposedLayer}
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-3031ab470125.network";

const QA: i16 = 256;
const QB: i16 = 512;
const FACTOR: i16 = 32;

const L1: usize = 4096;
const L2: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    l1: Layer<i16, { 768 * 4 }, L1>,
    l2: TransposedLayer<i16, L1, L2>,
    l3: TransposedLayer<f32, L2, { 1880 * 2 }>,
}

impl PolicyNetwork {
    pub fn hl(&self, pos: &Board) -> Accumulator<f32, L2> {
        let mut l1 = self.l1.biases;

        pos.map_policy_features(|feat| l1.add(&self.l1.weights[feat]));

        let mut hl = self.l2.forward_from_i16::<SCReLU, QA, QB, FACTOR>(&l1);

        for elem in &mut hl.0 {
            *elem = (*elem).clamp(0.0, 1.0).powi(2);
        }

        hl
    }

    pub fn get(&self, pos: &Board, mov: &Move, hl: &Accumulator<f32, L2>) -> f32 {
        let idx = map_move_to_index(pos, *mov);
        let weights = &self.l3.weights[idx];

        let mut res = self.l3.biases.0[idx];

        for (&w, &v) in weights.0.iter().zip(hl.0.iter()) {
            res += w * v;
        }

        res
    }
}

const PROMOS: usize = 4 * 22;

fn map_move_to_index(pos: &Board, mov: Move) -> usize {
    let good_see = (OFFSETS[64] + PROMOS) * usize::from(pos.see(&mov, -108));

    let idx = if mov.is_promo() {
        let ffile = mov.src() % 8;
        let tfile = mov.to() % 8;
        let promo_id = 2 * ffile + tfile;

        OFFSETS[64] + 22 * (mov.promo_pc() - 3) + usize::from(promo_id)
    } else {
        let flip = if pos.stm() == 1 { 56 } else { 0 };
        let from = usize::from(mov.src() ^ flip);
        let dest = usize::from(mov.to() ^ flip);

        let below = Attacks::ALL_DESTINATIONS[from] & ((1 << dest) - 1);

        OFFSETS[from] + below.count_ones() as usize
    };

    good_see + idx
}

const OFFSETS: [usize; 65] = {
    let mut offsets = [0; 65];

    let mut curr = 0;
    let mut sq = 0;

    while sq < 64 {
        offsets[sq] = curr;
        curr += Attacks::ALL_DESTINATIONS[sq].count_ones() as usize;
        sq += 1;
    }

    offsets[64] = curr;

    offsets
};

#[repr(C)]
pub struct UnquantisedPolicyNetwork {
    l1: Layer<f32, { 768 * 4 }, L1>,
    l2: Layer<f32, L1, L2>,
    l3: Layer<f32, L2, { 1880 * 2 }>,
}

impl UnquantisedPolicyNetwork {
    pub fn quantise(&self) -> Box<PolicyNetwork> {
        let mut quantised: Box<PolicyNetwork> = unsafe { boxed_and_zeroed() };

        self.l1.quantise_into_i16(&mut quantised.l1, QA, 1.98);
        self.l2
            .quantise_transpose_into_i16(&mut quantised.l2, QB, 1.98);
        self.l3.transpose_into(&mut quantised.l3);

        quantised
    }
}
