use crate::chess::{consts::{Flag, Piece, Side}, Board, Castling, Move};

use super::{
    accumulator::Accumulator,
    layer::{Layer, TransposedLayer},
};

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const PolicyFileDefaultName: &str = "nn-cfb555edbe8a.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedPolicyName: &str = "nn-4b70c6924179.network";

const QA: i16 = 128;
const QB: i16 = 128;

pub const L1: usize = 2560;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    l1: Layer<i8, { 768 * 2 }, L1>,
    l2: TransposedLayer<i8, L1, 1>,
}

impl PolicyNetwork {
    pub fn hl(&self, pos: &Board) -> Accumulator<i16, L1> {
        let mut l1 = Accumulator([0; L1]);

        for (r, &b) in l1.0.iter_mut().zip(self.l1.biases.0.iter()) {
            *r = i16::from(b);
        }

        let mut feats = [0usize; 256];
        let mut count = 0;
        pos.map_features(|feat| {
            feats[count] = feat;
            count += 1;
        });

        l1.add_multi_i8(&feats[..count], &self.l1.weights);

        l1
    }

    pub fn get(&self, pos: &Board, castling: &Castling, mov: Move, hl: &Accumulator<i16, L1>) -> f32 {
        let weights = &self.l2.weights[0];

        let mut thl = *hl;

        let diff = get_diff(pos, castling, mov);

        for &feat in &diff[..2] {
            if feat != -1 {
                thl.sub_i8(&self.l1.weights[768 + feat as usize]);
            }
        }

        for &feat in &diff[2..] {
            if feat != -1 {
                thl.add_i8(&self.l1.weights[768 + feat as usize]);
            }
        }

        let mut res = 0;

        for (&w, &v) in weights.0.iter().zip(thl.0.iter()) {
            res += i32::from(w) * i32::from(v.clamp(0, QA)).pow(2);
        }

        (res as f32 / f32::from(QA.pow(2)) + f32::from(self.l2.biases.0[0])) / f32::from(QB)
    }
}

fn get_diff(pos: &Board, castling: &Castling, mov: Move) -> [i32; 4] {
    let flip = |sq| {
        if pos.stm() == Side::BLACK {
            sq ^ 56
        } else {
            sq
        }
    };
    let idx = |stm, pc, sq| ([0, 384][stm] + 64 * (pc - 2) + flip(sq)) as i32;

    let mut diff = [-1; 4];

    let src = mov.src() as usize;
    let dst = mov.to() as usize;

    let moved = pos.get_pc(1 << src);
    diff[0] = idx(0, moved, src);

    if mov.is_en_passant() {
        diff[1] = idx(1, Piece::PAWN, dst ^ 8);
    } else if mov.is_capture() {
        diff[1] = idx(1, pos.get_pc(1 << dst), dst);
    }

    if mov.is_promo() {
        let promo = usize::from((mov.flag() & 3) + 3);
        diff[2] = idx(0, promo, dst);
    } else {
        diff[2] = idx(0, moved, dst);
    }

    if mov.flag() == Flag::KS || mov.flag() == Flag::QS {
        assert_eq!(diff[1], -1);

        let ks = usize::from(mov.flag() == Flag::KS);
        let sf = 56 * pos.stm();

        diff[1] = idx(0, Piece::ROOK, sf + castling.rook_file(pos.stm(), ks) as usize);
        diff[3] = idx(0, Piece::ROOK, sf + [3, 5][ks]);
    }

    for i in diff {
        assert!(i < 768);
        assert!(i >= -1);
    }

    diff
}
