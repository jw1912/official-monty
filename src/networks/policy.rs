use crate::chess::{
    consts::{Flag, Piece, Side},
    Attacks, Board, Castling, Move,
};

use super::{accumulator::Accumulator, layer::Layer};

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
    l2: [Accumulator<i8, L1>; 1880 * 2],
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

    pub fn get(
        &self,
        pos: &Board,
        castling: &Castling,
        mov: Move,
        hl: &Accumulator<i16, L1>,
    ) -> f32 {
        let weights = &self.l2[map_move_to_index(pos, mov)];

        let diff = get_diff(pos, castling, mov);

        let sub1 = diff[0];
        let sub2 = diff[1];
        let add1 = diff[2];
        let add2 = diff[3];

        assert_ne!(sub1, -1);
        assert_ne!(add1, -1);

        let sub1w = &self.l1.weights[768 + sub1 as usize];
        let add1w = &self.l1.weights[768 + add1 as usize];

        let mut res = 0;

        match (sub2, add2) {
            (-1, -1) => {
                for i in 0..L1 {
                    let v = hl.0[i] - i16::from(sub1w.0[i]) + i16::from(add1w.0[i]);
                    res += i32::from(weights.0[i]) * i32::from(v.clamp(0, QA).pow(2));
                }
            }
            (-1, x) => {
                let add2w = &self.l1.weights[768 + x as usize];

                for i in 0..L1 {
                    let v = hl.0[i] - i16::from(sub1w.0[i])
                        + i16::from(add1w.0[i])
                        + i16::from(add2w.0[i]);
                    res += i32::from(weights.0[i]) * i32::from(v.clamp(0, QA).pow(2));
                }
            }
            (x, -1) => {
                let sub2w = &self.l1.weights[768 + x as usize];

                for i in 0..L1 {
                    let v = hl.0[i] - i16::from(sub1w.0[i]) + i16::from(add1w.0[i])
                        - i16::from(sub2w.0[i]);
                    res += i32::from(weights.0[i]) * i32::from(v.clamp(0, QA).pow(2));
                }
            }
            (x, y) => {
                let sub2w = &self.l1.weights[768 + x as usize];
                let add2w = &self.l1.weights[768 + y as usize];

                for i in 0..L1 {
                    let v = hl.0[i] - i16::from(sub1w.0[i]) + i16::from(add1w.0[i])
                        - i16::from(sub2w.0[i])
                        + i16::from(add2w.0[i]);
                    res += i32::from(weights.0[i]) * i32::from(v.clamp(0, QA).pow(2));
                }
            }
        }

        (res / i32::from(QA)) as f32 / (f32::from(QA) * f32::from(QB))
    }
}

fn get_diff(pos: &Board, castling: &Castling, mov: Move) -> [i32; 4] {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;
    let idx = |stm, pc, sq| ([0, 384][stm] + 64 * (pc - 2) + (sq ^ flip)) as i32;

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

        diff[1] = idx(
            0,
            Piece::ROOK,
            sf + castling.rook_file(pos.stm(), ks) as usize,
        );
        diff[3] = idx(0, Piece::ROOK, sf + [3, 5][ks]);
    }

    for i in diff {
        assert!(i < 768);
        assert!(i >= -1);
    }

    diff
}

const PROMOS: usize = 4 * 22;

fn map_move_to_index(pos: &Board, mov: Move) -> usize {
    let good_see = (OFFSETS[64] + PROMOS) * usize::from(pos.see(&mov, -108));

    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;

    let idx = if mov.is_promo() {
        let ffile = (mov.src() ^ flip) % 8;
        let tfile = (mov.to() ^ flip) % 8;
        let promo_id = 2 * ffile + tfile;

        OFFSETS[64] + 22 * (mov.promo_pc() - 3) + usize::from(promo_id)
    } else {
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
