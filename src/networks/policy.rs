mod align;

#[cfg(not(target_feature = "avx2"))]
mod autovec;
use std::mem::MaybeUninit;

#[cfg(not(target_feature = "avx2"))]
use autovec as backend;
#[cfg(target_feature = "avx2")]
mod avx2;
#[cfg(target_feature = "avx2")]
use avx2 as backend;

use align::Align64;

use crate::chess::{consts::Side, Board, Castling, Move};

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const PolicyFileDefaultName: &str = "attnq.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedPolicyName: &str = "nn-4b70c6924179.network";

const QA: i16 = 255;
const QB: i16 = 64;

const L1: usize = 2048;
const DIM: usize = 32;
const SHIFT: usize = 10;

#[repr(C, align(64))]
pub struct PolicyNetwork {
    src1w: [Align64<i16, L1>; 768 * 4],
    src1b: Align64<i16, L1>,
    dst1w: [Align64<i16, L1>; 768 * 4],
    dst1b: Align64<i16, L1>,
    src2w: [Align64<i8, {1024 * DIM}>; 64],
    src2b: [Align64<i16, DIM>; 64],
    dst2w: [Align64<i8, {1024 * DIM}>; 128],
    dst2b: [Align64<i16, DIM>; 128],
}

impl PolicyNetwork {
    pub fn map_moves_with_policies<F: FnMut(Move, f32)>(&self, pos: &Board, castling: &Castling, mut f: F) {
        let mut feats = [0usize; 256];
        let mut count = 0;
        pos.map_features(|feat| {
            feats[count] = feat;
            count += 1;
        });

        let mut sl1 = self.src1b;
        sl1.add_multi(&feats[..count], &self.src1w);

        let mut dl1 = self.dst1b;
        dl1.add_multi(&feats[..count], &self.dst1w);

        let mut l1 = [Align64([0; L1 / 2]); 2];

        for (acc, hl) in l1.iter_mut().zip([&sl1, &dl1]) {
            for i in 0..L1 / 2 {
                let l = hl.0[i].clamp(0, QA);
                let r = hl.0[i + L1 / 2].clamp(0, QA);
                let pw = i32::from(l) * i32::from(r);

                // QA * QA / (1 << SHIFT)
                acc.0[i] = (pw >> SHIFT) as u8;
            }
        }

        let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };
        let flip = hm ^ if pos.stm() == Side::BLACK { 56 } else { 0 };

        let mut cache = [MaybeUninit::uninit(); 16];
        let mut count = 0;
        let mut hits = [16; 64];

        pos.map_legal_moves(castling, |mov| {
            let src_idx = usize::from(mov.src() ^ flip);
            let dst_idx = usize::from(mov.to() ^ flip) + 64 * usize::from(pos.see(&mov, -108));

            // QA * QA * QB / (1 << SHIFT)
            if hits[src_idx] == 16 {
                cache[count].write(unsafe { backend::l2(&self.src2w[src_idx], &l1[0]) });
                hits[src_idx] = count;
                count += 1;
            }
            
            let src_vec = unsafe { cache[hits[src_idx]].assume_init_ref() };
            let dst_vec = unsafe { backend::l2(&self.dst2w[dst_idx], &l1[1]) };

            let mut res = 0;

            for i in 0..DIM {
                let l = (src_vec.0[i] << SHIFT) / i32::from(QA) + i32::from(self.src2b[src_idx].0[i]);
                let r = (dst_vec.0[i] << SHIFT) / i32::from(QA) + i32::from(self.dst2b[dst_idx].0[i]);
                res += l * r;
            }

            f(mov, (res / i32::from(QB * QB)) as f32 / f32::from(QA).powi(2))
        });
    }
}


#[repr(C)]
#[derive(Clone, Copy)]
pub struct UnqPolicyNetwork {
    src1w: [[i16; L1]; 3072],
    src1b: [i16; L1],
    dst1w: [[i16; L1]; 3072],
    dst1b: [i16; L1],
    src2w: [[[i8; DIM]; 64]; L1 / 2],
    dst2w: [[[i8; DIM]; 128]; L1 / 2],
    src2b: [[i16; DIM]; 64],
    dst2b: [[i16; DIM]; 128],
}

impl UnqPolicyNetwork {
    pub fn quantise(&self) -> Box<PolicyNetwork> {
        let mut res = unsafe { crate::boxed_and_zeroed::<PolicyNetwork>() };

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            res.src1w = std::mem::transmute(self.src1w);
            res.dst1w = std::mem::transmute(self.dst1w);
            res.src1b = std::mem::transmute(self.src1b);
            res.dst1b = std::mem::transmute(self.dst1b);
            res.src2b = std::mem::transmute(self.src2b);
            res.dst2b = std::mem::transmute(self.dst2b);
        }

        for idx in 0..64 {
            for i in 0..L1 / 8 {
                for j in 0..DIM {
                    for k in 0..4 {
                        let lidx = 4 * (DIM * i + j) + k;
                        let n = 4 * i + k;

                        res.src2w[idx].0[lidx] = self.src2w[n][idx][j];
                        res.dst2w[idx].0[lidx] = self.dst2w[n][idx][j];
                        res.dst2w[idx + 64].0[lidx] = self.dst2w[n][idx + 64][j];
                    }
                }
            }
        }

        res
    }
}