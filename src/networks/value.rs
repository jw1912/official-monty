use crate::{boxed_and_zeroed, Board, Piece};

use super::{
    accumulator::Accumulator, activation::ReLU, layer::Layer
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-232535a63f1f.network";

const SCALE: i32 = 400;

const TOKENS: usize = 12;
const DI: usize = 256;
const DK: usize = 32;
const DV: usize = 8;
const D1: usize = 16;

#[repr(C)]
pub struct ValueNetwork {
    wq: [[Accumulator<f32, DK>; DI]; TOKENS],
    wk: [[Accumulator<f32, DK>; DI]; TOKENS],
    wv: [[Accumulator<f32, DV>; DI]; TOKENS],
    l1: Layer<f32, {TOKENS * DV}, D1>,
    l2: Layer<f32, D1, 1>,
}

impl ValueNetwork {
    pub fn eval(&self, pos: &Board) -> i32 {
        let mut bitboards = [[0; 4]; TOKENS];

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

        let mut queries = [[0.0; DK]; TOKENS];
        let mut keys = [[0.0; DK]; TOKENS];
        let mut values = [[0.0; DV]; TOKENS];

        for i in 0..TOKENS {
            embed_into(&bitboards[i], &self.wq[i], &mut queries[i]);
            embed_into(&bitboards[i], &self.wk[i], &mut keys[i]);
            embed_into(&bitboards[i], &self.wv[i], &mut values[i]);
        }
    
        let mut hl = Accumulator([0.0; TOKENS * DV]);
    
        for (i, query) in queries.iter().enumerate() {
            let mut temps = [0.0; TOKENS];
            let mut max = f32::NEG_INFINITY;
    
            for j in 0..TOKENS{
                let key = keys[j];
    
                for k in 0..DK {
                    temps[j] += query[k] * key[k]
                }

                temps[j] /= (DK as f32).sqrt();
    
                max = max.max(temps[j]);
            }
    
            let mut total = 0.0;
    
            for t in temps.iter_mut() {
                *t = (*t - max).exp();
                total += *t;
            }
    
            for j in 0..TOKENS {
                temps[j] /= total;
    
                let value = &values[j];
                let weight = temps[j];
        
                for (k, &val) in value.iter().enumerate() {
                    hl.0[i * DV + k] += weight * val;
                }
            }
        }

        let l1 = self.l1.forward::<ReLU>(&hl);
        let l2 = self.l2.forward::<ReLU>(&l1);

        (l2.0[0] * SCALE as f32) as i32
    }
}

fn embed_into<const N: usize>(input: &[u64; 4], weights: &[Accumulator<f32, N>; 256], out: &mut [f32; N]) {
    for (i, &(mut bb)) in input.iter().enumerate() {
        while bb > 0 {
            let sq = bb.trailing_zeros() as usize;

            let w = &weights[64 * i + sq];
            for (o, &w) in out.iter_mut().zip(w.0.iter()) {
                *o += w;
            }

            bb &= bb - 1;
        }
    }
}

#[repr(C)]
pub struct UnquantisedValueNetwork {
    l1: Layer<f32, { 768 * 4 }, 4096>,
    l2: Layer<f32, 4096, 16>,
    l3: Layer<f32, 16, 128>,
    l4: Layer<f32, 128, 1>,
}

impl UnquantisedValueNetwork {
    pub fn quantise(&self) -> Box<ValueNetwork> {
        unsafe { boxed_and_zeroed() }
    }
}
