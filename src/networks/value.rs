use crate::{boxed_and_zeroed, Board, Piece};

use super::{
    accumulator::Accumulator, activation::ReLU, layer::Layer
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-f0af0e30bc8c.network";

const SCALE: i32 = 400;

const DI: usize = 48;
const DK: usize = 32;
const DV: usize = 8;
const D1: usize = 16;

#[repr(C)]
pub struct ValueNetwork {
    wq: [[Accumulator<f32, DK>; DI]; 64],
    wk: [[Accumulator<f32, DK>; DI]; 64],
    wv: [[Accumulator<f32, DV>; DI]; 64],
    l1: Layer<f32, {64 * DV}, D1>,
    l2: Layer<f32, D1, 1>,
}

impl ValueNetwork {
    pub fn eval(&self, pos: &Board) -> i32 {
        let mut num_pieces = 0;

        let mut squares = [0; 32];
        let mut pieces = [0; 32];
    
        let flip = if pos.stm() > 0 { 56 } else { 0 };

        let threats = pos.threats_by(1 - pos.stm());
        let defences = pos.threats_by(pos.stm());
    
        for (stm, &side) in [pos.stm(), 1 - pos.stm()].iter().enumerate() {
            for piece in Piece::PAWN..=Piece::KING {
                let mut bb = pos.piece(side) & pos.piece(piece);
    
                while bb > 0 {
                    let sq = bb.trailing_zeros() as usize;

                    let bit = 1 << sq;
                    let state = usize::from(bit & threats > 0) + 2 * usize::from(bit & defences > 0);
    
                    squares[num_pieces] = sq ^ flip;
                    pieces[num_pieces] = 12 * state + 6 * stm + piece - 2;
                    num_pieces += 1;
    
                    bb &= bb - 1;
                }
            }
        }
    
        let mut hl = Accumulator([0.0; 64 * DV]);
    
        for i in 0..num_pieces {
            let mut temps = [0.0; 32];
            let mut max = 0f32;
    
            for j in 0..num_pieces {
                let query = &self.wq[squares[i]][pieces[i]];
                let key = &self.wk[squares[j]][pieces[j]];
    
                for k in 0..DK {
                    temps[j] += query.0[k] * key.0[k]
                }
    
                max = max.max(temps[j]);
            }
    
            let mut total = (64 - num_pieces) as f32 * (-max).exp();
    
            for t in temps.iter_mut().take(num_pieces) {
                *t = (*t - max).exp();
                total += *t;
            }
    
            for j in 0..num_pieces {
                temps[j] /= total;
    
                let value = &self.wv[squares[j]][pieces[j]];
                let weight = temps[j];
        
                for (k, &val) in value.0.iter().enumerate() {
                    hl.0[squares[i] * DV + k] += weight * val;
                }
            }
        }

        let l1 = self.l1.forward::<ReLU>(&hl);
        let l2 = self.l2.forward::<ReLU>(&l1);

        (l2.0[0] * SCALE as f32) as i32
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
