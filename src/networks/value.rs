use crate::ataxx::Board;

use super::NETS;

const INPUTS: usize = 2916;
const HIDDEN: usize = 256;

const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

#[repr(C, align(64))]
pub struct ValueNetwork {
    l1_weights: [Accumulator; INPUTS],
    l1_bias: Accumulator,
    l2_weights: Accumulator,
    l2_bias: i16,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Accumulator {
    vals: [i16; HIDDEN],
}

#[inline]
fn screlu(x: i16) -> i32 {
    i32::from(x).clamp(0, QA).pow(2)
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> i32 {
        let mut acc = self.l1_bias;

        board.value_features_map(|feat| {
            for (i, d) in acc.vals.iter_mut().zip(&self.l1_weights[feat].vals) {
                *i += *d;
            }
        });

        let mut eval = 0;

        for (&v, &w) in acc.vals.iter().zip(self.l2_weights.vals.iter()) {
            eval += screlu(v) * i32::from(w);
        }

        (eval / QA + i32::from(self.l2_bias)) * SCALE / QAB
    }
}

pub fn get(pos: &Board) -> f32 {
    let cp = NETS.0.eval(pos);
    1.0 / (1.0 + (-cp as f32 / 400.0).exp())
}
