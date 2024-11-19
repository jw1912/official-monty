use crate::Board;

use super::{
    activation::SCReLU,
    layer::Layer,
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-e922509c7ba5.network";

const L1: usize = 256;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<f32, 768, L1>,
    l2: Layer<f32, L1, 3>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> (f32, f32, f32) {
        let mut count = 0;
        let mut feats = [0; 32];
        board.map_value_features(|feat| {
            feats[count] = feat;
            count += 1;
        });

        let mut l2 = self.l1.biases;

        l2.add_multi(&feats[..count], &self.l1.weights);

        let out = self.l2.forward::<SCReLU>(&l2);

        let mut win = out.0[2];
        let mut draw = out.0[1];
        let mut loss = out.0[0];

        let max = win.max(draw).max(loss);

        win = (win - max).exp();
        draw = (draw - max).exp();
        loss = (loss - max).exp();

        let sum = win + draw + loss;

        (win / sum, draw / sum, loss / sum)
    }
}
