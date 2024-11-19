use crate::{networks::activation::Identity, Board};

use super::{
    conv::ConvLayer, layer::Layer, residual::ResidualBlock
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-820e30473d09.network";

const BLOCKS: usize = 4;
const FILTERS: usize = 4;
const OUTPUT_CHANNELS: usize = 2;

const EMBED: usize = FILTERS * 64;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<f32, 768, EMBED>,
    res_tower: [ResidualBlock<FILTERS>; BLOCKS],
    l2: ConvLayer<FILTERS, OUTPUT_CHANNELS>,
    l3: Layer<f32, { OUTPUT_CHANNELS * 64 }, 3>
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

        l2.relu();

        for block in &self.res_tower {
            l2 = block.forward(&l2);
        }

        let l3 = self.l2.forward(&l2);
        let out = self.l3.forward::<Identity>(&l3);

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
