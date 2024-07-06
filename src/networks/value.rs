use crate::Board;
use super::{Layer, SCReLU};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-031bf2c50080.network";

const SCALE: i32 = 400;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<{ 768 * 4 }, 512>,
    l2: Layer<512, 16>,
    l3: Layer<16, 16>,
    l4: Layer<16, 16>,
    l5: Layer<16, 16>,
    l6: Layer<16, 16>,
    l7: Layer<16, 16>,
    l8: Layer<16, 16>,
    l9: Layer<16, 16>,
    l10: Layer<16, 16>,
    l11: Layer<16, 1>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> i32 {
        let mut l2 = self.l1.biases;

        board.map_value_features(|feat| l2.add(&self.l1.weights[feat]));

        let l3 = self.l2.forward::<SCReLU>(&l2);
        let l4 = self.l3.forward::<SCReLU>(&l3);
        let l5 = self.l4.forward::<SCReLU>(&l4);
        let l6 = self.l5.forward::<SCReLU>(&l5);
        let l7 = self.l6.forward::<SCReLU>(&l6);
        let l8 = self.l7.forward::<SCReLU>(&l7);
        let l9 = self.l8.forward::<SCReLU>(&l8);
        let l10 = self.l9.forward::<SCReLU>(&l9);
        let l11 = self.l10.forward::<SCReLU>(&l10);
        let out = self.l11.forward::<SCReLU>(&l11);

        (out.vals[0] * SCALE as f32) as i32
    }
}
