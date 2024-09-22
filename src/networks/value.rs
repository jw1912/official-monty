use crate::{boxed_and_zeroed, Board};

use super::{
    activation::SCReLU,
    layer::Layer,
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-e735b9de604c.network";

const QA: i16 = 512;
const SCALE: i32 = 400;


#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<i16, { 768 * 4 }, 128>,
    l2: Layer<f32, 128, 1>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> i32 {
        let l2 = self.l1.forward(board);
        let out = self.l2.forward_from_i16::<SCReLU, QA>(&l2);

        (out.0[0] * SCALE as f32) as i32
    }
}

#[repr(C)]
pub struct UnquantisedValueNetwork {
    l1: Layer<f32, { 768 * 4 }, 128>,
    l2: Layer<f32, 128, 1>,
}

impl UnquantisedValueNetwork {
    pub fn quantise(&self) -> Box<ValueNetwork> {
        let mut quantised: Box<ValueNetwork> = unsafe { boxed_and_zeroed() };

        self.l1.quantise_into_i16(&mut quantised.l1, QA, 0.99);
        
        quantised.l2 = self.l2;

        quantised
    }
}
