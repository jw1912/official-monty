use crate::Board;

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-68271c19286d.network";

const SCALE: i32 = 400;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<{ 768 * 4 }, 128>,
    l2: Layer<128, 16>,
    l3: Layer<16, 16>,
    l4: Layer<16, 1>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> i32 {
        let mut l2 = self.l1.biases;

        board.map_value_features(|feat| {
            for (i, d) in l2.vals.iter_mut().zip(&self.l1.weights[feat].vals) {
                *i += *d;
            }
        });

        let l3 = self.l2.forward(&l2);
        let l4 = self.l3.forward(&l3);
        let out = self.l4.forward(&l4);

        (out.vals[0] * SCALE as f32) as i32
    }
}

#[derive(Clone, Copy)]
struct Layer<const M: usize, const N: usize> {
    weights: [Accumulator<N>; M],
    biases: Accumulator<N>,
}

impl<const M: usize, const N: usize> Layer<M, N> {
    fn forward(&self, inputs: &Accumulator<M>) -> Accumulator<N> {
        let mut fwd = self.biases;

        for (i, d) in inputs.vals.iter().zip(self.weights.iter()) {
            let act = screlu(*i);
            fwd.madd(act, d);
        }

        fwd
    }
}

#[inline]
fn screlu(x: f32) -> f32 {
    x.clamp(0.0, 1.0).powi(2)
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Accumulator<const HIDDEN: usize> {
    vals: [f32; HIDDEN],
}

impl<const HIDDEN: usize> Accumulator<HIDDEN> {
    fn madd(&mut self, mul: f32, other: &Self) {
        for (i, &j) in self.vals.iter_mut().zip(other.vals.iter()) {
            *i += mul * j;
        }
    }
}
