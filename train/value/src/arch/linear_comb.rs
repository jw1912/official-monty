use goober::{OutputLayer, Vector};

use super::{bitboard_layer::BitboardLayerLayers, TOKENS};

pub struct LinearComb<const N: usize>;

impl<const N: usize> LinearComb<N> {
    pub fn fwd(weights: &[f32; TOKENS], vecs: &[BitboardLayerLayers<N>; TOKENS]) -> Vector<N> {
        let mut res = Vector::zeroed();

        for pc in 0..TOKENS {
            res += weights[pc] * vecs[pc].output_layer();
        }

        res
    }

    pub fn backprop(vecs: &[BitboardLayerLayers<N>; TOKENS], err: Vector<N>) -> [f32; TOKENS] {
        let mut errs = [0.0; TOKENS];

        for pc in 0..TOKENS {
            errs[pc] = vecs[pc].output_layer().dot(&err)
        }

        errs
    }
}