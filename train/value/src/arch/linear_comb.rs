use goober::{OutputLayer, Vector};

use super::one_hot_layer::OneHotLayerLayers;

pub struct LinearComb<const N: usize>;

impl<const N: usize> LinearComb<N> {
    pub fn fwd(active: &[(usize, usize)], weights: &[f32; 64], vecs: &[OneHotLayerLayers<N>]) -> Vector<N> {
        let mut res = Vector::zeroed();

        for (i, &(sq, _)) in active.iter().enumerate() {
            res += weights[sq] * vecs[i].output_layer();
        }

        res
    }

    pub fn backprop(active: &[(usize, usize)], vecs: &[OneHotLayerLayers<N>], err: Vector<N>) -> [f32; 64] {
        let mut errs = [0.0; 64];

        for (i, &(sq, _)) in active.iter().enumerate() {
            errs[sq] = vecs[i].output_layer().dot(&err)
        }

        errs
    }
}