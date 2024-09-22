use super::{accumulator::Accumulator, activation::Activation};

#[derive(Clone, Copy)]
pub struct Layer<T: Copy, const M: usize, const N: usize> {
    weights: [Accumulator<T, N>; M],
    biases: Accumulator<T, N>,
}

impl<const M: usize, const N: usize> Layer<i16, M, N> {
    pub fn forward_from_slice(&self, feats: &[usize]) -> Accumulator<i16, N> {
        let mut out = self.biases;

        for &feat in feats {
            out.add(&self.weights[feat])
        }

        out
    }
}

impl<const M: usize, const N: usize> Layer<f32, M, N> {
    pub fn forward<T: Activation>(&self, inputs: &Accumulator<f32, M>) -> Accumulator<f32, N> {
        let mut fwd = self.biases;

        for (i, d) in inputs.0.iter().zip(self.weights.iter()) {
            let act = T::activate(*i);
            fwd.madd(act, d);
        }

        fwd
    }

    pub fn forward_from_i16<T: Activation, const QA: i16>(
        &self,
        inputs: &Accumulator<i16, M>,
    ) -> Accumulator<f32, N> {
        let mut fwd = self.biases;

        for (i, d) in inputs.0.iter().zip(self.weights.iter()) {
            let act = T::activate(f32::from(*i) / f32::from(QA));
            fwd.madd(act, d);
        }

        fwd
    }

    pub fn quantise_into_i16(&self, dest: &mut Layer<i16, M, N>, qa: i16, warn_limit: f32) {
        for (acc_i, acc_j) in dest.weights.iter_mut().zip(self.weights.iter()) {
            *acc_i = acc_j.quantise_i16(qa, warn_limit);
        }

        dest.biases = self.biases.quantise_i16(qa, warn_limit);
    }

    pub fn quantise_i16(&self, qa: i16, warn_limit: f32) -> Layer<i16, M, N> {
        let mut res = Layer {
            weights: [Accumulator([0; N]); M],
            biases: Accumulator([0; N]),
        };

        self.quantise_into_i16(&mut res, qa, warn_limit);

        res
    }
}