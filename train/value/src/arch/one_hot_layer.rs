use std::ops::AddAssign;

use goober::{activation::ReLU, FeedForwardNetwork, Matrix, OutputLayer, Vector};

#[derive(Clone, Copy)]
pub struct OneHotLayer<const M: usize, const N: usize> {
    weights: Matrix<M, N>,
    biases: Vector<N>
}

impl<const M: usize, const N: usize> OneHotLayer<M, N> {
    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        Self {
            weights: Matrix::from_fn(|_, _| f()),
            biases: Vector::from_fn(|_| f()),
        }
    }
}

impl<const M: usize, const N: usize> AddAssign<&OneHotLayer<M, N>> for OneHotLayer<M, N> {
    fn add_assign(&mut self, rhs: &OneHotLayer<M, N>) {
        self.weights += &rhs.weights;
    }
}

impl<const M: usize, const N: usize> FeedForwardNetwork for OneHotLayer<M, N> {
    type OutputType = Vector<N>;
    type InputType = usize;
    type Layers = OneHotLayerLayers<N>;

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        Self::Layers {
            out: (self.weights[*input] + self.biases).activate::<ReLU>(),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        mut out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        out_err = out_err * layers.out.derivative::<ReLU>();

        grad.weights[*input] += out_err;

        grad.biases += out_err;

        0
    }

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.weights.adam(&g.weights, &mut m.weights, &mut v.weights, adj, lr);
    }
}

#[derive(Clone, Copy)]
pub struct OneHotLayerLayers<const N: usize> {
    out: Vector<N>,
}

impl<const N: usize> OutputLayer<Vector<N>> for OneHotLayerLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.out
    }
}
