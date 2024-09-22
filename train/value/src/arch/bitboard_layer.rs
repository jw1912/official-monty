use std::ops::AddAssign;

use goober::{FeedForwardNetwork, Matrix, OutputLayer, Vector};

#[derive(Clone, Copy)]
pub struct BitboardLayer<const N: usize> {
    weights: Matrix<256, N>,
}

impl<const N: usize> BitboardLayer<N> {
    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        Self {
            weights: Matrix::from_fn(|_, _| f()),
        }
    }
}

impl<const N: usize> AddAssign<&BitboardLayer<N>> for BitboardLayer<N> {
    fn add_assign(&mut self, rhs: &BitboardLayer<N>) {
        self.weights += &rhs.weights;
    }
}

impl<const N: usize> FeedForwardNetwork for BitboardLayer<N> {
    type OutputType = Vector<N>;
    type InputType = [u64; 4];
    type Layers = BitboardLayerLayers<N>;

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        let mut out = Vector::zeroed();

        for (i, &(mut bb)) in input.iter().enumerate() {
            while bb > 0 {
                let sq = bb.trailing_zeros() as usize;
    
                out += self.weights[64 * i + sq];
    
                bb &= bb - 1;
            }
        }

        Self::Layers { out }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        _: &Self::Layers,
    ) -> Self::InputType {
        for (i, &(mut bb)) in input.iter().enumerate() {
            while bb > 0 {
                let sq = bb.trailing_zeros() as usize;
    
                grad.weights[64 * i + sq] += out_err;
    
                bb &= bb - 1;
            }
        }

        [0; 4]
    }

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.weights.adam(&g.weights, &mut m.weights, &mut v.weights, adj, lr);
    }
}

#[derive(Clone, Copy)]
pub struct BitboardLayerLayers<const N: usize> {
    out: Vector<N>,
}

impl<const N: usize> BitboardLayerLayers<N> {
    pub fn zeroed() -> Self {
        Self { out: Vector::zeroed() }
    }
}

impl<const N: usize> OutputLayer<Vector<N>> for BitboardLayerLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.out
    }
}
