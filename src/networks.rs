mod policy;
mod value;

pub use policy::{PolicyFeats, PolicyFileDefaultName, PolicyNetwork, SubNet};
pub use value::{ValueFileDefaultName, ValueNetwork};

#[derive(Clone, Copy)]
struct Layer<const M: usize, const N: usize> {
    weights: [Accumulator<N>; M],
    biases: Accumulator<N>,
}

impl<const M: usize, const N: usize> Layer<M, N> {
    fn forward<A: Activation>(&self, inputs: &Accumulator<M>) -> Accumulator<N> {
        let mut fwd = self.biases;

        for (i, d) in inputs.vals.iter().zip(self.weights.iter()) {
            let act = A::activate(*i);
            fwd.madd(act, d);
        }

        fwd
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Accumulator<const N: usize> {
    vals: [f32; N],
}

impl<const N: usize> Accumulator<N> {
    fn add(&mut self, other: &Self) {
        for (i, &j) in self.vals.iter_mut().zip(other.vals.iter()) {
            *i += j;
        }
    }

    fn madd(&mut self, mul: f32, other: &Self) {
        for (i, &j) in self.vals.iter_mut().zip(other.vals.iter()) {
            *i += mul * j;
        }
    }

    fn dot<A: Activation>(&self, other: &Self) -> f32 {
        let mut res = 0.0;

        for (&i, &j) in self.vals.iter().zip(other.vals.iter()) {
            res += A::activate(i) * A::activate(j);
        }

        res
    }
}

trait Activation {
    fn activate(x: f32) -> f32;
}

pub struct SCReLU;
impl Activation for SCReLU {
    #[inline]
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0).powi(2)
    }
}

pub struct ReLU;
impl Activation for ReLU {
    #[inline]
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }
}
