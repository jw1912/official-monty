use super::{conv::ConvLayer, Accumulator};

#[repr(C)]
pub struct ResidualBlock<const N: usize> {
    conv1: ConvLayer<N, N>,
    conv2: ConvLayer<N, N>,
}

impl<const N: usize> ResidualBlock<N> {
    pub fn forward<const M: usize>(&self, input: &Accumulator<f32, M>) -> Accumulator<f32, M> {
        assert_eq!(M, 64 * N);

        let conv1 = self.conv1.forward::<M, M>(input);
        let mut conv2 = self.conv2.forward::<M, M>(&conv1);

        conv2.add(input);
        conv2.relu();

        conv2
    }
}
