use super::Accumulator;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvKernel([[f32; 3]; 3]);

#[repr(C)]
pub struct ConvLayer<const IN: usize, const OUT: usize> {
    weights: [[ConvKernel; IN]; OUT],
    biases: [[f32; 64]; OUT],
}

impl<const IN: usize, const OUT: usize> ConvLayer<IN, OUT> {
    pub fn forward<const N: usize, const M: usize>(&self, input: &Accumulator<f32, N>) -> Accumulator<f32, M> {
        assert_eq!(N, 64 * IN);
        assert_eq!(M, 64 * OUT);

        let mut out = Accumulator([0.0; M]);

        for i in 0..OUT {
            for j in 0..IN {
                let kernel = self.weights[i][j];

                for x in 0..8 {
                    for y in 0..8 {
                        let mut elem = self.biases[i][8 * x + y];

                        for a in 0..3 {
                            for b in 0..3 {
                                let ia = (x + a).checked_sub(1);
                                let ib = (y + b).checked_sub(1);

                                if let Some(ia) = ia {
                                    if let Some(ib) = ib {
                                        if ia < 8 && ib < 8 {
                                            elem += kernel.0[a][b] * input.0[64 * i + 8 * ia + ib];
                                        }
                                    }
                                }
                            }
                        }

                        out.0[64 * i + 8 * x + y] = elem;
                    }
                }
            }
        }

        out
    }
}
