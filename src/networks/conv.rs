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
            for j in 0..64 {
                out.0[64 * i + j] = self.biases[i][j];
            }
        }

        for i in 0..OUT {
            for j in 0..IN {
                let kernel = self.weights[i][j];

                for x in 0..8 {
                    for y in 0..8 {
                        let mut elem = 0.0;

                        for a in 0..3 {
                            for b in 0..3 {
                                let ia: usize = x + a;
                                let ib: usize = y + b;
                                let ia = ia.checked_sub(1);
                                let ib = ib.checked_sub(1);

                                if let Some(ia) = ia {
                                    if let Some(ib) = ib {
                                        if ia < 8 && ib < 8 {
                                            elem += kernel.0[a][b] * input.0[64 * j + 8 * ia + ib];
                                        }
                                    }
                                }
                            }
                        }

                        out.0[64 * i + 8 * x + y] += elem;
                    }
                }
            }
        }

        out
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    #[test]
    fn t() {
        let conv = ConvLayer {
            weights: [
                [
                    ConvKernel([
                        [1.0, 1.0, 1.0],
                        [-1.0, -1.0, 1.0],
                        [-1.0, -1.0, 1.0],
                    ])
                ],
            ],
            biases: [[0.0; 64]; 1],
        };

        let input = Accumulator([
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
        ]);

        let output = conv.forward::<64, 64>(&input);

        assert_eq!(
            output.0,
            [
                 2.0, -4.0, -6.0, -16.0, -10.0, -4.0, -6.0, -22.0,
                 5.0,  2.0,  3.0,  -8.0,  -3.0,  2.0,  3.0, -15.0,
                13.0, 14.0, 15.0,   4.0,   9.0, 14.0, 15.0,  -7.0,
                 5.0,  2.0,  3.0,  -8.0,  -3.0,  2.0,  3.0, -15.0,
                13.0, 14.0, 15.0,   4.0,   9.0, 14.0, 15.0,  -7.0,
                 5.0,  2.0,  3.0,  -8.0,  -3.0,  2.0,  3.0, -15.0,
                13.0, 14.0, 15.0,   4.0,   9.0, 14.0, 15.0,  -7.0,
                 4.0,  2.0,  4.0,  -2.0,   0.0,  2.0,  4.0,  -8.0
            ]
        );
    }
}
