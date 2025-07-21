use std::ops::{AddAssign, Mul};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Accumulator<T: Copy, const N: usize>(pub [T; N]);

impl<T: AddAssign<T> + Copy + Mul<T, Output = T>, const N: usize> Accumulator<T, N> {
    pub fn add(&mut self, other: &Self) {
        for (i, &j) in self.0.iter_mut().zip(other.0.iter()) {
            *i += j;
        }
    }

    pub fn madd(&mut self, mul: T, other: &Self) {
        for (i, &j) in self.0.iter_mut().zip(other.0.iter()) {
            *i += mul * j;
        }
    }
}

impl<const N: usize> Accumulator<i16, N> {
    pub fn add_multi(&mut self, adds: &[usize], weights: &[Self]) {
        const REGS: usize = 8;
        const PER: usize = REGS * 16;

        let mut regs = [0i16; PER];

        for i in 0..N / PER {
            let offset = PER * i;

            for (j, reg) in regs.iter_mut().enumerate() {
                *reg = self.0[offset + j];
            }

            for &add in adds {
                let this_weight = &weights[add];

                for (j, reg) in regs.iter_mut().enumerate() {
                    *reg += this_weight.0[offset + j];
                }
            }

            for (j, reg) in regs.iter().enumerate() {
                self.0[offset + j] = *reg;
            }
        }
    }
}
