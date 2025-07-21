#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Align64<T, const N: usize>(pub [T; N]);

impl<const N: usize> Align64<i16, N> {
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