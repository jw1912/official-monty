use goober::{Matrix, Vector};

use super::TOKENS;

pub struct Softmax;

impl Softmax {
    pub fn backprop(output: &[f32; TOKENS], err: &[f32; TOKENS]) -> Vector<TOKENS> {
        let mut jacobian = [Vector::<TOKENS>::zeroed(); TOKENS];

        for sq1 in 0..TOKENS {
            for sq2 in 0..TOKENS {
                jacobian[sq1][sq2] = -output[sq1] * output[sq2];
            }
        }

        for sq in 0..TOKENS {
            jacobian[sq][sq] += output[sq];
        }

        let jacobian = Matrix::from_raw(jacobian);

        jacobian.mul(&Vector::from_raw(*err))
    }
}
