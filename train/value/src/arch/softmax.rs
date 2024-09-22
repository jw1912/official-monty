use goober::{Matrix, Vector};

pub struct Softmax;

impl Softmax {
    pub fn backprop(output: &[f32; 64], err: &[f32; 64]) -> Vector<64> {
        let mut jacobian = [Vector::<64>::zeroed(); 64];

        for sq1 in 0..64 {
            for sq2 in 0..64 {
                jacobian[sq1][sq2] = -output[sq1] * output[sq2];
            }
        }

        for sq in 0..64 {
            jacobian[sq][sq] += output[sq];
        }

        let jacobian = Matrix::from_raw(jacobian);

        jacobian.mul(&Vector::from_raw(*err))
    }
}
