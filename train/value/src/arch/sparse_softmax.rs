use goober::{Matrix, Vector};

pub struct SparseSoftmax;

impl SparseSoftmax {
    pub fn backprop(active: &[(usize, usize)], output: &[f32; 64], err: &[f32; 64]) -> Vector<64> {
        let mut jacobian = [Vector::<64>::zeroed(); 64];

        for &(sq1, _) in active {
            for &(sq2, _) in active {
                jacobian[sq1][sq2] = -output[sq1] * output[sq2];
            }
        }

        for &(sq, _) in active {
            jacobian[sq][sq] += output[sq];
        }

        let jacobian = Matrix::from_raw(jacobian);

        jacobian.mul(&Vector::from_raw(*err))
    }
}
