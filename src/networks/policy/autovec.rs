use super::{Align64, DIM, L1};

pub unsafe fn l2(w: &Align64<i8, {L1 * DIM / 2}>, l1: &Align64<u8, {L1 / 2}>) -> Align64<i32, DIM> {
    let mut l2 = Align64([0; DIM]);

    unsafe fn get4<T: Copy>(x: &[T], base: usize) -> [T; 4] {
        [
            *x.get_unchecked(4 * base),
            *x.get_unchecked(4 * base + 1),
            *x.get_unchecked(4 * base + 2),
            *x.get_unchecked(4 * base + 3),
        ]
    }

    for i in (0..L1 / 8).step_by(2) {
        // QA * QA / (1 << SHIFT)
        let inps1 = get4(&l1.0, i);
        let inps2 = get4(&l1.0, i + 1);

        for j in 0..DIM {
            // QB
            let ws1 = get4(&w.0, DIM * i + j);
            let ws2 = get4(&w.0, DIM * (i + 1) + j);

            // QA * QA * QB / (1 << SHIFT)
            for k in 0..4 {
                *l2.0.get_unchecked_mut(j) += i32::from(inps1[k]) * i32::from(ws1[k]);
                *l2.0.get_unchecked_mut(j) += i32::from(inps2[k]) * i32::from(ws2[k]);
            }
        }
    }

    l2
}
