use crate::ataxx::{Bitboard, Board, Move};

static POLICY: PolicyNetwork = unsafe {
    std::mem::transmute(*include_bytes!("../../checkpoints/policy001-40/quantised.network"))
};

const PER_TUPLE: usize = 3usize.pow(4);
const NUM_TUPLES: usize = 36;
const HIDDEN: usize = 128;
const Q: i16 = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Accumulator<T: Copy>([T; HIDDEN]);

#[repr(C)]
pub struct PolicyNetwork {
    l0w: [Accumulator<i8>; PER_TUPLE * NUM_TUPLES],
    l0b: Accumulator<i8>,
    l1w: [Accumulator<i8>; 578],
    l1b: [i8; 578],
}

impl PolicyNetwork {
    pub fn get(&self, mov: Move, feats: &Accumulator<i16>) -> f32 {
        let idx = map_move_to_index(mov);

        let mut res = 0;

        for (&i, &j) in feats.0.iter().zip(POLICY.l1w[idx].0.iter()) {
            res += i32::from(i) * i32::from(j);
        }

        (res as f32 / f32::from(Q) + f32::from(POLICY.l1b[idx])) / f32::from(Q)
    }
}

pub fn get(mov: Move, feats: &Accumulator<i16>) -> f32 {
    POLICY.get(mov, feats)
}

pub fn get_feats(pos: &Board) -> Accumulator<i16> {
    let mut hl = Accumulator([0; HIDDEN]);
    
    for (i, &j) in hl.0.iter_mut().zip(POLICY.l0b.0.iter()) {
        *i = i16::from(j);
    }

    map_policy_inputs(pos, |feat| {
        for (i, &j) in hl.0.iter_mut().zip(POLICY.l0w[feat].0.iter()) {
            *i += i16::from(j);
        }
    });

    for i in &mut hl.0 {
        *i = (*i).clamp(0, Q);
    }

    hl
}

fn map_move_to_index(mov: Move) -> usize {
    if mov.is_single() {
        mov.to()
    } else {
        let src = mov.src();
        let dst = mov.to();

        let doubles = Bitboard::doubles(src);
        let below = doubles & ((1 << dst) - 1);

        49 + OFFSETS[src] + below.count_ones() as usize
    }
}

static OFFSETS: [usize; 50] = {
    let mut src = 0;

    let mut res = [0; 50];

    while src < 49 {
        let reachable = Bitboard::doubles(src);
        src += 1;
        res[src] = res[src - 1] + reachable.count_ones() as usize;
    }

    res
};

fn map_policy_inputs(pos: &Board, mut f: impl FnMut(usize)) {
    let boys = pos.boys();
    let opps = pos.opps();

    for i in 0..6 {
        for j in 0..6 {
            const POWERS: [usize; 4] = [1, 3, 9, 27];
            const MASK: u64 = 0b0001_1000_0011;

            let tuple = 6 * i + j;
            let mut stm = PER_TUPLE * tuple;

            let offset = 7 * i + j;
            let mut b = (boys >> offset) & MASK;
            let mut o = (opps >> offset) & MASK;

            while b > 0 {
                let mut sq = b.trailing_zeros() as usize;
                if sq > 6 {
                    sq -= 5;
                }

                stm += POWERS[sq];

                b &= b - 1;
            }

            while o > 0 {
                let mut sq = o.trailing_zeros() as usize;
                if sq > 6 {
                    sq -= 5;
                }

                stm += 2 * POWERS[sq];
                o &= o - 1;
            }

            f(stm);
        }
    }
}
