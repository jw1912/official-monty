use crate::ataxx::{Bitboard, Board, Move};

static POLICY: PolicyNetwork = unsafe {
    std::mem::transmute(*include_bytes!("../../ataxx-policy.network"))
};

const HIDDEN: usize = 256;
const Q: i16 = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Accumulator<T: Copy>([T; HIDDEN]);

#[repr(C)]
pub struct PolicyNetwork {
    l0w: [Accumulator<i8>; 98],
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

    map_feats(pos, |feat| {
        for (i, &j) in hl.0.iter_mut().zip(POLICY.l0w[feat].0.iter()) {
            *i += i16::from(j);
        }
    });

    for i in &mut hl.0 {
        *i = (*i).clamp(0, Q);
    }

    hl
}

fn map_feats(pos: &Board, mut f: impl FnMut(usize)) {
    let mut bb = pos.boys();
    while bb > 0 {
        f(bb.trailing_zeros() as usize);
        bb &= bb - 1;
    }

    let mut bb = pos.opps();
    while bb > 0 {
        f(49 + bb.trailing_zeros() as usize);
        bb &= bb - 1;
    }
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
