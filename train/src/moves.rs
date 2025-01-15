use monty::ataxx::{Bitboard, Move};

pub const MAX_MOVES: usize = 96;
pub const NUM_MOVES: usize = 49 + OFFSETS[49];

pub fn map_move_to_index(mov: Move) -> usize {
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
