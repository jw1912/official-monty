use monty::ataxx::Board;

pub const INPUT_SIZE: usize = NUM_TUPLES * PER_TUPLE;
pub const MAX_ACTIVE: usize = NUM_TUPLES;

const PER_TUPLE: usize = 3usize.pow(4);
const NUM_TUPLES: usize = 36;

pub fn map_policy_inputs(pos: &Board, mut f: impl FnMut(usize)) {
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
