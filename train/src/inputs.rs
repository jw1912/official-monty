use monty::ataxx::Board;

pub const INPUT_SIZE: usize = 98;
pub const MAX_ACTIVE: usize = 49;

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Board, mut f: F) {
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
