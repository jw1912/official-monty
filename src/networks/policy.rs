use super::{Accumulator, Layer, ReLU};
use crate::chess::{Board, Move};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-6b5dc1d7fff9.network";

pub struct PolicyFeats {
    pub list: [u16; 32],
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubNet {
    ft: Layer<768, 16>,
    l2: Layer<16, 16>,
}

impl SubNet {
    fn out(&self, feats: &PolicyFeats) -> Accumulator<16> {
        let mut l2 = self.ft.biases;

        for &feat in &feats.list[..feats.len] {
            l2.add(&self.ft.weights[usize::from(feat)]);
        }

        self.l2.forward::<ReLU>(&l2)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub subnets: [[SubNet; 2]; 448],
    pub hce: [f32; 5],
}

impl PolicyNetwork {
    pub fn get(&self, pos: &Board, mov: &Move, feats: &PolicyFeats, threats: u64) -> f32 {
        let flip = pos.flip_val();
        let pc = pos.get_pc(1 << mov.src()) - 1;

        let from_threat = usize::from(threats & (1 << mov.src()) > 0);
        let from_subnet = &self.subnets[usize::from(mov.src() ^ flip)][from_threat];
        let from_vec = from_subnet.out(feats);

        let good_see = usize::from(pos.see(mov, -108));
        let to_subnet = &self.subnets[64 * pc + usize::from(mov.to() ^ flip)][good_see];
        let to_vec = to_subnet.out(feats);

        let hce = if mov.is_promo() {
            self.hce[mov.promo_pc() - 3]
        } else {
            0.0
        };

        from_vec.dot::<ReLU>(&to_vec) + hce
    }
}
