use crate::{
    boxed_and_zeroed,
    chess::{Board, Move},
};

use super::{accumulator::Accumulator, activation::ReLU, layer::Layer};

const QA: i16 = 512;

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-3329fb7f6624.network";

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubNet {
    ft: Layer<i16, 768, 16>,
}

impl SubNet {
    pub fn out(&self, feats: &[usize]) -> Accumulator<i16, 16> {
        self.ft.forward_from_slice(feats)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    subnets: [SubNet; 128],
    pub(crate) good_see_subnet: SubNet,
    hce: Layer<f32, 4, 1>,
}

impl PolicyNetwork {
    pub fn get(&self, pos: &Board, mov: &Move, feats: &[usize], good_see: &Accumulator<i16, 16>) -> f32 {
        let flip = pos.flip_val();

        let from_subnet = &self.subnets[usize::from(mov.src() ^ flip)];
        let from_vec = from_subnet.out(feats);

        let to_subnet = &self.subnets[64 + usize::from(mov.to() ^ flip)];
        let mut to_vec = to_subnet.out(feats);

        if pos.see(mov, -108) {
            to_vec.add(good_see);
        }

        let hce = self.hce.forward::<ReLU>(&Self::get_hce_feats(pos, mov)).0[0];

        from_vec.dot::<ReLU, QA>(&to_vec) + hce
    }

    pub fn get_hce_feats(_: &Board, mov: &Move) -> Accumulator<f32, 4> {
        let mut feats = [0.0; 4];

        if mov.is_promo() {
            feats[mov.promo_pc() - 3] = 1.0;
        }

        Accumulator(feats)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UnquantisedSubNet {
    ft: Layer<f32, 768, 16>,
}

impl UnquantisedSubNet {
    fn quantise(&self, qa: i16) -> SubNet {
        SubNet {
            ft: self.ft.quantise_i16(qa, 1.98),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UnquantisedPolicyNetwork {
    subnets: [UnquantisedSubNet; 128],
    good_see_subnet: UnquantisedSubNet,
    hce: Layer<f32, 4, 1>,
}

impl UnquantisedPolicyNetwork {
    pub fn quantise(&self) -> Box<PolicyNetwork> {
        let mut quant: Box<PolicyNetwork> = unsafe { boxed_and_zeroed() };

        for (q, unq) in quant.subnets.iter_mut().zip(self.subnets.iter()) {
            *q = unq.quantise(QA);
        }

        quant.good_see_subnet = self.good_see_subnet.quantise(QA);

        quant.hce = self.hce;

        quant
    }
}
