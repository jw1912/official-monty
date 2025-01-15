use crate::ataxx::{Board, Move};

use goober::{activation, layer, FeedForwardNetwork, SparseVector};

use super::NETS;

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: layer::SparseConnected<activation::ReLU, 2916, 8>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub subnets: [SubNet; 99],
}

impl PolicyNetwork {
    pub fn get(&self, mov: Move, feats: &SparseVector) -> f32 {
        let from_subnet = &self.subnets[mov.src().min(49)];
        let from_vec = from_subnet.out(feats);

        let to_subnet = &self.subnets[50 + mov.to().min(48)];
        let to_vec = to_subnet.out(feats);

        from_vec.dot(&to_vec)
    }
}

pub fn get(mov: Move, feats: &SparseVector) -> f32 {
    NETS.1.get(mov, feats)
}

pub fn get_feats(pos: &Board) -> SparseVector {
    let mut feats = SparseVector::with_capacity(36);

    pos.value_features_map(|feat| feats.push(feat));

    feats
}
