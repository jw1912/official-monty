use crate::chess::{Board, Move};

use goober::{activation, layer, FeedForwardNetwork, Matrix, SparseVector, Vector};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-a7af95e30ed6.network";

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: layer::SparseConnected<activation::ReLU, 768, 16>,
}

impl SubNet {
    pub const fn zeroed() -> Self {
        Self {
            ft: layer::SparseConnected::zeroed(),
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let matrix = Matrix::from_fn(|_, _| f());
        let vector = Vector::from_fn(|_| f());

        Self {
            ft: layer::SparseConnected::from_raw(matrix, vector),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub from_subnets: [[SubNet; 2]; 64],
    pub to_subnets: [[SubNet; 2]; 64],
    pub pc_subnets: [SubNet; 6],
    pub hce: layer::DenseConnected<activation::Identity, 4, 1>,
}

impl PolicyNetwork {
    pub const fn zeroed() -> Self {
        Self {
            from_subnets: [[SubNet::zeroed(); 2]; 64],
            to_subnets: [[SubNet::zeroed(); 2]; 64],
            pc_subnets: [SubNet::zeroed(); 6],
            hce: layer::DenseConnected::zeroed(),
        }
    }

    pub fn get(&self, pos: &Board, mov: &Move, feats: &SparseVector, threats: u64, pc_vecs: &[Vector<16>; 6]) -> f32 {
        let flip = pos.flip_val();
        let pc = pos.get_pc(1 << mov.src()) - 2;

        let from_threat = usize::from(threats & (1 << mov.src()) > 0);
        let from_subnet = &self.from_subnets[usize::from(mov.src() ^ flip)][from_threat];
        let from_vec = from_subnet.out(feats);

        let good_see = usize::from(pos.see(mov, -108));
        let to_subnet = &self.to_subnets[usize::from(mov.to() ^ flip)][good_see];
        let to_vec = to_subnet.out(feats);

        let hce = self.hce.out(&Self::get_hce_feats(pos, mov))[0];

        pc_vecs[pc].dot(&(from_vec * to_vec)) + hce
    }

    pub fn get_hce_feats(_: &Board, mov: &Move) -> Vector<4> {
        let mut feats = Vector::zeroed();

        if mov.is_promo() {
            feats[mov.promo_pc() - 3] = 1.0;
        }

        feats
    }
}
