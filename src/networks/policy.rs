use crate::chess::{Board, Move};

use goober::{activation, layer, FeedForwardNetwork, Matrix, SparseVector, Vector};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-a0d8daea67c1.network";

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
    pub from_subnets: [SubNet; 64],
    pub to_subnets: [SubNet; 64],
    pub hce: layer::DenseConnected<activation::Identity, 4, 1>,
}

impl PolicyNetwork {
    pub const fn zeroed() -> Self {
        Self {
            from_subnets: [SubNet::zeroed(); 64],
            to_subnets: [SubNet::zeroed(); 64],
            hce: layer::DenseConnected::zeroed(),
        }
    }

    pub fn get(&self, pos: &Board, mov: &Move, feats: &SparseVector) -> f32 {
        let flip = pos.flip_val();

        let from_subnet = &self.from_subnets[usize::from(mov.src() ^ flip)];
        let from_vec = from_subnet.out(feats);

        let to_subnet = &self.to_subnets[usize::from(mov.to() ^ flip)];
        let to_vec = to_subnet.out(feats);

        let hce = self.hce.out(&Self::get_hce_feats(pos, mov))[0];

        from_vec.dot(&to_vec) + hce
    }

    pub fn get_hce_feats(_: &Board, mov: &Move) -> Vector<4> {
        let mut feats = Vector::zeroed();

        if mov.is_promo() {
            feats[mov.promo_pc() - 3] = 1.0;
        }

        feats
    }
}
