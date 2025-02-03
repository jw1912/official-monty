use std::{
    ops::Add,
    sync::{
        atomic::{AtomicI32, AtomicI64, AtomicU16, AtomicU32, AtomicU8, Ordering},
        RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use crate::chess::{GameState, Move};

const QUANT: i16 = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodePtr(u32);

impl NodePtr {
    pub const NULL: Self = Self(u32::MAX);

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub fn new(half: bool, idx: u32) -> Self {
        Self((u32::from(half) << 31) | idx)
    }

    pub fn half(self) -> bool {
        self.0 & (1 << 31) > 0
    }

    pub fn idx(self) -> usize {
        (self.0 & 0x7FFFFFFF) as usize
    }

    pub fn inner(self) -> u32 {
        self.0
    }

    pub fn from_raw(inner: u32) -> Self {
        Self(inner)
    }
}

impl Add<usize> for NodePtr {
    type Output = NodePtr;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

#[derive(Debug)]
pub struct Node {
    actions: RwLock<NodePtr>,
    num_actions: AtomicU8,
    state: AtomicU16,
    threads: AtomicU16,
    mov: AtomicU16,
    policy: AtomicU16,
    visits: AtomicI32,
    summed_q: AtomicI64,
    summed_sq_q: AtomicI64,
    gini_impurity: AtomicU32,
}

impl Node {
    pub fn new(state: GameState) -> Self {
        Node {
            actions: RwLock::new(NodePtr::NULL),
            num_actions: AtomicU8::new(0),
            state: AtomicU16::new(u16::from(state)),
            threads: AtomicU16::new(0),
            mov: AtomicU16::new(0),
            policy: AtomicU16::new(0),
            visits: AtomicI32::new(0),
            summed_q: AtomicI64::new(0),
            summed_sq_q: AtomicI64::new(0),
            gini_impurity: AtomicU32::new(0),
        }
    }

    pub fn set_new(&self, mov: Move, policy: f32) {
        self.clear();
        self.mov.store(u16::from(mov), Ordering::Relaxed);
        self.set_policy(policy);
    }

    pub fn is_terminal(&self) -> bool {
        self.state() != GameState::Ongoing
    }

    pub fn num_actions(&self) -> usize {
        usize::from(self.num_actions.load(Ordering::Relaxed))
    }

    pub fn set_num_actions(&self, num: usize) {
        self.num_actions.store(num as u8, Ordering::Relaxed);
    }

    pub fn threads(&self) -> u16 {
        self.threads.load(Ordering::Relaxed)
    }

    pub fn visits(&self) -> i32 {
        self.visits.load(Ordering::Relaxed)
    }

    fn q64(&self) -> f64 {
        let summed_q = self.summed_q.load(Ordering::Relaxed);
        let visits = self.visits.load(Ordering::Relaxed);
        (summed_q / i64::from(visits)) as f64 / f64::from(QUANT)
    }

    pub fn q(&self) -> f32 {
        let summed_q = self.summed_q.load(Ordering::Relaxed);
        let visits = self.visits.load(Ordering::Relaxed);

        if visits == 0 {
            return 0.0;
        }

        (summed_q / i64::from(visits)) as f32 / f32::from(QUANT)
    }

    pub fn sq_q(&self) -> f64 {
        let summed_sq_q = self.summed_q.load(Ordering::Relaxed);
        let visits = self.visits.load(Ordering::Relaxed);
        (summed_sq_q / i64::from(visits)) as f64 / f64::from(QUANT).powi(2)
    }

    pub fn var(&self) -> f32 {
        (self.sq_q() - self.q64().powi(2)).max(0.0) as f32
    }

    pub fn inc_threads(&self) {
        self.threads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_threads(&self) {
        self.threads.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn actions(&self) -> RwLockReadGuard<NodePtr> {
        self.actions.read().unwrap()
    }

    pub fn actions_mut(&self) -> RwLockWriteGuard<NodePtr> {
        self.actions.write().unwrap()
    }

    pub fn state(&self) -> GameState {
        GameState::from(self.state.load(Ordering::Relaxed))
    }

    pub fn set_state(&self, state: GameState) {
        self.state.store(u16::from(state), Ordering::Relaxed);
    }

    pub fn policy(&self) -> f32 {
        f32::from(self.policy.load(Ordering::Relaxed)) / f32::from(u16::MAX)
    }

    pub fn set_policy(&self, policy: f32) {
        self.policy
            .store((policy * f32::from(u16::MAX)) as u16, Ordering::Relaxed);
    }

    pub fn has_children(&self) -> bool {
        self.num_actions() != 0
    }

    pub fn is_not_expanded(&self) -> bool {
        self.state() == GameState::Ongoing && self.num_actions() == 0
    }

    pub fn gini_impurity(&self) -> f32 {
        f32::from_bits(self.gini_impurity.load(Ordering::Relaxed))
    }

    pub fn set_gini_impurity(&self, gini_impurity: f32) {
        self.gini_impurity
            .store(f32::to_bits(gini_impurity), Ordering::Relaxed);
    }

    pub fn clear_actions(&self) {
        *self.actions.write().unwrap() = NodePtr::NULL;
        self.num_actions.store(0, Ordering::Relaxed);
    }

    pub fn parent_move(&self) -> Move {
        Move::from(self.mov.load(Ordering::Relaxed))
    }

    pub fn copy_from(&self, other: &Self) {
        use std::sync::atomic::Ordering::Relaxed;

        self.threads.store(other.threads.load(Relaxed), Relaxed);
        self.mov.store(other.mov.load(Relaxed), Relaxed);
        self.policy.store(other.policy.load(Relaxed), Relaxed);
        self.state.store(other.state.load(Relaxed), Relaxed);
        self.gini_impurity
            .store(other.gini_impurity.load(Relaxed), Relaxed);
        self.visits.store(other.visits.load(Relaxed), Relaxed);
        self.summed_q.store(other.summed_q.load(Relaxed), Relaxed);
        self.summed_sq_q.store(other.summed_sq_q.load(Relaxed), Relaxed);
    }

    pub fn clear(&self) {
        self.clear_actions();
        self.set_state(GameState::Ongoing);
        self.set_gini_impurity(0.0);
        self.visits.store(0, Ordering::Relaxed);
        self.summed_q.store(0, Ordering::Relaxed);
        self.summed_sq_q.store(0, Ordering::Relaxed);
        self.threads.store(0, Ordering::Relaxed);
    }

    pub fn update(&self, q: f32) -> f32 {
        let q = (q * f32::from(QUANT)) as i64;
        let old_v = self.visits.fetch_add(1, Ordering::Relaxed);
        let old_q = self.summed_q.fetch_add(q, Ordering::Relaxed);
        self.summed_sq_q.fetch_add(q * q, Ordering::Relaxed);

        ((q + old_q) / i64::from(1 + old_v)) as f32 / f32::from(QUANT)
    }
}
