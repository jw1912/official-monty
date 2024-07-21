#[derive(Clone, Copy, Debug)]
pub struct Edge {
    ptr: i32,
    mov: u16,
    policy: i16,
    visits: i32,
    q: f32,
    sq_q: f32,
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            ptr: -1,
            mov: 0,
            policy: 0,
            visits: 0,
            q: 0.0,
            sq_q: 0.0,
        }
    }
}

impl Edge {
    pub fn new(ptr: i32, mov: u16, policy: i16) -> Self {
        Self {
            ptr,
            mov,
            policy,
            visits: 0,
            q: 0.0,
            sq_q: 0.0,
        }
    }

    pub fn ptr(&self) -> i32 {
        self.ptr
    }

    pub fn mov(&self) -> u16 {
        self.mov
    }

    pub fn policy(&self) -> f32 {
        f32::from(self.policy) / f32::from(i16::MAX)
    }

    pub fn visits(&self) -> i32 {
        self.visits
    }

    pub fn q(&self) -> f32 {
        self.q
    }

    pub fn var(&self) -> f32 {
        let var = self.sq_q - self.q.powi(2);
        var.max(0.0)
    }

    pub fn set_ptr(&mut self, ptr: i32) {
        self.ptr = ptr;
    }

    pub fn set_policy(&mut self, policy: f32) {
        self.policy = (policy * f32::from(i16::MAX)) as i16
    }

    pub fn update(&mut self, result: f32) {
        let v = f64::from(self.visits);
        self.visits += 1;
        self.q = ((f64::from(self.q) * v + f64::from(result)) / (v + 1.0)) as f32;
        self.sq_q = ((f64::from(self.sq_q) * v + f64::from(result.powi(2))) / (v + 1.0)) as f32
    }
}
