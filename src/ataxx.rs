use std::{cmp::Ordering, fmt::Display};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GameState {
    #[default]
    Ongoing,
    Lost(u8),
    Draw,
    Won(u8),
}

impl From<GameState> for u16 {
    fn from(value: GameState) -> Self {
        match value {
            GameState::Ongoing => 0,
            GameState::Draw => 1 << 8,
            GameState::Lost(x) => (2 << 8) ^ u16::from(x),
            GameState::Won(x) => (3 << 8) ^ u16::from(x),
        }
    }
}

impl From<u16> for GameState {
    fn from(value: u16) -> Self {
        let discr = value >> 8;
        let x = value as u8;

        match discr {
            0 => GameState::Ongoing,
            1 => GameState::Draw,
            2 => GameState::Lost(x),
            3 => GameState::Won(x),
            _ => unreachable!(),
        }
    }
}

impl std::fmt::Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameState::Ongoing => write!(f, "O"),
            GameState::Lost(n) => write!(f, "L{n}"),
            GameState::Won(n) => write!(f, "W{n}"),
            GameState::Draw => write!(f, "D"),
        }
    }
}

#[macro_export]
macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Board {
    bbs: [u64; 2],
    gaps: u64,
    stm: bool,
    halfm: u8,
    fullm: u16,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(Self::STARTPOS)
    }
}

impl Board {
    pub const STARTPOS: &str = "x5o/7/7/7/7/7/o5x x 0 1";

    pub fn stm(&self) -> usize {
        usize::from(self.stm)
    }

    pub fn occ(&self) -> u64 {
        self.bbs[0] | self.bbs[1] | self.gaps
    }

    pub fn bbs(&self) -> [u64; 3] {
        [self.bbs[0], self.bbs[1], self.gaps]
    }

    pub fn halfm(&self) -> u8 {
        self.halfm
    }

    pub fn boys(&self) -> u64 {
        self.bbs[self.stm()]
    }

    pub fn opps(&self) -> u64 {
        self.bbs[self.stm() ^ 1]
    }

    pub fn fullm(&self) -> u16 {
        self.fullm
    }

    pub fn is_hfm_draw(&self, count: u8) -> bool {
        self.halfm() >= count
    }

    pub fn make(&mut self, mov: Move) {
        if !mov.is_pass() {
            let stm = self.stm();
            let from = mov.src();
            let to = mov.to();

            self.fullm += u16::from(stm == Side::BLU);

            if from != 63 {
                self.bbs[stm] ^= 1 << from;
                self.halfm += 1;
            } else {
                self.halfm = 0;
            }

            self.bbs[stm] ^= 1 << to;

            let singles = Bitboard::singles(to);
            let captures = singles & self.bbs[stm ^ 1];

            self.bbs[0] ^= captures;
            self.bbs[1] ^= captures;
        }

        self.stm = !self.stm;
    }

    pub fn game_over(&self) -> bool {
        let bocc = self.bbs[Side::BLU].count_ones();
        let rocc = self.bbs[Side::RED].count_ones();
        bocc == 0 || rocc == 0 || bocc + rocc == 49 || self.is_hfm_draw(100)
    }

    pub fn hash(&self) -> u64 {
        let mut hash = 0;

        let mut boys = self.boys();
        while boys > 0 {
            let sq = boys.trailing_zeros() as usize;
            boys &= boys - 1;

            hash ^= ZVALS[0][sq];
        }

        let mut opps = self.opps();
        while opps > 0 {
            let sq = opps.trailing_zeros() as usize;
            opps &= opps - 1;

            hash ^= ZVALS[1][sq];
        }

        hash
    }

    pub fn material(&self) -> i32 {
        let socc = self.boys().count_ones();
        let nocc = self.opps().count_ones();

        socc as i32 - nocc as i32
    }

    pub fn game_state(&self) -> GameState {
        let socc = self.boys().count_ones();
        let nocc = self.opps().count_ones();

        if socc + nocc == 49 {
            match socc.cmp(&nocc) {
                Ordering::Greater => GameState::Won(0),
                Ordering::Less => GameState::Lost(0),
                Ordering::Equal => GameState::Draw,
            }
        } else if socc == 0 {
            GameState::Lost(0)
        } else if nocc == 0 {
            GameState::Won(0)
        } else if self.is_hfm_draw(100) {
            GameState::Draw
        } else {
            GameState::Ongoing
        }
    }

    pub fn map_legal_moves<F: FnMut(Move)>(&self, mut f: F) {
        if self.game_over() {
            return;
        }

        let occ = self.occ();
        let nocc = Bitboard::not(occ);
        let mut boys = self.boys();
        let mut singles = Bitboard::expand(boys) & nocc;

        let mut num = singles.count_ones();

        while singles > 0 {
            let sq = singles.trailing_zeros();
            singles &= singles - 1;

            f(Move::new_single(sq as u8));
        }

        while boys > 0 {
            let from = boys.trailing_zeros();
            boys &= boys - 1;

            let mut doubles = Bitboard::doubles(from as usize) & nocc;

            while doubles > 0 {
                let to = doubles.trailing_zeros();
                doubles &= doubles - 1;

                num += 1;
                f(Move::new_double(from as u8, to as u8));
            }
        }

        if num == 0 {
            f(Move::new_pass());
        }
    }

    pub fn value_features_map<F: FnMut(usize)>(&self, mut f: F) {
        const PER_TUPLE: usize = 3usize.pow(4);
        const POWERS: [usize; 4] = [1, 3, 9, 27];
        const MASK: u64 = 0b0001_1000_0011;

        let boys = self.boys();
        let opps = self.opps();

        for i in 0..6 {
            for j in 0..6 {
                let tuple = 6 * i + j;
                let mut feat = PER_TUPLE * tuple;

                let offset = 7 * i + j;
                let mut b = (boys >> offset) & MASK;
                let mut o = (opps >> offset) & MASK;

                while b > 0 {
                    let mut sq = b.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += POWERS[sq];

                    b &= b - 1;
                }

                while o > 0 {
                    let mut sq = o.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += 2 * POWERS[sq];

                    o &= o - 1;
                }

                f(feat);
            }
        }
    }

    pub fn movegen_bulk(&self, pass: bool) -> u64 {
        let mut moves = u64::from(pass);

        let occ = self.occ();
        let nocc = Bitboard::not(occ);
        let mut boys = self.boys();

        let singles = Bitboard::expand(boys) & nocc;
        moves += u64::from(singles.count_ones());

        while boys > 0 {
            let from = boys.trailing_zeros();
            boys &= boys - 1;

            let doubles = Bitboard::doubles(from as usize) & nocc;
            moves += u64::from(doubles.count_ones());
        }

        moves
    }

    pub fn as_fen(&self) -> String {
        let mut fen = String::new();

        let occ = self.occ();

        let mut empty = 0;

        for rank in (0..7).rev() {
            for file in 0..7 {
                let sq = 7 * rank + file;
                let bit = 1 << sq;

                if occ & bit > 0 {
                    if empty > 0 {
                        fen += format!("{empty}").as_str();
                        empty = 0;
                    }

                    fen += if bit & self.bbs[Side::RED] > 0 {
                        "x"
                    } else if bit & self.bbs[Side::BLU] > 0 {
                        "o"
                    } else {
                        "-"
                    };
                } else {
                    empty += 1;
                }
            }

            if empty > 0 {
                fen += format!("{empty}").as_str();
                empty = 0;
            }

            if rank > 0 {
                fen += "/";
            }
        }

        fen += [" x", " o"][usize::from(self.stm)];
        fen += format!(" {}", self.halfm).as_str();
        fen += format!(" {}", self.fullm).as_str();

        fen
    }

    pub fn from_fen(fen: &str) -> Self {
        let split: Vec<_> = fen.split_whitespace().collect();

        let rows = split[0].split('/').collect::<Vec<_>>();
        let stm = split[1] == "o";
        let halfm = split.get(2).map(|x| x.parse().unwrap_or(0)).unwrap_or(0);
        let fullm = split.get(3).map(|x| x.parse().unwrap_or(1)).unwrap_or(1);

        let mut bbs = [0; 2];
        let mut gaps = 0;
        let mut sq = 0;

        for row in rows.iter().rev() {
            for mut ch in row.chars() {
                ch = ch.to_ascii_lowercase();
                if ('1'..='7').contains(&ch) {
                    sq += ch.to_string().parse().unwrap_or(0);
                } else if let Some(pc) = "xo".chars().position(|el| el == ch) {
                    bbs[pc] |= 1 << sq;
                    sq += 1;
                } else if ch == '-' {
                    gaps |= 1 << sq;
                    sq += 1;
                }
            }
        }

        Self {
            bbs,
            gaps,
            stm,
            halfm,
            fullm,
        }
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..7).rev() {
            for file in 0..7 {
                let sq = 7 * rank + file;
                let bit = 1 << sq;

                let add = if bit & self.bbs[Side::RED] > 0 {
                    " x"
                } else if bit & self.bbs[Side::BLU] > 0 {
                    " o"
                } else if bit & self.gaps > 0 {
                    " -"
                } else {
                    " ."
                };

                write!(f, "{add}")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

pub struct Side;
impl Side {
    pub const RED: usize = 0;
    pub const BLU: usize = 1;
}

pub struct Bitboard;
impl Bitboard {
    pub const ALL: u64 = 0x1_ffff_ffff_ffff;
    pub const NOTR: u64 = 0xfdfb_f7ef_dfbf;
    pub const NOTL: u64 = 0x1_fbf7_efdf_bf7e;

    pub const fn expand(bb: u64) -> u64 {
        let right = (bb & Self::NOTR) << 1;
        let left = (bb & Self::NOTL) >> 1;

        let bb2 = bb | right | left;

        let up = (bb2 << 7) & Self::ALL;
        let down = bb2 >> 7;

        right | left | up | down
    }

    pub const fn not(bb: u64) -> u64 {
        !bb & Self::ALL
    }

    pub fn singles(sq: usize) -> u64 {
        SINGLES[sq]
    }

    pub fn doubles(sq: usize) -> u64 {
        DOUBLES[sq]
    }
}

static SINGLES: [u64; 49] = {
    let mut res = [0; 49];
    let mut sq = 0;

    while sq < 49 {
        res[sq] = Bitboard::expand(1 << sq);
        sq += 1;
    }

    res
};

static DOUBLES: [u64; 49] = {
    let mut res = [0; 49];
    let mut sq = 0;

    while sq < 49 {
        let bb = 1 << sq;

        let singles = Bitboard::expand(bb);
        res[sq] = Bitboard::expand(singles) & Bitboard::not(singles);

        sq += 1;
    }

    res
};

const fn rand(mut seed: u64) -> u64 {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    seed
}

pub static ZVALS: [[u64; 49]; 2] = {
    let mut seed = 180_620_142;
    seed = rand(seed);

    init!(|side, 2| init!(|sq, 49| {
        seed = rand(seed);
        seed
    }))
};

#[repr(C, align(2))]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Move {
    from: u8,
    to: u8,
}

impl From<Move> for u16 {
    fn from(value: Move) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u16> for Move {
    fn from(value: u16) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.uai())
    }
}

impl Move {
    pub const NULL: Self = Self::new_null();

    pub fn new_single(to: u8) -> Self {
        Self { from: 63, to }
    }

    pub fn new_double(from: u8, to: u8) -> Self {
        Self { from, to }
    }

    pub fn new_pass() -> Self {
        Self { from: 63, to: 63 }
    }

    pub fn is_single(&self) -> bool {
        self.from == 63
    }

    pub const fn new_null() -> Self {
        Self { from: 0, to: 0 }
    }

    pub fn is_null(&self) -> bool {
        self.from == 0 && self.to == 0
    }

    pub fn src(&self) -> usize {
        usize::from(self.from)
    }

    pub fn to(&self) -> usize {
        usize::from(self.to)
    }

    pub fn is_pass(&self) -> bool {
        self.to == 63
    }

    pub fn uai(&self) -> String {
        let mut res = String::new();
        let chs = ('a'..'h').collect::<Vec<_>>();

        if self.src() != 63 {
            res += chs[self.src() % 7].to_string().as_str();
            res += format!("{}", 1 + self.src() / 7).as_str()
        }

        if self.to() != 63 {
            res += chs[self.to() % 7].to_string().as_str();
            res += format!("{}", 1 + self.to() / 7).as_str()
        } else {
            res += "0000"
        }

        res
    }
}
