use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::mpsc::{self, SyncSender},
    time::{SystemTime, UNIX_EPOCH},
};

use monty::{ataxx::Move, datagen::MontyAtaxxFormat};
use monty::ataxx::Board;

use crate::moves::MAX_MOVES;

#[derive(Clone, Copy)]
pub struct DecompressedData {
    pub pos: Board,
    pub moves: [(Move, u16); MAX_MOVES],
    pub num: usize,
}

#[derive(Clone)]
pub struct DataLoader {
    file_path: [String; 1],
    buffer_size: usize,
    threads: usize,
}

impl DataLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self {
            file_path: [path.to_string(); 1],
            buffer_size: buffer_size_mb * 1024 * 1024 / 512 / 2,
            threads,
        }
    }
}

impl bullet::default::loader::DataLoader<DecompressedData> for DataLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_path
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[DecompressedData]) -> bool>(
        &self,
        _: usize,
        batch_size: usize,
        mut f: F,
    ) {
        let file_path = self.file_path[0].clone();
        let buffer_size = self.buffer_size;

        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(256);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || 'dataloading: loop {
            let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

            let mut buffer = Vec::new();
            while let Ok(()) = MontyAtaxxFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                if msg_receiver.try_recv().unwrap_or(false) || sender.send(buffer).is_err() {
                    break 'dataloading;
                }

                buffer = Vec::new();
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let threads = self.threads;

        std::thread::spawn(move || {
            let mut reusable = Vec::new();
            'dataloading: while let Ok(game_bytes) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                reusable.push(game_bytes);

                if reusable.len() % (8192 * threads) == 0 {
                    convert_buffer(threads, &game_sender, &reusable);
                    reusable.clear();
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::new();
            shuffle_buffer.reserve_exact(buffer_size);

            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    game_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    shuffle_buffer.extend_from_slice(&game[..diff]);

                    shuffle(&mut shuffle_buffer);

                    if buffer_msg_receiver.try_recv().unwrap_or(false) || buffer_sender.send(shuffle_buffer).is_err() {
                        game_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                }
            }
        });

        'dataloading: while let Ok(inputs) = buffer_receiver.recv() {
            for batch in inputs.chunks(batch_size) {
                let should_break = f(batch);

                if should_break {
                    buffer_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        }

        drop(buffer_receiver);
    }
}

fn shuffle(data: &mut [DecompressedData]) {
    let mut rng = Rand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

fn convert_buffer(threads: usize, sender: &SyncSender<Vec<DecompressedData>>, games: &[Vec<u8>]) {
    let chunk_size = games.len().div_ceil(threads);
    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut rng = Rand::with_seed();
                let mut buffer = Vec::new();

                for game_bytes in chunk {
                    let mut reader = Cursor::new(game_bytes);
                    let game = MontyAtaxxFormat::deserialise_from(&mut reader).unwrap();
                    parse_into_buffer(game, &mut buffer, &mut rng);
                }

                this_sender.send(buffer)
            });
        }
    });
}

fn parse_into_buffer(game: MontyAtaxxFormat, buffer: &mut Vec<DecompressedData>, rng: &mut Rand) {
    let mut pos = Board::default();

    for (i, data) in game.moves.iter().enumerate() {
        if let Some(dist) = data.visit_distribution.as_ref() {
            if i >= 10 && dist.len() > 1 && dist.len() <= MAX_MOVES {
                let trans = [
                    Transform::Identity,
                    Transform::Horizontal,
                    Transform::Vertical,
                    Transform::Rotational,
                    Transform::Diagonal,
                    Transform::AntiDiagonal,
                ][rng.rng() as usize % 6];

                let mut policy_data = DecompressedData {
                    pos,
                    moves: [(Move::NULL, 0); MAX_MOVES],
                    num: dist.len(),
                };

                policy_data.pos.bbs[0] = transform_bb(policy_data.pos.bbs[0], trans);
                policy_data.pos.bbs[1] = transform_bb(policy_data.pos.bbs[1], trans);
                policy_data.pos.gaps = transform_bb(policy_data.pos.gaps, trans);

                for (i, &(mut mov, visits)) in dist.iter().enumerate() {
                    if !mov.is_pass() {
                        mov = if mov.is_single() {
                            Move::new_single(transform_sq(mov.to() as u8, trans))
                        } else {
                            Move::new_double(
                                transform_sq(mov.src() as u8, trans),
                                transform_sq(mov.to() as u8, trans),
                            )
                        }
                    }

                    policy_data.moves[i] = (mov, visits as u16);
                }


                buffer.push(policy_data);
            }
        }

        pos.make(data.best_move);
    }
}

pub struct Rand(u64);

impl Rand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u64
            & 0xFFFF_FFFF;

        Self(seed)
    }

    pub fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[derive(Clone, Copy)]
enum Transform {
    Identity,
    Horizontal,
    Vertical,
    Rotational,
    Diagonal,
    AntiDiagonal
}

fn transform_bb(bb: u64, trans: Transform) -> u64 {
    match trans {
        Transform::Identity => bb,
        Transform::Horizontal => flip_hor(bb),
        Transform::Vertical => flip_vert(bb),
        Transform::Rotational => flip_hor(flip_vert(bb)),
        Transform::Diagonal => flip_diag(bb),
        Transform::AntiDiagonal => flip_vert(flip_hor(flip_diag(bb))),
    }
}

fn transform_sq(sq: u8, trans: Transform) -> u8 {
    transform_bb(1 << sq, trans).trailing_zeros() as u8
}

fn flip_vert(bb: u64) -> u64 {
    const RANK: u64 = 127;
    let mut out = 0;

    for rank in 0..7 {
        let iso = (bb >> (7 * rank)) & RANK;
        out |= iso << (7 * (6 - rank));
    }

    out
}

fn flip_hor(bb: u64) -> u64 {
    const FILE: u64 = 4432676798593;
    let mut out = 0;

    for file in 0..7 {
        let iso = (bb >> file) & FILE;
        out |= iso << (6 - file);
    }

    out
}

fn flip_diag(bb: u64) -> u64 {
    const RANK: u64 = 127;
    let mut out = 0;

    for rank in 0..7 {
        let mut iso = (bb >> (7 * rank)) & RANK;
        let mut file = 0;

        while iso > 0 {
            file |= 1 << (iso.trailing_zeros() * 7);
            iso &= iso - 1;
        }

        out |= file << rank;
    }

    out
}
