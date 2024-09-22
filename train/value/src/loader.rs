use std::{fs::File, io::BufReader};

use montyformat::{chess::Position, MontyValueFormat};

use crate::rand::Rand;

pub struct DataLoader {
    file_path: String,
    buffer_size: usize,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(path: &str, buffer_size_mb: usize, batch_size: usize) -> Self {
        Self {
            file_path: path.to_string(),
            buffer_size: buffer_size_mb * 1024 * 1024 / 128,
            batch_size,
        }
    }

    pub fn map_batches<F: FnMut(&[(Position, f32)]) -> bool>(&self, mut f: F) {
        let mut reusable_buffer = Vec::new();

        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        'dataloading: loop {
            let mut reader = BufReader::new(File::open(self.file_path.as_str()).unwrap());

            while let Ok(game) = MontyValueFormat::deserialise_from(&mut reader, Vec::new()) {
                parse_into_buffer(game, &mut reusable_buffer);

                if shuffle_buffer.len() + reusable_buffer.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&reusable_buffer);
                } else {
                    println!("#[Shuffling]");
                    shuffle(&mut shuffle_buffer);

                    println!("#[Running Batches]");
                    for batch in shuffle_buffer.chunks(self.batch_size) {
                        let should_break = f(batch);

                        if should_break {
                            break 'dataloading;
                        }
                    }

                    println!();
                    shuffle_buffer.clear();
                }
            }
        }
    }
}

fn shuffle(data: &mut [(Position, f32)]) {
    let mut rng = Rand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rand_int() as usize % (i + 1);
        data.swap(idx, i);
    }
}

fn parse_into_buffer(game: MontyValueFormat, buffer: &mut Vec<(Position, f32)>) {
    buffer.clear();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        if data.score.abs() < 2000 && !pos.in_check() && !data.best_move.is_capture() {
            buffer.push((pos, 1.0 / (1.0 + (-f32::from(data.score) / 400.0).exp())));
        }

        pos.make(data.best_move, &castling);
    }
}
