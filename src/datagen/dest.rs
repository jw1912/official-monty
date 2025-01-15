use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::atomic::{AtomicBool, Ordering},
};

use super::format::MontyAtaxxFormat;

pub struct Destination {
    writer: BufWriter<File>,
    reusable_buffer: Vec<u8>,
    games: usize,
    limit: usize,
    results: [usize; 3],
}

impl Destination {
    pub fn new(path: &str, limit: usize) -> Self {
        let out = File::create(path).unwrap();
        let writer = BufWriter::new(out);

        Self {
            writer,
            reusable_buffer: Vec::new(),
            games: 0,
            limit,
            results: [0; 3],
        }
    }

    pub fn push(&mut self, game: &MontyAtaxxFormat, stop: &AtomicBool) {
        if stop.load(Ordering::Relaxed) {
            return;
        }

        let result = (game.result * 2.0) as usize;
        self.results[result] += 1;
        self.games += 1;

        game.serialise_into_buffer(&mut self.reusable_buffer)
            .unwrap();
        self.writer.write_all(&self.reusable_buffer).unwrap();
        self.reusable_buffer.clear();

        if self.games >= self.limit {
            stop.store(true, Ordering::Relaxed);
            return;
        }

        if self.games % 32 == 0 {
            self.report();
        }
    }

    pub fn report(&self) {
        println!(
            "finished games {} losses {} draws {} wins {}",
            self.games, self.results[0], self.results[1], self.results[2],
        )
    }
}
