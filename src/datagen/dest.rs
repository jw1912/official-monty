use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use super::format::MontyAtaxxFormat;

pub struct Destination {
    writer: BufWriter<File>,
    reusable_buffer: Vec<u8>,
    games: usize,
    positions: usize,
    bytes: usize,
    limit: usize,
    results: [usize; 3],
    timer: Instant,
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
            positions: 0,
            bytes: 0,
            results: [0; 3],
            timer: Instant::now(),
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

        self.positions += game.moves.len();
        self.bytes += self.reusable_buffer.len();
        self.reusable_buffer.clear();

        if self.games >= self.limit {
            stop.store(true, Ordering::Relaxed);
            return;
        }

        if self.games % 256 == 0 {
            self.report();
        }
    }

    pub fn report(&self) {
        let bpp = self.bytes / self.positions;
        let elapsed = self.timer.elapsed().as_secs() as usize;
        let gph = 60 * 60 * self.games / elapsed;
        let pph = 60 * 60 * self.positions / elapsed;
        let mpg = self.positions / self.games;

        println!(
            "finished games {} losses {} draws {} wins {} bytes/pos {bpp} games/hr {gph} pos/hr {pph} moves/game {mpg}",
            self.games, self.results[0], self.results[1], self.results[2],
        )
    }
}
