mod dest;
mod format;
mod thread;

use std::{
    env::Args,
    sync::{atomic::AtomicBool, Arc, Mutex},
    time::Duration,
};

use dest::Destination;
use thread::DatagenThread;

pub use format::{MontyAtaxxFormat, SearchData};

use crate::mcts::MctsParams;

pub fn run_datagen(params: MctsParams, opts: RunOptions) {
    let stop_base = AtomicBool::new(false);
    let stop = &stop_base;

    let dest = Destination::new(&opts.out_path, opts.games);
    let dst = Arc::new(Mutex::new(dest));

    std::thread::scope(|s| {
        for _ in 0..opts.threads {
            let params = params.clone();
            std::thread::sleep(Duration::from_millis(10));
            let this_dest = dst.clone();
            s.spawn(move || {
                let mut thread = DatagenThread::new(params.clone(), stop, this_dest, opts.nodes);
                thread.run();
            });
        }
    });

    let dest = dst.lock().unwrap();

    dest.report();
}

#[derive(Debug, Default)]
pub struct RunOptions {
    games: usize,
    threads: usize,
    nodes: usize,
    out_path: String,
}

pub fn parse_args(args: Args) -> Option<RunOptions> {
    let mut opts = RunOptions::default();

    let mut mode = 0;

    for arg in args {
        match arg.as_str() {
            "bench" => return None,
            "-t" | "--threads" => mode = 1,
            "-n" | "--nodes" => mode = 2,
            "-o" | "--output" => mode = 3,
            "-g" | "--games" => mode = 4,
            _ => match mode {
                1 => {
                    opts.threads = arg.parse().expect("can't parse");
                    mode = 0;
                }
                2 => {
                    opts.nodes = arg.parse().expect("can't parse");
                    mode = 0;
                }
                3 => {
                    opts.out_path = arg;
                    mode = 0;
                }
                4 => {
                    opts.games = arg.parse().expect("can't parse");
                    mode = 0;
                }
                _ => println!("unrecognised argument {arg}"),
            },
        }
    }

    Some(opts)
}
