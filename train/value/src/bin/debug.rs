use std::{
    fs::File,
    io::BufReader,
};

use datagen::Binpack;
use monty::{Board, Castling, ChessState};

fn debug(inp_path: &str) {
    let mut reader = BufReader::new(File::open(inp_path).unwrap());
    let mut prev = Binpack::new(ChessState::default());

    let mut count = 0;

    loop {
        count += 1;
        let buffer = Vec::with_capacity(128);
        let res = Binpack::deserialise_from(&mut reader, buffer);

        if let Ok(binpack) = res {
            let startpos = binpack.startpos;
            let pos = Board::from(startpos);
            let rook_files = startpos.rook_files();

            if Castling::from_raw(&pos, rook_files).is_none() {
                println!("Game No. {count}");
                println!("{prev}");
                println!("{}", prev.as_uci_position_command().unwrap());

                println!("{binpack}");
            }

            prev = binpack;
        } else {
            break;
        }
    }

    println!("Count: {count}");
}

fn main() {
    let paths = std::fs::read_dir("./data").unwrap();
    let mut present = [false; 241];

    for path in paths {
        let pb = path.unwrap().path();
        let p = pb.to_str().unwrap();
        println!("Name: {p}");
        debug(p);

        let a = p.split('-').nth(1).unwrap();
        present[a.split('.').next().unwrap().parse::<usize>().unwrap()] = true;
    }

    for (i, p) in present.iter().enumerate() {
        assert!(p, "{i} not present!");
    }
}
