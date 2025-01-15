use crate::{
    ataxx::{Board, Move},
    mcts::{Limits, MctsParams, SearchHelpers, Searcher},
    networks::{policy, value},
    tree::Tree,
};

use std::{
    io, process,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn run() {
    let mut prev = None;
    let mut pos = Board::default();
    let mut params = MctsParams::default();
    let mut tree = Tree::new_mb(64, 1);
    let mut report_moves = false;
    let mut threads = 1;
    let mut move_overhead = 40;

    let mut stored_message: Option<String> = None;

    loop {
        let input = if let Some(msg) = stored_message {
            msg.clone()
        } else {
            let mut input = String::new();
            let bytes_read = io::stdin().read_line(&mut input).unwrap();

            if bytes_read == 0 {
                break;
            }

            input
        };

        stored_message = None;

        let commands = input.split_whitespace().collect::<Vec<_>>();

        let cmd = *commands.first().unwrap_or(&"oops");
        match cmd {
            "isready" => println!("readyok"),
            "setoption" => setoption(
                &commands,
                &mut params,
                &mut report_moves,
                &mut tree,
                &mut threads,
                &mut move_overhead,
            ),
            "position" => position(commands, &mut pos),
            "go" => {
                go(
                    &commands,
                    &mut tree,
                    prev,
                    &pos,
                    &params,
                    report_moves,
                    threads,
                    move_overhead,
                    &mut stored_message,
                );

                prev = Some(pos);
            }
            "quit" => std::process::exit(0),
            "params" => params.list_spsa(),
            "eval" => {
                let eval = 100.0 * value::get(&pos);
                println!("wdl = {eval:.2}%");
            }
            "policy" => {
                let f = policy::get_feats(&pos);
                let mut max = f32::NEG_INFINITY;
                let mut moves = Vec::new();

                pos.map_legal_moves(|mov| {
                    let s = mov.uai();
                    let p = policy::get(mov, &f);

                    if p > max {
                        max = p;
                    }

                    moves.push((s, p));
                });

                let mut total = 0.0;

                for (_, p) in &mut moves {
                    *p = (*p - max).exp();
                    total += *p;
                }

                moves.sort_by_key(|(_, p)| (p * 1000.0) as u32);

                for (s, p) in moves {
                    println!("{s} -> {:.2}%", p / total * 100.0);
                }
            }
            "uai" => preamble(),
            "uainewgame" => {
                prev = None;
                tree.clear(threads);
            }
            _ => {}
        }
    }
}

fn preamble() {
    println!("id name montyxx");
    println!("id author Jamie Whiting, Viren & The Monty Authors");
    println!("option name Hash type spin default 64 min 1 max 8192");
    println!("option name Threads type spin default 1 min 1 max 512");
    println!("option name uai_Chess960 type check default false");
    println!("option name MoveOverhead type spin default 40 min 0 max 5000");
    println!("option name report_moves type button");
    println!("uaiok");
}

fn setoption(
    commands: &[&str],
    params: &mut MctsParams,
    report_moves: &mut bool,
    tree: &mut Tree,
    threads: &mut usize,
    move_overhead: &mut usize,
) {
    if let ["setoption", "name", "report_moves"] = commands {
        *report_moves = !*report_moves;
        return;
    }

    let (name, val) = if let ["setoption", "name", x, "value", y] = commands {
        if *x == "uai_Chess960" {
            return;
        }

        if *x == "Threads" {
            *threads = y.parse().unwrap();
            return;
        }

        if *x == "MoveOverhead" {
            *move_overhead = y.parse().unwrap();
            return;
        }

        (*x, y.parse::<i32>().unwrap_or(0))
    } else {
        return;
    };

    if name == "Hash" {
        *tree = Tree::new_mb(val as usize, *threads);
    } else {
        params.set(name, val);
    }
}

fn position(commands: Vec<&str>, pos: &mut Board) {
    let mut fen = String::new();
    let mut move_list = Vec::new();
    let mut moves = false;

    for cmd in commands {
        match cmd {
            "position" | "fen" => {}
            "startpos" => fen = Board::STARTPOS.to_string(),
            "moves" => moves = true,
            _ => {
                if moves {
                    move_list.push(cmd);
                } else {
                    fen.push_str(&format!("{cmd} "));
                }
            }
        }
    }

    *pos = Board::from_fen(&fen);

    for &m in move_list.iter() {
        let mut this_mov = Move::default();

        pos.map_legal_moves(|mov| {
            if m == mov.uai() {
                this_mov = mov;
            }
        });

        pos.make(this_mov);
    }
}

#[allow(clippy::too_many_arguments)]
fn go(
    commands: &[&str],
    tree: &mut Tree,
    prev: Option<Board>,
    pos: &Board,
    params: &MctsParams,
    report_moves: bool,
    threads: usize,
    move_overhead: usize,
    stored_message: &mut Option<String>,
) {
    let mut max_nodes = i32::MAX as usize;
    let mut max_time = None;
    let mut max_depth = 256;

    let mut times = [None; 2];
    let mut incs = [None; 2];
    let mut movestogo = None;

    let mut mode = "";

    let saturating_parse = |s: &str| s.parse::<i64>().ok().map(|val| val.max(0) as u64);

    for cmd in commands {
        match *cmd {
            "nodes" => mode = "nodes",
            "movetime" => mode = "movetime",
            "depth" => mode = "depth",
            "wtime" => mode = "wtime",
            "btime" => mode = "btime",
            "winc" => mode = "winc",
            "binc" => mode = "binc",
            "movestogo" => mode = "movestogo",
            _ => match mode {
                "nodes" => max_nodes = cmd.parse().unwrap_or(max_nodes),
                "movetime" => max_time = cmd.parse().ok(),
                "depth" => max_depth = cmd.parse().unwrap_or(max_depth),
                "wtime" => times[1] = saturating_parse(cmd),
                "btime" => times[0] = saturating_parse(cmd),
                "winc" => incs[1] = saturating_parse(cmd),
                "binc" => incs[0] = saturating_parse(cmd),
                "movestogo" => movestogo = saturating_parse(cmd),
                _ => mode = "none",
            },
        }
    }

    // `go wtime <wtime> btime <btime> winc <winc> binc <binc>``
    if let Some(remaining) = times[pos.stm()] {
        let timeman = SearchHelpers::get_time(remaining, incs[pos.stm()], movestogo, params);

        max_time = Some(timeman);
    }

    // `go movetime <time>`
    if let Some(max) = max_time {
        // if both movetime and increment time controls given, use
        max_time = Some(max_time.unwrap_or(u128::MAX).min(max));
    }

    // apply move overhead
    if let Some(t) = max_time.as_mut() {
        *t = t.saturating_sub(move_overhead as u128);
    }

    let abort = AtomicBool::new(false);

    tree.try_use_subtree(pos, &prev);

    let limits = Limits {
        max_time,
        max_depth,
        max_nodes,
    };

    std::thread::scope(|s| {
        s.spawn(|| {
            let searcher = Searcher::new(*pos, tree, params, &abort);
            let (mov, _) = searcher.search(threads, limits, true, &mut 0, None);
            println!("bestmove {}", mov.uai());

            if report_moves {
                searcher.display_moves();
            }
        });

        *stored_message = handle_search_input(&abort);
    });
}

fn handle_search_input(abort: &AtomicBool) -> Option<String> {
    loop {
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input).unwrap();

        if bytes_read == 0 {
            process::exit(0);
        }

        match input.as_str().trim() {
            "isready" => println!("readyok"),
            "quit" => std::process::exit(0),
            "stop" => {
                abort.store(true, Ordering::Relaxed);
                return None;
            }
            _ => return Some(input),
        };
    }
}
