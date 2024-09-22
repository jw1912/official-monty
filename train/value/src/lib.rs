mod arch;
mod loader;
mod rand;

use arch::Network;
use loader::DataLoader;
use montyformat::chess::{Castling, Position};

use std::{io::Write, time::Instant};

const BATCH_SIZE: usize = 16_384;
const BPSB: usize = 128;

pub fn train(
    buffer_size_mb: usize,
    threads: usize,
    data_path: String,
    superbatches: usize,
    lr_start: f32,
    lr_gamma: f32,
    lr_drop: usize,
) {
    let mut network = Network::rand_init();

    let mut lr = lr_start;
    let mut momentum = Network::boxed_and_zeroed();
    let mut velocity = Network::boxed_and_zeroed();

    let mut running_error = 0.0;
    let mut sb = 0;
    let mut batch_no = 0;

    let data_loader = DataLoader::new(data_path.as_str(), buffer_size_mb, BATCH_SIZE);

    let mut t = Instant::now();

    data_loader.map_batches(|batch| {
        let t2 = Instant::now();

        let mut grad = Network::boxed_and_zeroed();

        let loss = gradient_batch(threads, &network, &mut grad, batch);

        let adj = 1.0 / batch.len() as f32;
        network.update(&grad, &mut momentum, &mut velocity, adj, lr);

        let elapsed = t2.elapsed().as_secs_f32();

        batch_no += 1;
        running_error += loss;

        print!(
            "> Superbatch {}/{superbatches} Batch {}/{BPSB} Pos/Sec {:.0}k\r",
            sb + 1,
            batch_no % BPSB,
            batch.len() as f32 / elapsed
        );
        let _ = std::io::stdout().flush();

        if batch_no % BPSB == 0 {
            let elapsed = t.elapsed().as_secs_f32();
            t = Instant::now();

            sb += 1;
            println!(
                "> Superbatch {sb}/{superbatches} Running Loss {} Time {:.2}s",
                running_error / (BPSB * BATCH_SIZE) as f32,
                elapsed,
            );

            let mut seconds_left = ((superbatches - sb) as f32 * elapsed) as u64;
            let mut minutes_left = seconds_left / 60;
            seconds_left -= minutes_left * 60;
            let hours_left = minutes_left / 60;
            minutes_left -= hours_left * 60;

            println!("Estimated {hours_left}h {minutes_left}m {seconds_left}s Left in Training",);

            running_error = 0.0;

            if sb % lr_drop == 0 {
                lr *= lr_gamma;
                println!("Dropping LR to {lr}");
            }

            if sb % 10 == 0 {
                network.write_to_bin(format!("checkpoints/value-{sb}.bin").as_str());
            }

            sb == superbatches
        } else {
            false
        }
    });

    let mut g = Network::boxed_and_zeroed();

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    ] {
        let mut castling = Castling::default();
        let position = Position::parse_fen(fen, &mut castling);

        println!("FEN: {fen}");
        network.update_single_grad(&(position, 0.5), &mut g, &mut 0.0, true);
    }
}

fn gradient_batch(
    threads: usize,
    network: &Network,
    grad: &mut Network,
    batch: &[(Position, f32)],
) -> f32 {
    let size = (batch.len() / threads).max(1);
    let mut errors = vec![0.0; threads];

    std::thread::scope(|s| {
        batch
            .chunks(size)
            .zip(errors.iter_mut())
            .map(|(chunk, err)| {
                s.spawn(move || {
                    let mut inner_grad = Network::boxed_and_zeroed();
                    for pos in chunk {
                        network.update_single_grad(pos, &mut inner_grad, err, false);
                    }
                    inner_grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| grad.add_without_explicit_lifetime(&part));
    });

    errors.iter().sum::<f32>()
}
