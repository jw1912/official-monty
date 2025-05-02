use monty::{
    chess::ChessState, mcts::MctsParams, uci,
};

fn main() {
    let mut args = std::env::args();
    let arg1 = args.nth(1);
    if let Some("bench") = arg1.as_deref() {
        uci::bench(
            ChessState::BENCH_DEPTH,
            &MctsParams::default(),
        );
        return;
    }

    uci::run();
}
