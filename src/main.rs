use monty::{datagen, mcts::MctsParams, uai};

fn main() {
    let mut args = std::env::args();
    args.next();

    if let Some("datagen") = args.next().as_deref() {
        let params = MctsParams::default();
        let opts = datagen::parse_args(args).unwrap();
        datagen::run_datagen(params, opts);
        return;
    }

    uai::run();
}
