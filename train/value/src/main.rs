fn main() {
    let mut args = std::env::args();
    args.next();
    let buffer_size_mb = args.next().unwrap().parse().unwrap();
    let threads = args.next().unwrap().parse().unwrap();

    value::train(
        buffer_size_mb,
        threads,
        "../binpacks/new-data.binpack".to_string(),
        40,
        0.001,
        0.1,
        18,
    );
}
