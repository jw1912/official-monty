fn main() {
    let mut args = std::env::args();
    args.next();
    let buffer_size_mb = args.next().unwrap().parse().unwrap();
    let threads = args.next().unwrap().parse().unwrap();

    policy::train(
        buffer_size_mb,
        threads,
        "../binpacks/policygen8.binpack".to_string(),
        40,
        0.001,
        0.00001,
        40,
    );
}
