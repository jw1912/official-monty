use chrono::Utc;
use std::process::Command;

fn main() {
    // Get the current Git commit hash
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("Failed to execute git command");

    let git_commit_hash = String::from_utf8(output.stdout)
        .expect("Git output was not valid UTF-8")
        .trim()
        .to_string();

    // Get the current date in YYYYMMDD format
    let current_date = Utc::now().format("%Y%m%d").to_string();

    // Combine into the desired format
    let formatted_name = format!("Monty-dev-{}-{}", current_date, &git_commit_hash[..8]);

    // Pass the formatted name as an environment variable
    println!("cargo:rustc-env=FORMATTED_NAME={}", formatted_name);
}

