//! Test 31 — CLI argument parsing.

use std::process::Command;

fn binary_path() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    // target/debug/deps/cli-XXXX → target/debug/emamba
    exe.parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("emamba")
}

#[test]
fn help_flag_prints_usage() {
    let out = Command::new(binary_path())
        .arg("--help")
        .output()
        .expect("run binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("emamba"));
    assert!(stdout.contains("config"));
    assert!(stdout.contains("trace"));
}

#[test]
fn missing_args_exits_nonzero() {
    let out = Command::new(binary_path()).output().expect("run binary");
    assert!(!out.status.success());
}
