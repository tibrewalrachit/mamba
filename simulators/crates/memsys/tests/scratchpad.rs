//! Tests 20-22 — `memsys::scratchpad::*`
//! Banked SRAM with per-bank read/write ports.
//! - Two writes to the same bank in the same cycle: second is rejected.
//! - Different banks: both succeed.
//! - Write at t=0, read at t=1, same bank: succeeds (port freed).

use emamba_memsys::scratchpad::BankedScratchpad;

#[test]
fn bank_port_serializes_two_writes() {
    let mut sp = BankedScratchpad::new(/* banks */ 4, /* bank_bytes */ 1024, /* port_width */ 16);

    let now = 0;
    let first = sp.try_write(/* bank */ 0, /* bytes */ 16, now);
    let second = sp.try_write(/* bank */ 0, /* bytes */ 16, now);

    assert_eq!(first, Some(1), "first write completes at now+1");
    assert_eq!(second, None, "second write to same bank in same cycle rejected");
}

#[test]
fn different_banks_parallel() {
    let mut sp = BankedScratchpad::new(4, 1024, 16);

    let a = sp.try_write(0, 16, 0);
    let b = sp.try_write(1, 16, 0);

    assert!(a.is_some());
    assert!(b.is_some());
}

#[test]
fn write_then_read_same_bank_after_port_free() {
    let mut sp = BankedScratchpad::new(4, 1024, 16);

    let w = sp.try_write(0, 16, 0).expect("write t=0 ok");
    assert_eq!(w, 1);

    // Read at t=1 on the same bank's read port — different port, so should
    // still succeed even though write port was just busy.
    let r = sp.try_read(0, 16, 1).expect("read t=1 ok");
    assert_eq!(r, 2);
}

#[test]
fn write_takes_ceil_div_bytes_over_port_width() {
    let mut sp = BankedScratchpad::new(1, 1024, 16);
    // 64 bytes / 16 B/cycle = 4 cycles → done at t=0+4 = 4
    let done = sp.try_write(0, 64, 0).expect("write ok");
    assert_eq!(done, 4);
}

#[test]
fn write_blocks_until_previous_write_completes() {
    let mut sp = BankedScratchpad::new(1, 1024, 16);
    // first write 64 bytes occupies port until t=4
    let first = sp.try_write(0, 64, 0).expect("first ok");
    assert_eq!(first, 4);

    // write at t=2 should be blocked (port still busy)
    assert!(sp.try_write(0, 16, 2).is_none());

    // write at t=4 should succeed
    let later = sp.try_write(0, 16, 4).expect("once port free");
    assert_eq!(later, 5);
}

#[test]
fn capacity_bytes_reflects_construction() {
    let sp = BankedScratchpad::new(4, 1024, 16);
    assert_eq!(sp.capacity_bytes(), 4 * 1024);
}
