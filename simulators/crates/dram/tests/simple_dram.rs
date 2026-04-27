//! Tests 23-24 — `dram::simple::*`
//! SimpleDram: fixed latency + per-cycle bandwidth queue.
//!
//!     done_by = enqueue_time + latency_cycles + ceil(bytes / bandwidth)
//!
//! Queue-depth full → enqueue returns the request as Err (back-pressure).

use emamba_dram::simple::SimpleDram;

#[test]
fn latency_plus_bandwidth() {
    // bw=8 B/cycle, lat=80, 80 bytes → done_by = 0 + 80 + ceil(80/8) = 90
    let mut dram = SimpleDram::new(/* bw */ 8, /* lat */ 80, /* queue */ 16);
    let id = dram.enqueue(/* bytes */ 80, /* now */ 0).expect("enqueue ok");
    assert_eq!(id, 0);

    // No drains until done_by=90.
    for t in 0..90 {
        let drained = dram.drain(t);
        assert!(drained.is_empty(), "drain at t={} should be empty", t);
    }

    let drained = dram.drain(90);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0], 0);
}

#[test]
fn queue_full_returns_back_pressure() {
    let mut dram = SimpleDram::new(/* bw */ 8, /* lat */ 80, /* queue */ 2);

    assert!(dram.enqueue(8, 0).is_ok());
    assert!(dram.enqueue(8, 0).is_ok());
    let err = dram.enqueue(8, 0);
    assert!(err.is_err(), "third enqueue must back-pressure");
}

#[test]
fn occupancy_tracks_in_flight_count() {
    let mut dram = SimpleDram::new(8, 80, 16);
    assert_eq!(dram.occupancy(), 0);

    dram.enqueue(8, 0).unwrap();
    dram.enqueue(8, 0).unwrap();
    assert_eq!(dram.occupancy(), 2);

    // Both done by t=81 (lat 80 + 1 cycle bw)
    let _ = dram.drain(81);
    assert_eq!(dram.occupancy(), 0);
}
