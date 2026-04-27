//! Tests 23-24, 38 — `dram::simple::*`
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

    // Bandwidth serialized: req1 done at 81, req2 done at 82.
    let _ = dram.drain(81);
    assert_eq!(dram.occupancy(), 1, "req2 still in flight at t=81");

    let _ = dram.drain(82);
    assert_eq!(dram.occupancy(), 0);
}

/// Test 38 — two back-to-back loads of 80 bytes each at bw=8, lat=80.
/// The bus is shared: req2 cannot start transferring until req1 finishes.
///   req1: transfer_start = max(bw_free=0, 0+80) = 80, done_by = 80+10 = 90, bw_free = 90
///   req2: transfer_start = max(bw_free=90, 0+80) = 90, done_by = 90+10 = 100, bw_free = 100
#[test]
fn bandwidth_pressure_serializes_two_loads() {
    let mut dram = SimpleDram::new(/* bw */ 8, /* lat */ 80, /* queue */ 16);
    dram.enqueue(80, 0).unwrap();
    dram.enqueue(80, 0).unwrap();

    // Nothing done at t=89.
    assert!(dram.drain(89).is_empty(), "nothing done at t=89");

    // Only req1 done at t=90.
    let r = dram.drain(90);
    assert_eq!(r.len(), 1, "exactly req1 done at t=90");

    // req2 not yet done at t=99.
    assert!(dram.drain(99).is_empty(), "req2 not done at t=99");

    // req2 done at t=100.
    let r = dram.drain(100);
    assert_eq!(r.len(), 1, "req2 done at t=100");
}
