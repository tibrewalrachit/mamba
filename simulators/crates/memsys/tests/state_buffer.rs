//! Test 39 — `memsys::state_buffer::StateBuffer`
//!
//! SSM_STATE writes the state buffer during its execution window; SSM_OUTPUT
//! must not read until SSM_STATE has completed (or the buffer is explicitly
//! released).  StateBuffer models that exclusion.

use emamba_memsys::StateBuffer;

/// The buffer starts unlocked (available at any cycle).
#[test]
fn state_buffer_held_between_state_and_output() {
    let mut buf = StateBuffer::new(4096);
    assert!(buf.is_available(0), "initially available");

    // SSM_STATE starts at cycle 5 and takes 12 cycles → done_by = 17.
    // Lock the buffer for that window.
    buf.lock(/* until */ 17);

    // SSM_OUTPUT checks at cycle 16 — still locked.
    assert!(
        !buf.is_available(16),
        "state buffer locked during SSM_STATE execution (cycle 16)"
    );

    // At cycle 17 the lock expires.
    assert!(
        buf.is_available(17),
        "state buffer available once SSM_STATE completes (cycle 17)"
    );

    // Explicit release also works (e.g., callback path).
    buf.lock(100);
    assert!(!buf.is_available(99));
    buf.release();
    assert!(buf.is_available(99), "available after explicit release");
}

#[test]
fn state_buffer_capacity_is_reported() {
    let buf = StateBuffer::new(4096);
    assert_eq!(buf.capacity_bytes(), 4096);
}
