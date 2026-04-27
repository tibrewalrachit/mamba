//! Test 19 — `compute::linear_proj::cycles_match_mac_width`
//! cycles = ceil(in_dim * out_dim / mac_width) + PIPELINE_FILL  (paper §4.6)

use emamba_compute::linear::LinearProjUnit;

#[test]
fn cycles_match_mac_width() {
    let u = LinearProjUnit::new(64, 2); // mac_width=64, fill=2
    // 20 * 40 / 64 = 12.5 → 13; + 2 = 15
    assert_eq!(u.cycles_for(20, 40), 15);
}

#[test]
fn cycles_round_up_when_mac_doesnt_divide() {
    let u = LinearProjUnit::new(7, 0);
    // 20 * 40 / 7 = 800 / 7 = 114.28... → 115
    assert_eq!(u.cycles_for(20, 40), 115);
}

#[test]
fn wider_mac_is_faster() {
    let narrow = LinearProjUnit::new(16, 0);
    let wide = LinearProjUnit::new(128, 0);
    assert!(narrow.cycles_for(64, 64) > wide.cycles_for(64, 64));
}

#[test]
fn pipeline_fill_is_additive() {
    let bare = LinearProjUnit::new(64, 0);
    let with_fill = LinearProjUnit::new(64, 5);
    assert_eq!(
        with_fill.cycles_for(20, 40) - bare.cycles_for(20, 40),
        5
    );
}
