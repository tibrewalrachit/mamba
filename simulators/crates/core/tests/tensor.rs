//! Test 2 — `core::tensor_desc::bytes_product_of_shape`
//! TensorDesc.bytes() == product(shape) * dtype.bytes().

use emamba_core::{Dtype, MemSpace, TensorDesc};

#[test]
fn bytes_product_of_shape() {
    let v = TensorDesc {
        base: 0x1000,
        space: MemSpace::Sram,
        shape: vec![20].into(),
        dtype: Dtype::Int8,
    };
    assert_eq!(v.bytes(), 20);

    let m = TensorDesc {
        base: 0x2000,
        space: MemSpace::Sram,
        shape: vec![20, 8].into(),
        dtype: Dtype::Int24,
    };
    assert_eq!(m.bytes(), 480);
}

#[test]
fn scalar_shape_bytes() {
    let s = TensorDesc {
        base: 0,
        space: MemSpace::Dram,
        shape: vec![1].into(),
        dtype: Dtype::Fp32,
    };
    assert_eq!(s.bytes(), 4);
}
