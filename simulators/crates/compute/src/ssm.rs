//! SSM compute units — paper §4.4, Figure 5.
//!
//! The SSM block has two sub-stages per token, each backed by an ED×N MAC
//! array. State update `h_t = Ā h_{t-1} + B̄ x_t` consumes a state read +
//! write; output `y_t = C h_t + D x_t` is a read-only consumer with an
//! adder-chain pipeline fill.
//!
//! `ssm_state.cycles  = ceil(D*N*E / mac_array_width) + READ_STATE + WRITE_STATE`
//! `ssm_output.cycles = ceil(D*N*E / mac_array_width) + PIPELINE_FILL`

#[derive(Clone, Debug)]
pub struct SsmStateUnit {
    mac_array_width: u32,
    read_state_lat: u32,
    write_state_lat: u32,
}

#[derive(Clone, Debug)]
pub struct SsmOutputUnit {
    mac_array_width: u32,
    pipeline_fill: u32,
}

impl SsmStateUnit {
    pub fn new(mac_array_width: u32, read_state_lat: u32, write_state_lat: u32) -> Self {
        assert!(mac_array_width >= 1, "MAC array width must be >= 1");
        Self {
            mac_array_width,
            read_state_lat,
            write_state_lat,
        }
    }

    pub fn cycles_for(&self, d: u32, n: u32, e: u32) -> u32 {
        let work = d as u64 * n as u64 * e as u64;
        let mac_cycles = (work + self.mac_array_width as u64 - 1) / self.mac_array_width as u64;
        mac_cycles as u32 + self.read_state_lat + self.write_state_lat
    }
}

impl SsmOutputUnit {
    pub fn new(mac_array_width: u32, pipeline_fill: u32) -> Self {
        assert!(mac_array_width >= 1, "MAC array width must be >= 1");
        Self {
            mac_array_width,
            pipeline_fill,
        }
    }

    pub fn cycles_for(&self, d: u32, n: u32, e: u32) -> u32 {
        let work = d as u64 * n as u64 * e as u64;
        let mac_cycles = (work + self.mac_array_width as u64 - 1) / self.mac_array_width as u64;
        mac_cycles as u32 + self.pipeline_fill
    }
}
