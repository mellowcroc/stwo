const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;

// Define constants
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 8;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2u * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const LOG_N_LANES: u32 = 4;

// Initialize EXTERNAL_ROUND_CONSTS with explicit values
var<private> EXTERNAL_ROUND_CONSTS: array<array<u32, N_STATE>, FULL_ROUNDS> = array<array<u32, N_STATE>, FULL_ROUNDS>(
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
);

// Initialize INTERNAL_ROUND_CONSTS with explicit values
var<private> INTERNAL_ROUND_CONSTS: array<u32, N_PARTIAL_ROUNDS> = array<u32, N_PARTIAL_ROUNDS>(
    1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234
);

// Create ColumnVec struct
struct ColumnVec {
    data: array<array<u32, N_STATE>, N_INSTANCES_PER_ROW>,
    length: u32,
}

struct BaseColumn {
    data: array<PackedM31, N_STATE>,
    length: u32,
}

struct PackedM31 {
    data: array<u32, N_LANES>,
}

struct GenTraceInput {
    log_size: u32,
}

struct DebugData {
    index: array<u32, 16>,
    values: array<u32, 16>,
    counter: atomic<u32>,
}

struct LookupData {
    initial_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
    final_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
}

struct GenTraceOutput {
    data: array<PackedM31, N_STATE>,
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}

@group(0) @binding(0)
var<storage, read> input: GenTraceInput;

// Output buffer
@group(0) @binding(1)
var<storage, read_write> output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> debug_buffer: DebugData;

fn from_u32(value: u32) -> PackedM31 {
    var packedM31 = PackedM31();
    for (var i = 0u; i < N_LANES; i++) {
        packedM31.data[i] = value;
    }
    return packedM31;
}

fn add(a: PackedM31, b: PackedM31) -> PackedM31 {
    var packedM31 = PackedM31();
    for (var i = 0u; i < N_LANES; i++) {
        packedM31.data[i] = partial_reduce(a.data[i] + b.data[i]);
    }
    return packedM31;
}

fn mul(a: PackedM31, b: PackedM31) -> PackedM31 {
    var packedM31 = PackedM31();
    for (var i = 0u; i < N_LANES; i++) {
        var temp: u64 = u64(a.data[i]);
        temp = temp * u64(b.data[i]);
        packedM31.data[i] = full_reduce(temp);
    }
    return packedM31;
}

// Partial reduce for values in [0, 2P)
fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn full_reduce(val: u64) -> u32 {
    let first_shift = val >> MODULUS_BITS;
    let first_sum = first_shift + val + 1;
    let second_shift = first_sum >> MODULUS_BITS;
    let final_sum = second_shift + val;
    return u32(final_sum & u64(P));
}

// Function to apply pow5 operation
fn pow5(x: PackedM31) -> PackedM31 {
    return mul(mul(mul(x, x), mul(x, x)), x);
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix(state: array<PackedM31, N_STATE>) -> array<PackedM31, N_STATE> {
    // Applies circ(2M4, M4, M4, M4).
    var modified_state = state;
    for (var i = 0u; i < 4u; i++) {
        let partial_state = array<PackedM31, 4>(
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        );
        let modified_partial_state = apply_m4(partial_state);
        modified_state[4 * i] = modified_partial_state[0];
        modified_state[4 * i + 1] = modified_partial_state[1];
        modified_state[4 * i + 2] = modified_partial_state[2];
        modified_state[4 * i + 3] = modified_partial_state[3];
    }
    for (var j = 0u; j < 4u; j++) {
        let s = add(add(modified_state[j], modified_state[j + 4]), add(modified_state[j + 8], modified_state[j + 12]));
        for (var i = 0u; i < 4u; i++) {
            modified_state[4 * i + j] = add(modified_state[4 * i + j], s);
        }
    }
    return modified_state;
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix(state: array<PackedM31, N_STATE>) -> array<PackedM31, N_STATE> {
    var sum = state[0];
    for (var i = 1u; i < N_STATE; i++) {
        sum = add(sum, state[i]);
    }

    var result = array<PackedM31, N_STATE>();
    for (var i = 0u; i < N_STATE; i++) {
        let factor = partial_reduce(1u << (i + 1));
        result[i] = add(mul(from_u32(factor), state[i]), sum);
    }

    return result;
}

/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4(x: array<PackedM31, 4>) -> array<PackedM31, 4> {
    let t0 = add(x[0], x[1]);
    let t02 = add(t0, t0);
    let t1 = add(x[2], x[3]);
    let t12 = add(t1, t1);
    let t2 = add(add(x[1], x[1]), t1);
    let t3 = add(add(x[3], x[3]), t0);
    let t4 = add(add(t12, t12), t3);
    let t5 = add(add(t02, t02), t2);
    let t6 = add(t3, t5);
    let t7 = add(t2, t4);
    return array<PackedM31, 4>(t6, t5, t7, t4);
}

fn store_debug_value(index: u32, value: u32) {
    let debug_idx = atomicAdd(&debug_buffer.counter, 1u);
    debug_buffer.index[debug_idx] = index;
    debug_buffer.values[debug_idx] = value;
}

@compute @workgroup_size(256)
fn gen_trace(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    if (GlobalInvocationID.x != 0u) {
        return;
    }

    let log_size = input.log_size;

    if (log_size < LOG_N_LANES) {
        return;
    }

    for (var vec_index = 0u; vec_index < (1u << (log_size - LOG_N_LANES)); vec_index++) {
        var col_index = 0u;

        for (var rep_i = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
            var state: array<PackedM31, N_STATE> = initialize_state(vec_index, rep_i);

            for (var i = 0u; i < N_STATE; i++) {
                output.trace[col_index].data[vec_index] = state[i];
                col_index += 1u;
            }

            for (var i = 0u; i < N_STATE; i++) {
                output.lookup_data.initial_state[rep_i][i].data[vec_index] = state[i];
            }

            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = add(state[j], from_u32(EXTERNAL_ROUND_CONSTS[i][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    output.trace[col_index].data[vec_index] = state[j];
                    col_index += 1u;
                }
            }
            // Partial rounds
            for (var i = 0u; i < N_PARTIAL_ROUNDS; i++) {
                state[0] = add(state[0], from_u32(INTERNAL_ROUND_CONSTS[i]));
                state = apply_internal_round_matrix(state);
                state[0] = pow5(state[0]);
                output.trace[col_index].data[vec_index] = state[0];
                col_index += 1u;
            }
            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = add(state[j], from_u32(EXTERNAL_ROUND_CONSTS[i + N_HALF_FULL_ROUNDS][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    output.trace[col_index].data[vec_index] = state[j];
                    col_index += 1u;
                }
            }

            for (var j = 0u; j < N_STATE; j++) {
                output.lookup_data.final_state[rep_i][j].data[vec_index] = state[j];
            }
        }
    }
}

// Function to initialize the state array
fn initialize_state(vec_index: u32, rep_i: u32) -> array<PackedM31, N_STATE> {
    var state: array<PackedM31, N_STATE>;

    for (var state_i = 0u; state_i < N_STATE; state_i++) {
        // Initialize each element of the state array
        var packed_value = PackedM31();

        for (var i = 0u; i < N_LANES; i++) {
            // Calculate the value based on vec_index, state_i, and rep_i
            let value: u32 = vec_index * 16u + i + state_i + rep_i;
            // Here, you would typically pack this value into a PackedBaseField equivalent
            // For simplicity, we'll just assign it directly
            packed_value.data[i] = value; // Replace with actual packing logic if needed
        }
        state[state_i] = packed_value;
    }

    return state;
}
