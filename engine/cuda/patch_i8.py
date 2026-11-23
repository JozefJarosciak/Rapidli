"""Patch for ggml-cuda.dll (Rapidli): gated-delta-net state stored as 8-bit codes, updated in place.

EXPERIMENTAL, CHANGES THE ANSWERS (see llama-build/patch_i8.py for the layout). When the state operand of
GATED_DELTA_NET has type I8, a cell holds H*S_v*S_v int8 codes followed by H*S_v float32 scales. The kernel reads a
column as code * scale, runs the usual float32 update, computes the output from the float32 state, then writes the
column back as codes with scale = largest magnitude / 127 and stochastic rounding (round up with probability equal to
the fraction, so that small updates are kept on average; plain rounding to nearest loses them). The random numbers
come from a hash of the element index and of the bits of the column's new scale, so a run is reproducible and the
draw changes from token to token without any counter (a counter would be frozen inside a recorded CUDA graph).
GGML_CUDA_GDN_I8_RTN=1 switches to rounding to nearest, for comparison.
  python patch_i8.py        (edits llama-src/ggml/src/ggml-cuda/gated_delta_net.cu in place, once)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "gated_delta_net.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI i8" in s:
    print("already patched")
    raise SystemExit(0)


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


# 1. pointers to the codes and the scale of this column, before the float pointers are advanced
rep("""    const bool out_sr   = (state_half & 4) != 0;
    const __half * curr_state_h = (const __half *) curr_state + state_in_offset + col * S_v;""",
    """    const bool out_sr   = (state_half & 4) != 0;
    // RAPIDLI i8: bit 3 = the state is 8-bit codes with one float32 scale per column, read and written in place
    const bool    io_i8         = (state_half & 8) != 0;
    const int64_t i8_seq_stride = H * S_v * (S_v + 4);
    int8_t * s8       = (int8_t *) const_cast<float *>(curr_state) + sequence * i8_seq_stride + ((int64_t) h_idx * S_v + col) * S_v;
    float *  s8_scale = (float *) ((int8_t *) const_cast<float *>(curr_state) + sequence * i8_seq_stride + H * S_v * S_v) + (int64_t) h_idx * S_v + col;
    const __half * curr_state_h = (const __half *) curr_state + state_in_offset + col * S_v;""")

# 2. read
rep("""        s_shard[r]  = in_half ? __half2float(curr_state_h[i]) : curr_state[i];
    }""",
    """        s_shard[r]  = in_half ? __half2float(curr_state_h[i]) : curr_state[i];
    }
    if (io_i8) {
        const float sc_in = *s8_scale;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            s_shard[r]  = (float) s8[i] * sc_in;
        }
    }""")
# the float reads above must not touch memory in 8-bit mode
rep("""        s_shard[r]  = in_half ? __half2float(curr_state_h[i]) : curr_state[i];
    }
    if (io_i8) {""",
    """        s_shard[r]  = io_i8 ? 0.0f : in_half ? __half2float(curr_state_h[i]) : curr_state[i];
    }
    if (io_i8) {""")

# 3. write
rep("""    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            if (out_half) {""",
    """    if constexpr (!keep_rs_t) {
        if (io_i8) {
            float m = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                m = fmaxf(m, fabsf(s_shard[r]));
            }
            m = warp_reduce_max<lanes_per_col>(m);
            const float    inv  = m > 0.0f ? 127.0f / m : 0.0f;
            const uint32_t salt = __float_as_uint(m) * 0x9E3779B1U;   // no call counter: a run must be repeatable
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int   i = r * lanes_per_col + lane_in_col;
                const float x = s_shard[r] * inv;
                float c;
                if (out_sr) {
                    const float f = floorf(x);
                    uint32_t z = (uint32_t) (col * S_v + i) * 0x85EBCA6BU + salt;
                    z ^= z >> 16; z *= 0x7feb352dU; z ^= z >> 15; z *= 0x846ca68bU; z ^= z >> 16;   // lowbias32
                    const float u = (float) (z >> 8) * (1.0f / 16777216.0f);
                    c = f + ((x - f) > u ? 1.0f : 0.0f);
                } else {
                    c = rintf(x);
                }
                s8[i] = (int8_t) fminf(fmaxf(c, -127.0f), 127.0f);
            }
            if (lane_in_col == 0) {
                *s8_scale = m * (1.0f / 127.0f);
            }
            return;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            if (out_half) {""")

# 4. the caller
rep("""    const int  state_half = (in_half ? 1 : 0) | (out_half ? 2 : 0) | (out_half && sr_env ? 4 : 0);""",
    """    // RAPIDLI i8
    static const bool i8_rtn = getenv("GGML_CUDA_GDN_I8_RTN") != nullptr && getenv("GGML_CUDA_GDN_I8_RTN")[0] == '1';
    const bool io_i8 = src_state->type == GGML_TYPE_I8;
    if (io_i8) {
        GGML_ASSERT(cache == nullptr && "the 8-bit state is written in place, there is no copy to fuse");
        GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == 1);
        GGML_ASSERT(src_state->ne[0] == nev1 * nev0 * (nev0 + 4) && src_state->ne[1] == nev3);
    }
    const int  state_half = io_i8 ? (8 | (i8_rtn ? 0 : 4)) : ((in_half ? 1 : 0) | (out_half ? 2 : 0) | (out_half && sr_env ? 4 : 0));""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched: 8-bit state in gated_delta_net.cu")
