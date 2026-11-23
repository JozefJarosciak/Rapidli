"""Patch for ggml-cuda (our copy in llama-src/): the gated-delta-net kernel reads and writes a 16-bit state.
The arithmetic stays float32 (the state lives in registers as float32 while the kernel works); only the load from the
cache and the store into it convert. Used with LLAMA_RS_F16=1 in our llama.dll (llama-build/patch_f16.py).
  input  16-bit: the state source tensor has type F16 (a direct view of the cache)
  output 16-bit: the write-back copy that the backend fuses into the kernel has an F16 destination (one token per pass)
With GGML_CUDA_GDN_SR=1 the store rounds stochastically (a hash of the element address and a per-launch seed decides
up or down in proportion to the remainder), which keeps the rounding error unbiased.
  python patch_f16.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda")
NL = chr(10)


def patch(name, pairs):
    p = os.path.join(D, name)
    s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)
    if "RAPIDLI f16" in s:
        print(name, "already patched"); return
    for pair in pairs:
        a, b, cnt = pair[0], pair[1], (pair[2] if len(pair) > 2 else 1)
        if cnt == "first":                      # the same text follows in another function: only the first one is ours
            assert s.count(a) >= 1, (name, a[:70])
            s = s.replace(a, b, 1)
            continue
        assert s.count(a) == cnt, (name, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline=NL).write(s)
    print("patched", name)


patch("gated_delta_net.cuh", [(
    """    int64_t slot_stride; // between rollback slots (0 when K==1)
};""",
    """    int64_t slot_stride; // between rollback slots (0 when K==1)
    bool    half = false; // RAPIDLI f16: the cache holds 16-bit floats
};""")])

patch("gated_delta_net.cu", [
    ("""#include <cstdint>
""",
     """#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

// RAPIDLI f16: store a float32 as IEEE half, rounding up or down at random in proportion to the remainder
static __device__ __forceinline__ __half gdn_store_half(const float x, const uint32_t key, const bool stochastic) {
    const __half h = __float2half(x);          // round to nearest
    if (!stochastic) {
        return h;
    }
    const float back = __half2float(h);
    if (back == x || isinf(back) || isnan(back)) {
        return h;
    }
    // the neighbour on the other side of x
    const uint16_t bits = __half_as_ushort(h);
    const bool toward_larger_magnitude = fabsf(x) > fabsf(back);
    const uint16_t nbits = toward_larger_magnitude ? (uint16_t) (bits + 1) : (uint16_t) (bits - 1);
    const __half n = __ushort_as_half(nbits);
    const float other = __half2float(n);
    if (isinf(other) || isnan(other) || other == back) {
        return h;
    }
    // probability of taking the neighbour = distance to the nearest / gap (below one half by construction)
    const float p = fabsf(x - back) / fabsf(other - back);
    uint32_t z = key;
    z ^= z >> 16; z *= 0x7feb352dU; z ^= z >> 15; z *= 0x846ca68bU; z ^= z >> 16;   // lowbias32
    const float u = (float) (z >> 8) * (1.0f / 16777216.0f);
    return u < p ? n : h;
}
"""),
    ("""                                     int64_t       state_slot_stride,
                                     int           K) {""",
     """                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           state_half,
                                     uint32_t      sr_seed) {"""),
    ("""    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;""",
     """    // RAPIDLI f16: bit 0 = the state is read as half, bit 1 = it is written as half, bit 2 = stochastic rounding
    const bool in_half  = (state_half & 1) != 0;
    const bool out_half = (state_half & 2) != 0;
    const bool out_sr   = (state_half & 4) != 0;
    const __half * curr_state_h = (const __half *) curr_state + state_in_offset + col * S_v;
    __half *       state_h      = (__half *) state + state_out_offset;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;"""),
    ("""        s_shard[r]  = curr_state[i];""",
     """        s_shard[r]  = in_half ? __half2float(curr_state_h[i]) : curr_state[i];"""),
    ("""            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];""",
     """            const int i          = r * lanes_per_col + lane_in_col;
            if (out_half) {
                const uint32_t key = (uint32_t) (state_out_offset + col * S_v + i) * 0x9E3779B1U + sr_seed;
                state_h[col * S_v + i] = gdn_store_half(s_shard[r], key, out_sr);
            } else {
                state[col * S_v + i] = s_shard[r];
            }"""),
    ("""        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {""",
     """        float scale, int64_t state_slot_stride, int K, int state_half, uint32_t sr_seed, cudaStream_t stream) {"""),
    ("""                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);""",
     """                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_half, sr_seed);""", 4),
    ("""    const float * s_d   = (const float *) src_state->data;""",
     """    const float * s_d   = (const float *) src_state->data;

    // RAPIDLI f16
    static const bool sr_env = getenv("GGML_CUDA_GDN_SR") != nullptr && getenv("GGML_CUDA_GDN_SR")[0] == '1';
    static uint32_t sr_counter = 0;
    const bool in_half  = src_state->type == GGML_TYPE_F16;
    const bool out_half = cache != nullptr && cache->half;
    const int  state_half = (in_half ? 1 : 0) | (out_half ? 2 : 0) | (out_half && sr_env ? 4 : 0);
    const uint32_t sr_seed = (sr_counter++) * 0x85EBCA6BU + 0x1234567U;"""),
    ("""        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {""",
     """        if (state_half == 0 && ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {"""),
    ("""                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);""",
     """                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_half, sr_seed, stream);""", 4),
])

patch("ggml-cuda.cu", [(
    """    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }
""",
    """    // RAPIDLI f16: a 16-bit cache is written by the kernel too, when one token is decoded per pass
    // (a run of tokens only when the state is READ as 16-bit too: that case always uses the register kernel)
    const bool dst_half = dst->type == GGML_TYPE_F16 && K == 1 && (n_tokens == 1 || (gdn->src[5] != nullptr && gdn->src[5]->type == GGML_TYPE_F16));
    if (dst->op != GGML_OP_VIEW || (dst->type != GGML_TYPE_F32 && !dst_half) || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(dst->type) || dst->nb[1] != (size_t) ggml_row_size(dst->type, D)) {
        return 0;
    }
    fused_state_cpy.half = dst_half;
""", "first")])   # the gated-delta-net fusion check comes before the ssm_scan one, which has the same lines
