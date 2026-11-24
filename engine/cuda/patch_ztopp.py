"""Patch for ggml-cuda (llama-src/): top-p in the fused sampling kernel (after patch_sample.py).

params[3] of a row (0 until now, put there by our llama.dll, see engine/llama/patch_ztopp.py) is top_p: when it lies
between 0 and 1, only the smallest set of the most likely tokens whose probabilities add up to top_p takes part in the
draw. As llama.cpp's sampler does, on the probabilities before the temperature and after min-p's cut.

The set is found without sorting: its edge is a logit threshold, the largest one that still keeps top_p of the mass, and
the mass above a threshold only falls as the threshold rises, so 24 halvings of [max - 40, max] find it (to 40 / 2^24 of
a logit; below max - 40 a token's share is under e^-40). That is 25 passes over the row, only for rows that ask for it,
each a block sum; the draw itself is the same Gumbel-max as before, over the tokens at or above the threshold.
A row whose params[3] is 0 (every row of an older llama.dll) samples exactly as before.
  python patch_ztopp.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "argmax.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI-TOPP" in s:
    raise SystemExit("already patched")
assert "sample_f32" in s, "patch_sample.py first"


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep("""// RAPIDLI: one draw per row (see cuda-build/patch_sample.py)""",
    """// RAPIDLI-TOPP: the sum over the whole block, in every thread (see engine/cuda/patch_ztopp.py)
static __device__ __forceinline__ float fq_block_sum(float v, float * sh) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset, WARP_SIZE);
    }
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    __syncthreads();                       // (sh[0] of the last call has been read by every thread)
    if (lane_id == 0) {
        sh[warp_id] = v;
    }
    __syncthreads();
    if (warp_id == 0) {
        v = lane_id < n_warps ? sh[lane_id] : 0.0f;
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset, WARP_SIZE);
        }
        if (lane_id == 0) {
            sh[0] = v;
        }
    }
    __syncthreads();
    return sh[0];
}

// RAPIDLI: one draw per row (see cuda-build/patch_sample.py)""")

rep("""    if (inv_t > 0.0f) {
        const float low = maxval - margin;""",
    """    if (inv_t > 0.0f) {
        float low = maxval - margin;
        // RAPIDLI-TOPP: top-p, the threshold found by halving (see engine/cuda/patch_ztopp.py)
        const float top_p = params[row * 4 + 3];
        if (top_p > 0.0f && top_p < 1.0f) {
            __shared__ float sh_sum[max_warps];
            float lo = fmaxf(low, maxval - 40.0f), hi = maxval;
            float z = 0.0f;
            for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
                const float val = rowx[col];
                if (val >= lo) {
                    z += expf(val - maxval);
                }
            }
            z = fq_block_sum(z, sh_sum);
            const float want = top_p * z;
            for (int it = 0; it < 24; ++it) {
                const float mid = 0.5f * (lo + hi);
                float m = 0.0f;
                for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
                    const float val = rowx[col];
                    if (val >= mid) {
                        m += expf(val - maxval);
                    }
                }
                m = fq_block_sum(m, sh_sum);
                if (m >= want) {
                    lo = mid;              // (every thread has the same sum, so all take the same branch)
                } else {
                    hi = mid;
                }
            }
            low = fmaxf(low, lo);
        }""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched: top-p in the fused sampling kernel (argmax.cu)")
