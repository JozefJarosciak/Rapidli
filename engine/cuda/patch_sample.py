"""Patch for ggml-cuda.dll (Rapidli): one fused sampling kernel for all rows of logits.

ARGMAX with a second operand "params" (float32 [4, rows], put there by our llama.dll, see llama-build/patch_greedy.py)
draws one token per row instead of taking the arg max:
    params[0] = 1 / temperature        0: plain arg max for this row
    params[1] = margin                 logits below (largest - margin) take no part (this is min-p: margin = -ln(min_p))
    params[2] = the bits of a 32-bit seed for this row and this pass
The draw is the Gumbel-max trick: arg max of logit/T + g with g = -log(-log(u)), u uniform in (0,1), is exactly one
draw from softmax(logits/T) over the tokens that take part. u comes from a hash of the seed and the token number, so
nothing is stored and nothing but the 4 numbers per row goes to the card.
  python patch_sample.py        (edits llama-src/ggml/src/ggml-cuda/argmax.cu in place, once)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "argmax.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI" in s:
    print("already patched")
    raise SystemExit(0)


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {""",
    """// RAPIDLI: the largest value of the whole block and its index, in every thread
static __device__ __forceinline__ void fq_block_max(float & val, int & idx, float * sh_val, int * sh_idx) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float v = __shfl_xor_sync(0xFFFFFFFF, val, offset, WARP_SIZE);
        const int   c = __shfl_xor_sync(0xFFFFFFFF, idx, offset, WARP_SIZE);
        if (v > val) { val = v; idx = c; }
    }
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        sh_val[warp_id] = val;
        sh_idx[warp_id] = idx;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < n_warps ? sh_val[lane_id] : -FLT_MAX;
        idx = lane_id < n_warps ? sh_idx[lane_id] : -1;
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float v = __shfl_xor_sync(0xFFFFFFFF, val, offset, WARP_SIZE);
            const int   c = __shfl_xor_sync(0xFFFFFFFF, idx, offset, WARP_SIZE);
            if (v > val) { val = v; idx = c; }
        }
        if (lane_id == 0) {
            sh_val[0] = val;
            sh_idx[0] = idx;
        }
    }
    __syncthreads();
    val = sh_val[0];
    idx = sh_idx[0];
}

// RAPIDLI: one draw per row (see cuda-build/patch_sample.py)
static __global__ void sample_f32(const float * __restrict__ x, const float * __restrict__ params, int32_t * __restrict__ dst, const int64_t ncols) {
    constexpr int    max_warps = 1024 / WARP_SIZE;
    __shared__ float sh_val[max_warps];
    __shared__ int   sh_idx[max_warps];

    const int64_t  row    = blockIdx.x;
    const float *  rowx   = x + row * ncols;
    const float    inv_t  = params[row * 4 + 0];
    const float    margin = params[row * 4 + 1];
    const uint32_t seed   = ((const uint32_t *) params)[row * 4 + 2];

    float maxval = -FLT_MAX;
    int   argmax = -1;
    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }
    fq_block_max(maxval, argmax, sh_val, sh_idx);

    if (inv_t > 0.0f) {
        const float low = maxval - margin;
        float best  = -FLT_MAX;
        int   besti = -1;
        for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const float val = rowx[col];
            if (val >= low) {
                uint32_t z = seed ^ ((uint32_t) col * 0x9E3779B1U);
                z ^= z >> 16; z *= 0x7feb352dU; z ^= z >> 15; z *= 0x846ca68bU; z ^= z >> 16;   // lowbias32
                const float u = ((float) (z >> 9) + 0.5f) * (1.0f / 8388608.0f);               // in (0, 1), never 0 or 1
                const float score = val * inv_t - logf(-logf(u));
                if (score > best) {
                    best  = score;
                    besti = col;
                }
            }
        }
        fq_block_max(best, besti, sh_val, sh_idx);
        if (besti >= 0) {
            argmax = besti;
        }
    }

    if (threadIdx.x == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {""")

rep("""    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);""",
    """    // RAPIDLI: with a second operand, one draw per row
    if (dst->src[1] != nullptr) {
        const ggml_tensor * src1 = dst->src[1];
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->ne[0] == 4 && src1->ne[1] == nrows && ggml_is_contiguous(src1));
        sample_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, (const float *) src1->data, dst_d, ne00);
        return;
    }

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched: fused sampling kernel in argmax.cu")
