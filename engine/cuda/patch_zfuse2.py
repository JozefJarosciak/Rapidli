"""Patch for ggml-cuda (llama-src/): two small fusions for Inkling 2's many tiny operations.

One Inkling-2-Mini pass of one token is ~5,400 kernels of ~2 us (Nsight, 24 Nov): the kernel count, not the work, sets
a single lane's speed (~12 ms a token). Two chains are frequent and plain:
  ADD -> SCALE           (cur + conv) * s at the end of every short convolution: 192 a pass
  UNARY(TANH) -> MUL     tanh(gate) * x in the looped layers' controllers: 192 a pass
Each pair becomes one kernel when both operands are contiguous F32 tensors of the same shape and the first result is
used only by the second; the arithmetic is the same (the same roundings), so the results are bit-identical.
GGML_CUDA_DISABLE_RAPIDLI_FUSE2=1: one by one.
  python patch_zfuse2.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI-FUSE2" in s:
    raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep("static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {",
    """// RAPIDLI-FUSE2: ADD -> SCALE and TANH -> MUL in one kernel each (see engine/cuda/patch_zfuse2.py)
static __global__ void rapidli_add_scale_f32(const float * a, const float * b, float * dst, const int64_t n, const float scale, const float bias) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float t = a[i] + b[i];
    dst[i] = scale * t + bias;
}

static __global__ void rapidli_tanh_mul_f32(const float * x, const float * y, float * dst, const int64_t n, const bool tanh_first) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float t = tanhf(x[i]);
    dst[i] = tanh_first ? t * y[i] : y[i] * t;
}

// an elementwise kernel may write over an input that starts at the same address (each element is read before it is
// written; ggml's in-place allocation does exactly that), but not over one that overlaps it shifted
static bool rapidli_elementwise_alias_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * d0 = (const char *) dst->data, * d1 = d0 + ggml_nbytes(dst);
    const char * s0 = (const char *) src->data, * s1 = s0 + ggml_nbytes(src);
    return d0 == s0 || d1 <= s0 || s1 <= d0;
}

static int ggml_cuda_try_fuse_rapidli2(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool off = ggml_cuda_env_flag("GGML_CUDA_DISABLE_RAPIDLI_FUSE2");
    if (off || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * n0 = cgraph->nodes[i];
    ggml_tensor * n1 = cgraph->nodes[i + 1];
    const auto plain = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); };
    // (not when the SCALE starts a SiTU gate, SCALE -> TANH -> ..., which has its own fusion)
    const bool situ_next = i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_UNARY &&
        ggml_get_unary_op(cgraph->nodes[i + 2]) == GGML_UNARY_OP_TANH;
    if (n0->op == GGML_OP_ADD && n1->op == GGML_OP_SCALE && n1->src[0] == n0 && !situ_next) {
        const ggml_tensor * a = n0->src[0];
        const ggml_tensor * b = n0->src[1];
        if (!plain(a) || !plain(b) || !plain(n0) || !plain(n1) || !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, n1) ||
                !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_SCALE }, { i + 1 })) {
            return 0;
        }
        if (!rapidli_elementwise_alias_ok(n1, a) || !rapidli_elementwise_alias_ok(n1, b)) {
            return 0;
        }
        float scale, bias;
        memcpy(&scale, (const float *) n1->op_params + 0, sizeof(float));
        memcpy(&bias,  (const float *) n1->op_params + 1, sizeof(float));
        const int64_t n = ggml_nelements(n1);
        rapidli_add_scale_f32<<<(n + 255)/256, 256, 0, cuda_ctx->stream()>>>(
            (const float *) a->data, (const float *) b->data, (float *) n1->data, n, scale, bias);
        return 1;
    }
    if (n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_TANH && n1->op == GGML_OP_MUL &&
            (n1->src[0] == n0) != (n1->src[1] == n0)) {
        const bool tanh_first = n1->src[0] == n0;
        const ggml_tensor * x = n0->src[0];
        const ggml_tensor * y = tanh_first ? n1->src[1] : n1->src[0];
        if (!plain(x) || !plain(y) || !plain(n0) || !plain(n1) || !ggml_are_same_shape(x, y) || !ggml_are_same_shape(x, n1) ||
                !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { i + 1 })) {
            return 0;
        }
        if (!rapidli_elementwise_alias_ok(n1, x) || !rapidli_elementwise_alias_ok(n1, y)) {
            return 0;
        }
        const int64_t n = ggml_nelements(n1);
        rapidli_tanh_mul_f32<<<(n + 255)/256, 256, 0, cuda_ctx->stream()>>>(
            (const float *) x->data, (const float *) y->data, (float *) n1->data, n, tanh_first);
        return 1;
    }
    return 0;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {""")

rep("""    // SiTU gate: SCALE, TANH, SCALE, SIGMOID, MUL, VIEW, SCALE, TANH, SCALE, MUL
    if (node->op == GGML_OP_SCALE) {""", """    // RAPIDLI-FUSE2
    {
        const int n_fused = ggml_cuda_try_fuse_rapidli2(cuda_ctx, cgraph, i);
        if (n_fused > 0) {
            return n_fused;
        }
    }

    // SiTU gate: SCALE, TANH, SCALE, SIGMOID, MUL, VIEW, SCALE, TANH, SCALE, MUL
    if (node->op == GGML_OP_SCALE) {""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched", P)
