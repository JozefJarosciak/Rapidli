"""Patch for ggml-cuda (llama-src): the gated-delta-net convolution in one kernel. With engine/llama/patch_zzconv.py
the graph of a qwen35 layer has, one after another: CONCAT (the stored state, 3 columns, joined with the new channels),
SSM_CONV, SILU, and the CPY that stores the last 3 columns back as the new state. Run one by one, the joined input is
written, read back by the convolution and read again by the copy. The fused kernel reads the state and the new channels
once, computes the convolution in the same order of additions (the same numbers), and writes the output and the new
state; the joined input is never written. GGML_CUDA_CONV_FUSE=0 at run time: the operations one by one.
  python patch_convfuse.py"""
import io, os
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"),
                 "ggml", "src", "ggml-cuda") + os.sep


def edit(name, pairs):
    p = D + name
    s = io.open(p, encoding="utf8").read().replace("\r\n", "\n")
    if "RAPIDLI-CONVFUSE" in s:
        raise SystemExit("already patched: " + name)
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf8", newline="\n").write(s)


edit("ssm-conv.cuh", [(
    "void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);",
    "void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);\n"
    "\n// RAPIDLI-CONVFUSE: CONCAT(state, x), SSM_CONV, SILU and the CPY of the new state, as one kernel\n"
    "void ggml_cuda_op_ssm_conv_store_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * cat, const ggml_tensor * conv, ggml_tensor * silu, ggml_tensor * cpy);\n")])

edit("ssm-conv.cu", [("""template <bool apply_silu>
static void ssm_conv_f32_cuda(""", """// RAPIDLI-CONVFUSE: one thread per channel and sequence. X holds the last d_conv inputs: the state, then each new token;
// y = silu(sum_j X[j] * w[j]) added in the order of ssm_conv_f32 (j = 0 .. d_conv-1, then the bias, here 0)
template <int d_conv>
static __global__ void ssm_conv_store_f32(const char * s, const int64_t s_nb1, const int64_t s_nb2,
                                          const char * x, const int64_t x_nb0, const int64_t x_nb1, const int64_t x_nb2,
                                          const float * w, const int64_t w_nb1,
                                          char * y, const int64_t y_nb1, const int64_t y_nb2,
                                          char * st, const int64_t st_nb2,
                                          const int nr, const int64_t n_t, const float b) {
    const int c  = blockIdx.y * blockDim.x + threadIdx.x;
    const int sq = blockIdx.x;
    if (c >= nr) {
        return;
    }
    float X[d_conv];
    float wk[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        wk[j] = w[c * (w_nb1 / sizeof(float)) + j];
    }
    const float * sp = (const float *) (s + sq * s_nb2 + c * s_nb1);
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        X[j] = sp[j];
    }
    const char * xp = x + sq * x_nb2 + c * x_nb1;
    char *       yp = y + sq * y_nb2 + c * sizeof(float);
    for (int64_t i = 0; i < n_t; i++) {
        X[d_conv - 1] = *(const float *) (xp + i * x_nb0);
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += X[j] * wk[j];
        }
        sumf += b;
        *(float *) (yp + i * y_nb1) = ggml_cuda_op_silu_single(sumf);
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            X[j] = X[j + 1];
        }
    }
    float * stp = (float *) (st + sq * st_nb2) + c * (d_conv - 1);
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        stp[j] = X[j];
    }
}

void ggml_cuda_op_ssm_conv_store_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * cat, const ggml_tensor * conv, ggml_tensor * silu, ggml_tensor * cpy) {
    const ggml_tensor * s  = cat->src[0];   // the stored state [d_conv-1, nr, n_s]
    const ggml_tensor * x  = cat->src[1];   // the new channels [n_t, nr, n_s]
    const ggml_tensor * w  = conv->src[1];  // [d_conv, nr]
    ggml_tensor *       st = cpy->src[1];   // where the new state goes: [(d_conv-1) * nr, n_s]
    const int     nr  = (int) s->ne[1];
    const int64_t n_t = x->ne[0];
    const int64_t n_s = s->ne[2];
    const int threads = 128;
    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    GGML_ASSERT(w->ne[0] == 4);
    ssm_conv_store_f32<4><<<blocks, threads, 0, ctx.stream()>>>((const char *) s->data, s->nb[1], s->nb[2],
        (const char *) x->data, x->nb[0], x->nb[1], x->nb[2], (const float *) w->data, w->nb[1],
        (char *) silu->data, silu->nb[1], silu->nb[2], (char *) st->data, st->nb[1], nr, n_t, 0.0f);
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(""")])

edit("ggml-cuda.cu", [
    ("static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {\n",
     """// RAPIDLI-CONVFUSE: nodes i.. are CONCAT(state, x) along dim 0, SSM_CONV of it (kernel 4), SILU of that, then only views
// and the CPY of the concat's last 3 columns (the new state) into a 2D view: returns the CPY's index, else 0
static int ggml_cuda_conv_store_match(const ggml_cgraph * cg, int i) {
    static const bool off = getenv("GGML_CUDA_CONV_FUSE") != nullptr && atoi(getenv("GGML_CUDA_CONV_FUSE")) == 0;
    if (off || i + 4 >= cg->n_nodes) {
        return 0;
    }
    const ggml_tensor * cat = cg->nodes[i];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * conv = cg->nodes[i + 1];
    const ggml_tensor * silu = cg->nodes[i + 2];
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != cat || silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != conv) {
        return 0;
    }
    const ggml_tensor * s = cat->src[0];
    const ggml_tensor * x = cat->src[1];
    const ggml_tensor * w = conv->src[1];
    if (s->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 ||
            w->ne[0] != 4 || w->nb[0] != sizeof(float) || s->ne[0] != 3 || s->nb[0] != sizeof(float) ||
            s->nb[1] != 3 * sizeof(float) || s->ne[1] != x->ne[1] || s->ne[2] != x->ne[2] || x->ne[3] != 1 || s->ne[3] != 1 ||
            w->ne[1] != s->ne[1] || silu->ne[0] != s->ne[1] || silu->ne[1] != x->ne[0] || silu->ne[2] != s->ne[2] ||
            silu->nb[0] != sizeof(float) || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    if (ggml_node_get_use_count(cg, i) != 2 || ggml_node_get_use_count(cg, i + 1) != 1) {
        return 0;
    }
    int j = i + 3;
    for (; j < cg->n_nodes && j < i + 8; j++) {
        if (cg->nodes[j]->op == GGML_OP_CPY) {
            break;
        }
        if (!ggml_cuda_is_view_or_noop(cg->nodes[j])) {
            return 0;
        }
    }
    if (j >= cg->n_nodes || cg->nodes[j]->op != GGML_OP_CPY) {
        return 0;
    }
    const ggml_tensor * v  = cg->nodes[j]->src[0];
    const ggml_tensor * st = cg->nodes[j]->src[1];
    if (v->view_src != cat || v->view_offs != (size_t) x->ne[0] * sizeof(float) || v->ne[0] != 3 || v->ne[1] != s->ne[1] ||
            v->ne[2] != s->ne[2] || v->nb[1] != cat->nb[1] || v->nb[2] != cat->nb[2] || st->type != GGML_TYPE_F32 ||
            st->nb[0] != sizeof(float) || st->ne[0] != 3 * s->ne[1] || st->ne[1] != s->ne[2] || st->ne[2] != 1 || st->ne[3] != 1) {
        return 0;
    }
    return j;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
"""),
    ("""    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&""",
     """    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_CONCAT) {   // RAPIDLI-CONVFUSE
        const int j = ggml_cuda_conv_store_match(cgraph, i);
        if (j > 0) {
            ggml_cuda_op_ssm_conv_store_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[j]);
            return j - i;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&"""),
    ("""        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive""",
     """        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // RAPIDLI-CONVFUSE: the state and the new channels stay allocated until the state's CPY, so that the
            // convolution's output (written by the fused kernel) cannot share their memory
            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                const int j = ggml_cuda_conv_store_match(cgraph, i);
                if (j > 0) {
                    add_alloc_deps(i, j);
                    i = j;
                    continue;
                }
            }
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive"""),
])
print("conv + state store fused")
