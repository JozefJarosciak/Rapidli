"""Patch for ggml-cuda (llama-src): the sign flip of a Hadamard-rotated input in the Hadamard kernel. A Bonsai (prism)
model rotates the input of most matrix multiplications: MUL(x, signs), then MUL_MAT(hadamard, x) which ggml-cuda runs as
a fast Walsh-Hadamard transform. Each MUL is a kernel of its own and a pass over the activations, about 258 a pass. With
this patch the transform reads x and the signs and computes (x * sign) * scale, the same two multiplications in the same
order (the same numbers), and the MUL is skipped. GGML_CUDA_FWHT_SIGNS=0 at run time: the two kernels.
  python patch_fwhtsigns.py"""
import io, os
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"),
                 "ggml", "src", "ggml-cuda") + os.sep


def edit(name, pairs):
    p = D + name
    s = io.open(p, encoding="utf8").read().replace("\r\n", "\n")
    if "RAPIDLI-FWHTSIGNS" in s:
        raise SystemExit("already patched: " + name)
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf8", newline="\n").write(s)


edit("fwht.cuh", [(
    "bool ggml_cuda_op_fwht(",
    "// RAPIDLI-FWHTSIGNS: the transform of x * signs (signs broadcast along rows), as MUL then the transform would give\n"
    "bool ggml_cuda_op_fwht_signs(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * signs, ggml_tensor * dst);\n\n"
    "bool ggml_cuda_op_fwht(")])

edit("fwht.cu", [
    ("""__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {""",
     """__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs = nullptr, const int64_t signs_n = 0) {   // RAPIDLI-FWHTSIGNS"""),
    ("""    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }
""",
     """    float reg[NE];
    if (signs != nullptr) {
        // (x * sign) * scale: the MUL's product, then the transform's scaling, as the two kernels did it
        const float * sg = signs + (r * N) % signs_n;
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float xs = ggml_cuda_cast<float>(src[i * NT + tid]) * sg[i * NT + tid];
            reg[i] = xs * scale;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
        }
    }
"""),
    ("""template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];""",
     """template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                                   const float * signs = nullptr, const int64_t signs_n = 0) {
    const int     n    = src->ne[0];"""),
    ("""    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>""",
     """    switch (n) {
        case 64: case 128: case 256: case 512:
            if (signs != nullptr) {
                return false;   // RAPIDLI-FWHTSIGNS: only the block kernels take signs
            }
            break;
        default:
            break;
    }

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>"""),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);"),
    ("""bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {""",
     """// RAPIDLI-FWHTSIGNS: x is the MUL's first source (f32, contiguous), signs its second (one row, a multiple of the block)
bool ggml_cuda_op_fwht_signs(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * signs, ggml_tensor * dst) {
    const int n = (int) dst->ne[0];
    if (x->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst) || ggml_nelements(x) != ggml_nelements(dst) || n < 1024 ||
            signs->ne[0] % n != 0 || x->ne[0] != signs->ne[0]) {
        return false;
    }
    ggml_tensor xv = *x;    // x seen with the transform's rows
    for (int k = 0; k < GGML_MAX_DIMS; k++) {
        xv.ne[k] = dst->ne[k];
        xv.nb[k] = dst->nb[k];
    }
    return ggml_cuda_op_fwht_impl<float>(ctx, &xv, dst, (const float *) signs->data, signs->ne[0]);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {"""),
])

edit("ggml-cuda.cu", [
    ("static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {\n",
     """// RAPIDLI-FWHTSIGNS: node i is MUL(x, signs) used only by views that lead to a MUL_MAT run as a Hadamard transform:
// returns the MUL_MAT's index, else 0
static int ggml_cuda_fwht_signs_match(const ggml_cgraph * cg, int i) {
    static const bool off = getenv("GGML_CUDA_FWHT_SIGNS") != nullptr && atoi(getenv("GGML_CUDA_FWHT_SIGNS")) == 0;
    const ggml_tensor * mul = cg->nodes[i];
    if (off || mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * s = mul->src[1];
    if (x->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(s) ||
            s->ne[0] != x->ne[0] || ggml_nrows(s) != 1 || !ggml_are_same_shape(x, mul) || ggml_node_get_use_count(cg, i) != 1) {
        return 0;
    }
    int j = i + 1;
    const ggml_tensor * prev = mul;
    for (; j < cg->n_nodes && j < i + 4; j++) {
        const ggml_tensor * t = cg->nodes[j];
        if (t->op == GGML_OP_MUL_MAT) {
            break;
        }
        if (!ggml_cuda_is_view_or_noop(t) || t->src[0] != prev || t->view_offs != 0 || ggml_node_get_use_count(cg, j) != 1) {
            return 0;
        }
        prev = t;
    }
    if (j >= cg->n_nodes || j >= i + 4) {
        return 0;
    }
    const ggml_tensor * mm = cg->nodes[j];
    if (mm->src[1] != prev || !ggml_cuda_op_mul_mat_use_fwht(mm) || mm->src[1]->type != GGML_TYPE_F32 ||
            ggml_nelements(mm->src[1]) != ggml_nelements(mul) || mm->ne[0] < 1024 || s->ne[0] % mm->ne[0] != 0) {
        return 0;
    }
    return j;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
"""),
    ("""    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_CONCAT) {   // RAPIDLI-CONVFUSE""",
     """    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL) {      // RAPIDLI-FWHTSIGNS
        const int j = ggml_cuda_fwht_signs_match(cgraph, i);
        if (j > 0 && ggml_cuda_op_fwht_signs(*cuda_ctx, node->src[0], node->src[1], cgraph->nodes[j])) {
            return j - i;
        }
    }

    if (node->op == GGML_OP_CONCAT) {   // RAPIDLI-CONVFUSE"""),
    ("""            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                const int j = ggml_cuda_conv_store_match(cgraph, i);""",
     """            if (cgraph->nodes[i]->op == GGML_OP_MUL) {     // RAPIDLI-FWHTSIGNS: x stays allocated until the transform
                const int j = ggml_cuda_fwht_signs_match(cgraph, i);
                if (j > 0) {
                    add_alloc_deps(i, j);
                    i = j;
                    continue;
                }
            }
            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                const int j = ggml_cuda_conv_store_match(cgraph, i);"""),
])
print("signs in the Hadamard transform")
