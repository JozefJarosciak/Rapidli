"""Patch for ggml-cuda (llama-src), after patch_fwhtsigns.py and patch_optime.py (hence the z, the patches go in name order): the Hadamard kernel also writes its output in the q8_1
blocks that the vector kernel (MMVQ) reads, so the matrix-vector product that follows skips its own quantize kernel.
In fwht_cuda_block a warp holds 32 consecutive values of the output, one q8_1 block: the same max, the same d =
amax / 127, the same roundf(x / d) and sum as quantize_q8_1, so the blocks are the same bytes. The blocks go to a
buffer of the context (made once, outside a graph capture, for up to 8 tokens of 17,408 values); MMVQ takes them when
its input is that transform's output (same data, same size), and the eval loop forgets them at the first node that
is not a matrix product reading that output. GGML_CUDA_FWHT_Q8=0 at run time: off.
  python patch_zfwhtq8.py"""
import io, os
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"),
                 "ggml", "src", "ggml-cuda") + os.sep


def edit(name, pairs):
    p = D + name
    s = io.open(p, encoding="utf8").read().replace("\r\n", "\n")
    if "RAPIDLI-FWHTQ8" in s:
        raise SystemExit("already patched: " + name)
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf8", newline="\n").write(s)


edit("common.cuh", [(
    "    std::unique_ptr<ggml_cuda_stream_context> concurrent_stream_context;\n",
    "    std::unique_ptr<ggml_cuda_stream_context> concurrent_stream_context;\n\n"
    "    // RAPIDLI-FWHTQ8: the q8_1 blocks of the last Hadamard transform's output (see fwht.cu), for MMVQ\n"
    "    const void * fq_q8_src   = nullptr;   // that output's data\n"
    "    size_t       fq_q8_bytes = 0;         // its size in bytes\n"
    "    void *       fq_q8_buf   = nullptr;   // the blocks (made once, kept)\n"
    "    size_t       fq_q8_cap   = 0;\n")])

edit("fwht.cu", [
    ("""__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs = nullptr, const int64_t signs_n = 0) {   // RAPIDLI-FWHTSIGNS""",
     """__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs = nullptr, const int64_t signs_n = 0,      // RAPIDLI-FWHTSIGNS
                                block_q8_1 * q8 = nullptr) {                                   // RAPIDLI-FWHTQ8"""),
    ("""#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}""",
     """    if (q8 != nullptr) {
        // RAPIDLI-FWHTQ8: the 32 threads of a warp hold 32 consecutive values (i*NT + tid, NT a multiple of 32), one
        // q8_1 block, quantized as quantize_q8_1 does it
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float xi   = reg[i];
            const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
            const float d    = amax / 127.0f;
            const int8_t q   = amax == 0.0f ? 0 : roundf(xi / d);
            const int    sum = warp_reduce_sum<QK8_1>((int) q);
            const int64_t pos = r * N + i * NT + tid;
            q8[pos / QK8_1].qs[pos % QK8_1] = q;
            if (pos % QK8_1 == 0) {
                q8[pos / QK8_1].ds = make_half2(d, d * sum);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}"""),
    ("""template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                                   const float * signs = nullptr, const int64_t signs_n = 0) {
    const int     n    = src->ne[0];""",
     """// RAPIDLI-FWHTQ8: where the transform writes q8_1 blocks of dst too (nullptr: not), and dst noted as their source
static block_q8_1 * ggml_cuda_fwht_q8_target(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    static const bool off = getenv("GGML_CUDA_FWHT_Q8") != nullptr && atoi(getenv("GGML_CUDA_FWHT_Q8")) == 0;
    ctx.fq_q8_src = nullptr;
    const int64_t ne = ggml_nelements(dst);
    if (off || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ne % QK8_1 != 0 || ne > 8 * 17408) {
        return nullptr;
    }
    const size_t need = (size_t) (ne / QK8_1) * sizeof(block_q8_1);
    if (ctx.fq_q8_cap < need) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ctx.stream(), &st) != cudaSuccess || st != cudaStreamCaptureStatusNone || ctx.fq_q8_cap > 0) {
            return nullptr;       // (made once, never inside a capture)
        }
        const size_t cap = (size_t) (8 * 17408 / QK8_1) * sizeof(block_q8_1);
        if (cudaMalloc(&ctx.fq_q8_buf, cap) != cudaSuccess) {
            ctx.fq_q8_buf = nullptr;
            ctx.fq_q8_cap = 1;    // (not again)
            (void) cudaGetLastError();
            return nullptr;
        }
        ctx.fq_q8_cap = cap;
    }
    ctx.fq_q8_src   = dst->data;
    ctx.fq_q8_bytes = ggml_nbytes(dst);
    return (block_q8_1 *) ctx.fq_q8_buf;
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                                   const float * signs = nullptr, const int64_t signs_n = 0) {
    const int     n    = src->ne[0];"""),
    ("""    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;
""",
     """    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;
        static_assert(nt % QK8_1 == 0, "a warp must hold whole q8_1 blocks");
        block_q8_1 * q8 = (n == 1024 || n == 2048 || n == 4096 || n == 8192) ? ggml_cuda_fwht_q8_target(ctx, dst) : nullptr;
"""),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n, q8);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n, q8);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n, q8);"),
    ("ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n);",
     "ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale, signs, signs_n, q8);"),
])

edit("mmvq.cu", [
    ("""    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }""",
     """    // RAPIDLI-FWHTQ8: the input is the last Hadamard transform's output, whose kernel wrote its q8_1 blocks already
    const bool fq_hit = ctx.fq_q8_src != nullptr && src1->data == ctx.fq_q8_src && ggml_nbytes(src1) == ctx.fq_q8_bytes &&
        src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && ne10 == ne10_padded;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    if (!fq_hit) {
        src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
    void * src1_q8 = fq_hit ? ctx.fq_q8_buf : (void *) src1_q8_1.get();"""),
    ("        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,",
     "        src0->data, src0->type, src1_q8, ids_d, fusion_local, dst_d, ne00,"),
])

edit("ggml-cuda.cu", [(
    """                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                double fq_t0 = 0;""",
    """                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // RAPIDLI-FWHTQ8: the q8_1 blocks of the last transform hold only for the products that read it next
                if (cuda_ctx->fq_q8_src != nullptr &&
                        !(node->op == GGML_OP_MUL_MAT && node->src[1] != nullptr && node->src[1]->data == cuda_ctx->fq_q8_src)) {
                    cuda_ctx->fq_q8_src = nullptr;
                }

                double fq_t0 = 0;""")])
print("Hadamard output also as q8_1 blocks for MMVQ")
