"""Patch for ggml-cuda (llama-src): MUL_MAT_ID (a mixture of experts) for 9 to 64 tokens on the vector kernel, 8 tokens
at a time, when the tokens rarely share an expert.

llama.cpp takes the vector kernel (one weight row read per output, any expert per token) for up to 8 tokens, and MMQ
tiles from 9 on. With llama.cpp's MoE cache the expert tensor is the cache's bank of slots (about 11,000 for
Qwen3.8-Flash-Next with a 15 GB cache), and the MMQ grid covers every slot, used or not: a pass of 16 lanes took
about three times as long as one of 8 (Qwen3.8-Flash-Next Q2_0 on an RTX 4090, 20 Nov 2026: 16 lanes 141 tok/s,
8 lanes 200). The vector kernel's cost does not depend on the number of slots, and with 16 lanes x 10 experts over
512 a token rarely shares an expert with another, so reading the weights per slice of 8 tokens costs about what MMQ
reads once.

Taken when the experts outnumber the token-expert pairs at least 4 times (ne02 >= 4 x tokens x experts used): a
batch that uses the same experts again and again (a long prompt) still goes to MMQ. GGML_CUDA_MMID_SLICE=0 at run
time: as before; GGML_CUDA_MMID_SLICE=<n>: up to n tokens (default 64); GGML_CUDA_MMID_SLICE_RATIO=<r>: the experts
must outnumber the token-expert pairs r times (default 4; 0: always).
  python patch_zmmidslice.py"""
import io, os
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"), "ggml", "src", "ggml-cuda") + os.sep


def edit(name, pairs):
    p = D + name
    s = io.open(p, encoding="utf8").read().replace("\r\n", "\n")
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:70])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf8", newline="\n").write(s)


edit("ggml-cuda.cu", [(
    """        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }""",
    """        // RAPIDLI-MMIDSLICE: 9-64 tokens that rarely share an expert, on the vector kernel 8 tokens at a time (see
        // engine/cuda/patch_zmmidslice.py)
        if (ggml_is_quantized(src0->type) && ne2 > MMVQ_MAX_BATCH_SIZE) {
            static const int slice_max = getenv("GGML_CUDA_MMID_SLICE") ? atoi(getenv("GGML_CUDA_MMID_SLICE")) : 64;
            static const float ratio = getenv("GGML_CUDA_MMID_SLICE_RATIO") ? (float) atof(getenv("GGML_CUDA_MMID_SLICE_RATIO")) : 4.0f;
            const int per = get_mmvq_mmid_max_batch(src0->type, cc);
            if (ne2 <= slice_max && per > 0 && (float) ne02 >= ratio * ne2 * ids->ne[0]) {
                for (int64_t t0 = 0; t0 < ne2; t0 += per) {
                    const int64_t nt = std::min<int64_t>(per, ne2 - t0);
                    ggml_tensor s1 = *src1; s1.ne[2] = nt; s1.data = (char *) src1->data + t0 * src1->nb[2];
                    ggml_tensor si = *ids;  si.ne[1] = nt; si.data = (char *) ids->data  + t0 * ids->nb[1];
                    ggml_tensor sd = *dst;  sd.ne[2] = nt; sd.data = (char *) dst->data  + t0 * dst->nb[2];
                    ggml_cuda_mul_mat_vec_q(ctx, src0, &s1, &si, &sd);
                }
                return;
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }""")])
print("MUL_MAT_ID for 9-64 tokens on the vector kernel in slices (RAPIDLI-MMIDSLICE)")
