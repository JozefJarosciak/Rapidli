"""Patch for ggml-cuda (llama-src/): with GGML_CUDA_MMVF_F32_SPLIT=1, an F32 matrix on 9-16 columns (2D, contiguous)
runs the matrix-vector kernel (MMVF) on slices of 8 columns instead of the tiled mul_mat_f. Off by default (experiment).

Inkling 2's f32 router (2304 x 128) and loop controls (4608 x 64, 64 x 2304) have few output rows, so mul_mat_f
starts only 2-4 blocks of 32 rows: with 16 lanes mul_mat_f<float, 32, 16, 8> took 20 us a call, 9.6% of the GPU time
(Nsight Systems, 24 Nov).
  python patch_zmmvfsplit.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "rapidli_mmvf_f32_split" in s:
    raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVF (i.e. ne01 == 1)""",
    """        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // RAPIDLI: an F32 matrix on 9-16 columns in slices of MMVF_MAX_BATCH_SIZE (GGML_CUDA_MMVF_F32_SPLIT=1, see
    // engine/cuda/patch_zmmvfsplit.py)
    {
        static const bool rapidli_mmvf_f32_split = [] {
            const char * e = getenv("GGML_CUDA_MMVF_F32_SPLIT");
            return e != nullptr && atoi(e) != 0;
        }();
        if (rapidli_mmvf_f32_split && src0->type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc)
                && ne11 > MMVF_MAX_BATCH_SIZE && ne11 <= 2*MMVF_MAX_BATCH_SIZE
                && src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1
                && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
                && ggml_cuda_should_use_mmvf(src0, cc, warp_size, MMVF_MAX_BATCH_SIZE)) {
            for (int64_t c0 = 0; c0 < ne11; c0 += MMVF_MAX_BATCH_SIZE) {
                const int64_t nc = std::min<int64_t>(MMVF_MAX_BATCH_SIZE, ne11 - c0);
                ggml_tensor s1 = *src1;
                ggml_tensor d  = *dst;
                s1.ne[1] = nc;
                s1.data  = (char *) src1->data + c0*src1->nb[1];
                s1.nb[2] = s1.nb[1]*nc;
                s1.nb[3] = s1.nb[2];
                d.ne[1]  = nc;
                d.data   = (char *) dst->data + c0*dst->nb[1];
                d.nb[2]  = d.nb[1]*nc;
                d.nb[3]  = d.nb[2];
                d.src[1] = &s1;
                ggml_cuda_mul_mat_vec_f(ctx, src0, &s1, nullptr, &d);
            }
            return;
        }
    }
    // A transposed vector can still use MMVF (i.e. ne01 == 1)""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched", P)
