"""Patch for ggml-cuda (llama-src/): the largest batch for which an f32 matrix uses the matrix-vector kernel (MMVF) on
NVIDIA GPUs with tensor cores: up to MMVF_MAX_BATCH_SIZE (8) instead of 3 (GGML_CUDA_MMVF_F32_MAX=3 restores stock).

Inkling 2 keeps its expert router (2304 x 128) and its loop controls (4608 x 64, 64 x 2304) in f32. With 8 lanes
mul_mat_f<float, 32, 8, 8> ran 152 times a pass at 15.4 us, ~80 GB/s and 10.7% of the GPU time (Nsight Systems,
24 Nov), where one lane's MMVF reads the same matrix in a few us. Inkling-2-Mini Q2_K_XL passes: 4 lanes 18.3 -> 16.5 ms,
8 lanes 22.1 -> 20.8 ms, 16 unchanged; Qwen3.8-35B-A3B and Ternary-Bonsai-2-2B at 4 and 8 lanes unchanged within noise.
MMVF computes in full fp32 (as at 1-3 lanes), so token choices can differ slightly from the tiled kernel.
  python patch_zmmvf32.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "mmvf.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "rapidli_mmvf_f32_max" in s:
    raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, int cc, int warp_size, int64_t ne11) {""",
    """// RAPIDLI: GGML_CUDA_MMVF_F32_MAX, the largest batch an f32 matrix takes to MMVF on NVIDIA with tensor cores
// (see engine/cuda/patch_zmmvf32.py)
static int rapidli_mmvf_f32_max() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_MMVF_F32_MAX");
        const int n = e ? atoi(e) : MMVF_MAX_BATCH_SIZE;
        return n < 1 ? 1 : (n > MMVF_MAX_BATCH_SIZE ? MMVF_MAX_BATCH_SIZE : n);
    }();
    return v;
}

bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, int cc, int warp_size, int64_t ne11) {""")
rep("""        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }""",
    """        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= rapidli_mmvf_f32_max();
                }""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched", P)
