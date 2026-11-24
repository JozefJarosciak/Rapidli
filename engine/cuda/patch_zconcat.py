"""Patch for ggml-cuda (llama-src/): a concat of non-contiguous tensors with a short first dimension, one thread per
element.

The generic kernel (concat_non_cont) starts one block of 256 threads per row (dst ne1 x ne2 x ne3) and loops over ne0
within it. With a short ne0 almost all threads idle: Inkling 2 joins its block states along dim 0 (ne0 = up to 9, a
grid of 2304 x 124 blocks with 124 lanes) and each short convolution's history with its input (ne0 = 4, 1024 x 124):
96 + 96 calls a pass at 162 and 72 us, 22 ms of a 98 ms pass (Nsight Systems, 23 Nov). For ne0 < 64 the copy is now
spread over all elements; it moves the same bytes, so the result is the same.
  python patch_zconcat.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "concat.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "concat_non_cont_flat" in s:
    raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {""",
    """// RAPIDLI: one thread per dst element, for a short ne0 (see engine/cuda/patch_zconcat.py)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont_flat(
        const char * src0, const char * src1, char * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t n,
        const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3) {
    const int64_t idx = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    int64_t t = idx;
    const int64_t i0 = t % ne0; t /= ne0;
    const int64_t i1 = t % ne1; t /= ne1;
    const int64_t i2 = t % ne2;
    const int64_t i3 = t / ne2;

    const T * x;
    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
        x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
    } else if constexpr (dim == 0) {
        x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
    } else if constexpr (dim == 1) {
        x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
    } else if constexpr (dim == 2) {
        x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
    } else {
        x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
    }
    *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = *x;
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {""")

rep("""        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);""",
    """        GGML_ASSERT(!ggml_is_quantized(src0->type));

        if (dst->ne[0] < 64) {
            // RAPIDLI: a short first dimension, one thread per element (see concat_non_cont_flat)
            const int64_t n = ggml_nelements(dst);
            const int64_t n_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
            auto launch_flat = [&](auto dim_c) {
                concat_non_cont_flat<T, dim_c><<<n_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], n,
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0: launch_flat(std::integral_constant<int, 0>{}); break;
                case 1: launch_flat(std::integral_constant<int, 1>{}); break;
                case 2: launch_flat(std::integral_constant<int, 2>{}); break;
                case 3: launch_flat(std::integral_constant<int, 3>{}); break;
                default: GGML_ABORT("Invalid dim: %d", dim);
            }
            return;
        }

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched", P)
