"""Patch for ggml-cuda.dll (Rapidli): clearing the state of a starting lane without 96 waits.

A lane that starts a request has its recurrent state cleared in every layer (llama-build/patch_direct.py, 96 tensors).
ggml's memset of part of a tensor waits for the card after every call; 96 waits are about 3 ms, in every pass in which
a request starts. With GGML_CUDA_LAZY_MEMSET=1 (set by rapidli-engine) the memsets are only queued and the next graph
computation waits for them once, before it launches anything. They are queued on the calling thread's stream and the
graph is computed by the same thread (llama_decode), so that one wait covers them all; host reads and writes of
tensors use the same stream and wait themselves.
  python patch_memset.py        (edits llama-src/ggml/src/ggml-cuda/ggml-cuda.cu in place, once)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI lazy memset" in s:
    print("already patched")
    raise SystemExit(0)


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}""",
    """// RAPIDLI lazy memset: memsets are queued, the next graph computation waits for them once (GGML_CUDA_LAZY_MEMSET=1)
// (per thread: the memsets wait on the calling thread's own stream, and two threads compute graphs - the lanes and the
// drafting context. One flag for both let a drafting pass clear it and wait on its own stream, and the lanes' next
// pass ran while their memsets were still queued, 22 Nov)
static thread_local bool fq_memset_pending = false;

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    static const bool fq_lazy = getenv("GGML_CUDA_LAZY_MEMSET") != nullptr && getenv("GGML_CUDA_LAZY_MEMSET")[0] == '1';
    if (fq_lazy) {
        fq_memset_pending = true;
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}""")

rep("""static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
""",
    """static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // RAPIDLI lazy memset: the queued memsets must be done before anything of this graph runs
    if (fq_memset_pending) {
        fq_memset_pending = false;
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched: lazy memset in ggml-cuda.cu")
