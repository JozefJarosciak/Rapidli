"""Patch for ggml-cuda (our copy in llama-src/): a per-operation timer.
With the environment variable GGML_CUDA_OPTIME=<n> (and GGML_CUDA_DISABLE_GRAPHS=1, or the kernels are replayed as one
CUDA graph and cannot be told apart) the backend waits for the card after every operation, adds up the time per kind of
operation, and prints a table to the log every <n> graph evaluations. For measurement only: the waiting slows the run.
  python patch_optime.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "fq_optime" in s:
    raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph,""",
    """// RAPIDLI: per-operation timer (GGML_CUDA_OPTIME=<print every n graphs>)
#include <chrono>
#include <map>
#include <string>
struct fq_optime {
    int every = getenv("GGML_CUDA_OPTIME") ? atoi(getenv("GGML_CUDA_OPTIME")) : 0;
    std::map<std::string, std::pair<double, long>> acc;
    long graphs = 0;
    double t_start = 0;
};
static fq_optime & fq_prof() { static fq_optime p; return p; }
static double fq_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void fq_add(const ggml_tensor * n, bool fused, double ms) {
    std::string k = ggml_op_name(n->op);
    if (n->op == GGML_OP_MUL_MAT && n->src[0]) { k += std::string(" ") + ggml_type_name(n->src[0]->type); }
    if (n->op == GGML_OP_CPY || n->op == GGML_OP_GET_ROWS || n->op == GGML_OP_CONT) { k += ggml_nbytes(n) >= (1u << 20) ? " big" : " small"; }
    if (fused) { k += " (fused)"; }
    auto & e = fq_prof().acc[k];
    e.first += ms; e.second += 1;
}
static void fq_graph_begin(int n_nodes) {
    auto & p = fq_prof();
    if (p.every <= 0) { return; }
    if (p.graphs > 0 && p.graphs % p.every == 0) {
        double total = 0;
        for (auto & kv : p.acc) { total += kv.second.first; }
        fprintf(stderr, "FQ-OPTIME %ld graphs, %.1f ms in operations per graph, %.1f ms wall per graph (last graph %d nodes)\\n", (long) p.every, total / p.every, (fq_now() - p.t_start) / p.every, n_nodes);
        std::vector<std::pair<double, std::string>> rows;
        for (auto & kv : p.acc) { rows.push_back({ kv.second.first, kv.first }); }
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (size_t i = 0; i < rows.size() && i < 16; ++i) {
            fprintf(stderr, "FQ-OPTIME   %-34s %8.3f ms/graph  %7.1f calls/graph\\n", rows[i].second.c_str(), rows[i].first / p.every, (double) p.acc[rows[i].second].second / p.every);
        }
        fflush(stderr);
        p.acc.clear();
    }
    if (p.graphs % p.every == 0) { p.t_start = fq_now(); }
    p.graphs++;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph,""")

rep("""    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu""",
    """    bool graph_evaluated_or_captured = false;
    fq_graph_begin(cgraph->n_nodes);
    const bool fq_on = fq_prof().every > 0;

    // flag used to determine whether it is an integrated_gpu""")

rep("""                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {""",
    """                double fq_t0 = 0;
                if (fq_on) { CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream())); fq_t0 = fq_now(); }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    if (fq_on) { CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream())); fq_add(node, true, fq_now() - fq_t0); }""")

rep("""                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {""",
    """                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (fq_on) { CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream())); fq_add(node, false, fq_now() - fq_t0); }
                if (!ok) {""")
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched ggml-cuda.cu: per-operation timer")
