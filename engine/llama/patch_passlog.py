"""Patch 3 for llama.dll (Rapidli): a log of every pass, for finding where time goes outside steady decoding.

With LLAMA_PASS_LOG=<file> every call of process_ubatch appends one line:
  <microseconds since the first pass> <sequences> <tokens per sequence> <R = graph reused | B = graph built>
The time between two lines is one full cycle (pass + sampling + the server's work between passes).
  python patch_passlog.py        (run after the other patches on the source in src/)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NL = chr(10)


def patch(name, pairs):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)
    for pair in pairs:
        a, b, cnt = pair[0], pair[1], (pair[2] if len(pair) > 2 else 1)
        assert s.count(a) == cnt, (name, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline=NL).write(s)


patch("llama-context.cpp", [
    ("""#include <algorithm>
#include <cinttypes>
""",
     """#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>

// RAPIDLI: one line per pass in the file named by LLAMA_PASS_LOG
static void fq_pass_log(uint32_t n_seqs, uint32_t n_seq_tokens, bool reused) {
    static FILE * f = getenv("LLAMA_PASS_LOG") ? fopen(getenv("LLAMA_PASS_LOG"), "w") : nullptr;
    if (f == nullptr) {
        return;
    }
    static const int64_t t0 = ggml_time_us();
    fprintf(f, "%lld %u %u %c\\n", (long long) (ggml_time_us() - t0), n_seqs, n_seq_tokens, reused ? 'R' : 'B');
    fflush(f);
}
"""),
    ("""        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();
""",
     """        n_reused++;
        fq_pass_log(ubatch.n_seqs, ubatch.n_seq_tokens, true);
    } else {
        fq_pass_log(ubatch.n_seqs, ubatch.n_seq_tokens, false);
        gf_res_prev_active = nullptr;
        res->reset();
"""),
])
# With a unified attention cache the number of cache entries a pass looks at grows by one per stream per token; it is
# rounded up to a multiple of 256 and every change of it rebuilds the graph (every 4th pass at 64 streams).
# LLAMA_KV_PAD=<n> rounds to a larger multiple, so rebuilds become rare.
patch("llama-kv-cache.cpp", [(
    """#include <cstring>
""",
    """#include <cstring>
#include <cstdlib>
"""), (
    """    const uint32_t n_pad = 256;
""",
    """    static const uint32_t n_pad = getenv("LLAMA_KV_PAD") ? (uint32_t) std::max(256, atoi(getenv("LLAMA_KV_PAD"))) : 256; // RAPIDLI
""")])
print("patched: pass log (llama-context), cache padding (llama-kv-cache)")
