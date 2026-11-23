"""Patch for llama.dll: the MoE cache's figures while it runs. With LLAMA_MOE_CACHE_LOG=<n> the cache prints, every n
layer lookups, its hits, misses (and how many the CPU took), the MB it uploaded, and the time spent in its lookup
(waiting for the router's choice on the card, then queueing the uploads) since the last line. Without the variable
nothing changes; llama.cpp prints these figures only when a context is freed, which a stopped engine never does.
Also LLAMA_MOE_CACHE_MAX_BATCH=<n>: passes of up to n tokens use the cache (llama.cpp: 32).
  python patch_zzmoelog.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))


def edit(name, pairs=(), append=""):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline="\n").write(s + append)


edit("llama-moe-cache.cpp", [
    ("#include <algorithm>\n", "#include <algorithm>\n#include <chrono>   // patch_zzmoelog.py\n"),
    ("""        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];""",
     """        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        // (patch_zzmoelog.py) the time of this lookup, and the figures every LLAMA_MOE_CACHE_LOG lookups
        static const long long log_every = getenv("LLAMA_MOE_CACHE_LOG") ? atoll(getenv("LLAMA_MOE_CACHE_LOG")) : 0;
        const auto t_lookup = std::chrono::steady_clock::now();
        struct lookup_timer {
            impl * self; std::chrono::steady_clock::time_point t0;
            ~lookup_timer() {
                if (log_every <= 0) { return; }
                self->log_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (++self->log_n % log_every == 0) {
                    const stats a = self->stats_small, b = self->stats_large;
                    const size_t hits = a.hits + b.hits - self->log_last.hits, miss = a.misses + b.misses - self->log_last.misses;
                    const size_t cpu = a.cpu + b.cpu - self->log_last.cpu, bytes = a.bytes + b.bytes - self->log_last.bytes;
                    LLAMA_LOG_INFO("llama_moe_cache: last %lld lookups: hit rate %.1f%% (%zu hits, %zu misses, %zu on the CPU), uploaded %.1f MiB, %.1f ms in lookups\\n",
                        log_every, 100.0 * hits / std::max<size_t>(1, hits + miss), hits, miss, cpu, bytes / 1048576.0, self->log_us / 1000.0);
                    self->log_last.hits = a.hits + b.hits; self->log_last.misses = a.misses + b.misses;
                    self->log_last.cpu = a.cpu + b.cpu; self->log_last.bytes = a.bytes + b.bytes;
                    self->log_us = 0;
                }
            }
        } timer{this, t_lookup};"""),
    ("""    static constexpr int64_t max_batch = 32;""",
     """    // (patch_zzmoelog.py) LLAMA_MOE_CACHE_MAX_BATCH: passes of up to this many tokens use the cache (llama.cpp: 32).
    // With the experts' products on the vector kernel in slices (engine/cuda/patch_zmmidslice.py) a pass of 48 or 64
    // lanes gains from the cache too
    static int64_t max_batch_env() {
        static const int64_t v = getenv("LLAMA_MOE_CACHE_MAX_BATCH") ? std::max<long long>(1, atoll(getenv("LLAMA_MOE_CACHE_MAX_BATCH"))) : 32;
        return v;
    }"""),
    ("""        if (n_tokens == 0 || n_tokens > max_batch ||""", """        if (n_tokens == 0 || n_tokens > max_batch_env() ||"""),
    ("""    void log_stats() const {""",
     """    // (patch_zzmoelog.py) the periodic figures
    long long log_n = 0;
    double    log_us = 0.0;
    stats     log_last = {};

    void log_stats() const {"""),
])
print("MoE cache figures every LLAMA_MOE_CACHE_LOG lookups (patch_zzmoelog.py)")
