"""Patch for llama.dll: llama_fq_fork (patch_zfork.py) for models whose memory has an indexer cache besides the attention
cache and the recurrent state (llama_memory_hybrid_idx: Qwen3.8-Flash-Next, arch qwen4exp, with its sparse attention
indexer and pooled keys). The fork copied the attention cells and the recurrent rows but not the indexer's cells, so a
lane that continued from a parked prompt had no pooled keys and the next pass stopped on
llama-memory-hybrid-idx.cpp: GGML_ASSERT(r >= 0) (20 Nov 2026). Now the indexer cache is copied as the memory's own
seq_cp copies it (its cells tagged with the new sequence, the pools of the new sequence rebuilt from position 0).
  python patch_zzidx.py        (after patch_zfork.py: named to come last)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))


def edit(name, pairs=(), append=""):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline="\n").write(s + append)


edit("llama-memory-hybrid-idx.h", [
    ("    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer",
     "    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer\n\n"
     "    // Rapidli (patch_zzidx.py): the indexer cache of seq_src copied to the sequences seq_dst[0..n_dst), as seq_cp\n"
     "    // copies it; llama_fq_fork copies the attention cells and the recurrent rows itself\n"
     "    void fq_fork_idx(llama_seq_id seq_src, const llama_seq_id * seq_dst, int32_t n_dst);"),
])

edit("llama-memory-hybrid-idx.cpp", append=r'''

// Rapidli (patch_zzidx.py): see the header
void llama_memory_hybrid_idx::fq_fork_idx(llama_seq_id seq_src, const llama_seq_id * seq_dst, int32_t n_dst) {
    if (!mem_idx) {
        return;
    }
    for (int32_t k = 0; k < n_dst; ++k) {
        mem_idx->seq_rm(seq_dst[k], -1, -1);
        mem_idx->seq_cp(seq_src, seq_dst[k], -1, -1);
        mem_idx_stale_set(seq_dst[k], 0);
    }
}
''')

edit("llama-memory-hybrid.cpp", [
    ("""    for (int32_t k = 0; k < n_dst; ++k) {
        hyb->get_mem_attn()->seq_rm(seq_dst[k], -1, -1);
        hyb->get_mem_attn()->seq_cp(seq_src, seq_dst[k], -1, -1);
    }
    return hyb->get_mem_recr()->fq_fork(seq_src, seq_dst, n_dst) ? 1 : 0;""",
     """    for (int32_t k = 0; k < n_dst; ++k) {
        hyb->get_mem_attn()->seq_rm(seq_dst[k], -1, -1);
        hyb->get_mem_attn()->seq_cp(seq_src, seq_dst[k], -1, -1);
    }
    // (patch_zzidx.py: a memory with an indexer cache copies it too)
    if (auto * hi = dynamic_cast<llama_memory_hybrid_idx *>(hyb)) {
        hi->fq_fork_idx(seq_src, seq_dst, n_dst);
    }
    return hyb->get_mem_recr()->fq_fork(seq_src, seq_dst, n_dst) ? 1 : 0;"""),
    ('#include "llama-memory-hybrid.h"\n', '#include "llama-memory-hybrid.h"\n#include "llama-memory-hybrid-idx.h"   // patch_zzidx.py\n'),
])
print("llama_fq_fork copies the indexer cache too (patch_zzidx.py)")
