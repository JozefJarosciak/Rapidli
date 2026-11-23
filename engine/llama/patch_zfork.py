"""Patch for llama.dll (Rapidli, version 2): fork a sequence - its whole memory copied to other sequences.

Version 2 answers ONE question with many lanes: a lane reads the question (and writes a plan), then its memory is
forked to N lanes which continue from that point, each with its own next tokens. Reading the shared text N times would
cost as much as writing it (about 0.4 ms per token and lane); the fork costs a few milliseconds.

  llama_fq_fork(ctx, seq_src, seq_dst[], n)        exported, 1 = done
    attention cache (one cache shared by all lanes, --kvu 1): the cells of seq_src are tagged with the new sequences
        too - no data moves
    recurrent state: the row of seq_src in every state tensor (r, s and the PLE history) is copied to the rows of the
        new sequences ON THE DEVICE, as stored (float32, 16-bit or 8-bit codes with scales). Every new sequence gets the
        cell with its own number, as LLAMA_RS_DIRECT=1 needs. When the new sequences are neighbouring lanes, the
        first gets the source's row and the others are filled by doubling (1, 2, 4, ... rows per copy).
What the new sequences held before is dropped. The source must own its cell.
  python patch_zfork.py        (run after the other patches on the source in src/; named z to come last)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))


def edit(name, pairs=(), append=""):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline="\n").write(s + append)


edit("llama-memory-recurrent.h", [
    ("    bool find_slot(const llama_ubatch & ubatch);",
     "    bool find_slot(const llama_ubatch & ubatch);\n\n"
     "    // RAPIDLI v2: the state of seq_src copied to the sequences seq_dst[0..n_dst) on the device\n"
     "    bool fq_fork(llama_seq_id seq_src, const llama_seq_id * seq_dst, int32_t n_dst);"),
])

edit("llama-memory-recurrent.cpp", append=r'''

// RAPIDLI v2 (patch_zfork.py): the whole recurrent state of one sequence copied to other sequences, on the device
bool llama_memory_recurrent::fq_fork(llama_seq_id seq_src, const llama_seq_id * seq_dst, int32_t n_dst) {
    if (seq_src < 0 || (uint32_t) seq_src >= size || cells[seq_src].tail < 0) {
        return false;
    }
    const int32_t cs = cells[seq_src].tail;
    if (cells[cs].seq_id.size() != 1) {
        return false;
    }
    bool run = true;                                 // the new sequences are cells d0, d0+1, d0+2, ... in this order
    for (int32_t k = 0; k < n_dst; ++k) {
        const llama_seq_id d = seq_dst[k];
        if (d < 0 || (uint32_t) d >= size || d == seq_src || d == cs) {
            return false;
        }
        if (cells[d].tail >= 0) {                    // drop what the sequence held
            auto & old = cells[cells[d].tail];
            old.seq_id.erase(d);
            if (old.seq_id.empty()) {
                old.pos = -1;
                old.src = -1;
                used -= 1;
            }
            cells[d].tail = -1;
        }
        auto & c = cells[d];                         // the cell with the sequence's own number
        if (!c.is_empty()) {
            return false;
        }
        c.pos    = cells[cs].pos;
        c.src    = d;                                // nothing to move or to clear in the next pass
        c.src0   = d;
        c.rs_max = cells[cs].rs_max;
        c.seq_id.insert(d);
        cells[d].tail = d;
        used += 1;
        run = run && d == seq_dst[0] + k;
    }
    auto copy_rows = [&](ggml_tensor * t, int64_t from, int64_t to, int64_t cnt) {
        if (t == nullptr) {
            return;
        }
        for (uint32_t g = 0; g <= n_rs_seq; ++g) {
            ggml_tensor a = *t;                      // views of cnt rows of the cache
            ggml_tensor b = *t;
            a.ne[1] = cnt;
            b.ne[1] = cnt;
            a.data = (char *) t->data + (from + (int64_t) g * size) * t->nb[1];
            b.data = (char *) t->data + (to   + (int64_t) g * size) * t->nb[1];
            ggml_backend_tensor_copy(&a, &b);
        }
    };
    for (size_t il = 0; il < r_l.size(); ++il) {
        for (ggml_tensor * t : { r_l[il], s_l[il], il < p_l.size() ? p_l[il] : nullptr }) {
            copy_rows(t, cs, seq_dst[0], 1);
            if (run) {
                int64_t filled = 1;                  // rows of the run that hold the state already
                while (filled < (int64_t) n_dst) {
                    const int64_t cnt = std::min<int64_t>(filled, (int64_t) n_dst - filled);
                    copy_rows(t, seq_dst[0], seq_dst[0] + filled, cnt);
                    filled += cnt;
                }
            } else {
                for (int32_t k = 1; k < n_dst; ++k) {
                    copy_rows(t, cs, seq_dst[k], 1);
                }
            }
        }
    }
    return true;
}
''')

edit("llama-memory-hybrid.cpp", append=r'''

// RAPIDLI v2 (patch_zfork.py): fork a sequence. The attention cells are shared (one cache for all lanes), the
// recurrent state is copied on the device.
#ifndef FQ_EXPORT
#if defined(_WIN32)
#  define FQ_EXPORT extern "C" __declspec(dllexport)
#else
#  define FQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif
#endif
FQ_EXPORT int32_t llama_fq_fork(llama_context * ctx, int32_t seq_src, const int32_t * seq_dst, int32_t n_dst) {
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    if (n_dst <= 0) {
        return 0;
    }
    if (hyb == nullptr) {
        // any other memory (an attention cache, also with a sliding window): llama.cpp's own sequence copy
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem == nullptr) {
            return 0;
        }
        for (int32_t k = 0; k < n_dst; ++k) {
            llama_memory_seq_rm(mem, seq_dst[k], -1, -1);
            llama_memory_seq_cp(mem, seq_src, seq_dst[k], -1, -1);
        }
        return 1;
    }
    for (int32_t k = 0; k < n_dst; ++k) {
        hyb->get_mem_attn()->seq_rm(seq_dst[k], -1, -1);
        hyb->get_mem_attn()->seq_cp(seq_src, seq_dst[k], -1, -1);
    }
    return hyb->get_mem_recr()->fq_fork(seq_src, seq_dst, n_dst) ? 1 : 0;
}
''')
print("patch_zfork: sequence fork added")
