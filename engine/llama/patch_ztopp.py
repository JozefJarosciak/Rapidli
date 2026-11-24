"""Patch for llama.dll (Rapidli, after patch_greedy.py): top-p for the fused sampler.

    void llama_fq_set_sampling_p(int32_t seq, float temperature, float min_p, float top_p, uint32_t seed);   (exported)
sets a sequence's sampling as llama_fq_set_sampling does, plus top_p (between 0 and 1; 1 = off). It goes to the card as
the fourth number of the row's parameters, which our ggml-cuda.dll reads (engine/cuda/patch_ztopp.py; an older one
ignores it and samples without top-p). llama_fq_set_sampling, as before, sets top_p to off. rapidli-engine looks the new
function up when it starts, so an older llama.dll still runs it (without top-p).
  python patch_ztopp.py        (run after patch_greedy.py on the source in src/)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NL = chr(10)


def patch(name, pairs):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)
    if "RAPIDLI-TOPP" in s:
        raise SystemExit("already patched")
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline=NL).write(s)


patch("llama-graph.cpp", [
    ("""    float    min_p = 0.0f;
    uint32_t seed  = 0;
    uint32_t draws = 0;
};""",
     """    float    min_p = 0.0f;
    uint32_t seed  = 0;
    uint32_t draws = 0;
    float    top_p = 1.0f;   // RAPIDLI-TOPP (patch_ztopp.py): 1 = off
};"""),
    ("""    fq_sample_table[seq].temp  = temp;
    fq_sample_table[seq].min_p = min_p;
    fq_sample_table[seq].seed  = seed;
    fq_sample_table[seq].draws = 0;
}""",
     """    fq_sample_table[seq].temp  = temp;
    fq_sample_table[seq].min_p = min_p;
    fq_sample_table[seq].seed  = seed;
    fq_sample_table[seq].draws = 0;
    fq_sample_table[seq].top_p = 1.0f;
}
// RAPIDLI-TOPP: the same with top-p (see patch_ztopp.py)
FQ_EXPORT void llama_fq_set_sampling_p(int32_t seq, float temp, float min_p, float top_p, uint32_t seed) {
    if (seq < 0 || seq >= LLAMA_MAX_SEQ) {
        return;
    }
    llama_fq_set_sampling(seq, temp, min_p, seed);
    fq_sample_table[seq].top_p = (top_p > 0.0f && top_p < 1.0f) ? top_p : 1.0f;
}"""),
    ("""                memcpy(&data[4*r + 2], &z, sizeof(z));""",
     """                memcpy(&data[4*r + 2], &z, sizeof(z));
                data[4*r + 3] = c.top_p < 1.0f ? c.top_p : 0.0f;   // RAPIDLI-TOPP: 0 = no top-p"""),
])
print("patched: llama_fq_set_sampling_p (top-p)")
