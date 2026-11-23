"""RAPIDLI version 2: the 8-bit in-place recurrent state becomes a property of a context, not of the process.
A context with rollback snapshots (n_rs_seq > 0, used to verify drafted tokens) keeps the ordinary 32-bit state and the
ordinary path; contexts without snapshots (the lanes) keep the 8-bit state read and written in place. Both can live
in one process and share the weights."""
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


def patch(name, pairs):
    p = os.path.join(SRC, name)
    s = open(p, encoding="utf8").read()
    for a, b, n in pairs:
        c = s.count(a)
        assert c == n, (name, a[:70], c)
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline="\n").write(s)
    print("patched", name)


patch("llama-model.cpp", [
    ("/* recurrent_type_v  */ llm_rs_i8() ? GGML_TYPE_I8 : llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,",
     "/* recurrent_type_v  */ (llm_rs_i8() && cparams.n_rs_seq == 0) ? GGML_TYPE_I8 : ((llm_rs_f16() && cparams.n_rs_seq == 0) || (cparams.n_rs_seq > 0 && llm_rs_spec_f16())) ? GGML_TYPE_F16 : GGML_TYPE_F32,", 2),
    ("/* recurrent_type_s  */ llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,",
     "/* recurrent_type_s  */ ((llm_rs_f16() && cparams.n_rs_seq == 0) || (cparams.n_rs_seq > 0 && llm_rs_spec_f16())) ? GGML_TYPE_F16 : GGML_TYPE_F32,", 2),
    ('#include "llama-graph.h" // RAPIDLI: llm_rs_f16()', '''#include "llama-graph.h" // RAPIDLI: llm_rs_f16()
// RAPIDLI version 2: LLAMA_RS_SPEC_F16=1 keeps the state of contexts with rollback snapshots in 16 bits
static bool llm_rs_spec_f16() { static const bool e = getenv("LLAMA_RS_SPEC_F16") != nullptr && getenv("LLAMA_RS_SPEC_F16")[0] == '1'; return e; }''', 1),
])

patch("llama-graph.cpp", [
    ('''    if (!direct && llm_rs_i8()) {
        GGML_ABORT(''', '''    bool state_i8 = false;               // RAPIDLI: only contexts without snapshots have an 8-bit state
    for (int32_t il = 0; il < mctx->get_n_s_l(); ++il) {
        const ggml_tensor * s = mctx->get_s_l(il);
        if (s) { state_i8 = s->type == GGML_TYPE_I8; break; }
    }
    if (!direct && state_i8) {
        GGML_ABORT(''', 1),
])
