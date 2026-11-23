"""Patch for llama.dll (Rapidli): the gated-delta-net state kept as 8-bit codes (LLAMA_RS_I8=1).

EXPERIMENTAL, CHANGES THE ANSWERS. Each cell of the state cache holds, per block,
    H * S_v * S_v  int8 codes          (S_v = 128: the state is stored column by column, a column is the vector that is
                                         multiplied with the key and with the query)
    H * S_v        float32 scales      (one per column: largest magnitude of the column / 127)
which is 33/32 bytes per state number instead of 2 (IEEE half) or 4. The kernel on the card (ggml-cuda, see
cuda-build/patch_i8.py) reads the codes, updates the state in float32, computes the output from the float32 state and
writes the codes back in place with stochastic rounding. There is no copy of the state through the graph, so every
pass has to read the state in place (LLAMA_RS_DIRECT=1 and the lanes in order, which is what rapidli-engine does).
  python patch_i8.py        (run after patch_direct.py and patch_f16.py on the source in src/)"""
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


patch("llama-graph.h", [(
    """bool llm_rs_f16();
""",
    """bool llm_rs_f16();
// RAPIDLI: the gated-delta-net state is kept as 8-bit codes with one scale per column (LLAMA_RS_I8=1)
bool llm_rs_i8();
""")])

patch("llama-graph.cpp", [
    ("""bool llm_rs_direct(const llama_ubatch & ubatch, const llama_memory_recurrent_context * mctx, uint32_t n_rs_seq) {""",
     """bool llm_rs_i8() {
    static const bool enabled = getenv("LLAMA_RS_I8") != nullptr && getenv("LLAMA_RS_I8")[0] == '1';
    return enabled;
}

// the longest run of tokens of one sequence that is still read with the state in place (LLAMA_RS_DIRECT_MAX, default 64)
static uint32_t llm_rs_direct_max() {
    static const uint32_t n = getenv("LLAMA_RS_DIRECT_MAX") != nullptr ? (uint32_t) std::max(1, atoi(getenv("LLAMA_RS_DIRECT_MAX"))) : 64;
    return n;
}

bool llm_rs_direct(const llama_ubatch & ubatch, const llama_memory_recurrent_context * mctx, uint32_t n_rs_seq) {"""),
    ("""    if (ubatch.n_seq_tokens != 1 && !(llm_rs_f16() && ubatch.n_seq_tokens <= 64)) {""",
     """    if (ubatch.n_seq_tokens != 1 && !((llm_rs_f16() || llm_rs_i8()) && ubatch.n_seq_tokens <= llm_rs_direct_max())) {"""),
    # the graph built at start-up to size the buffers is not "direct", but it is never run: the 8-bit state is always a
    # view of the cache, and a pass that is not direct stops in set_input (llm_rs_prepare) instead
    ("""    if (inp->direct && !get_state_rows) {
        ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);""",
     """    if ((inp->direct || s->type == GGML_TYPE_I8) && !get_state_rows) {
        ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);"""),
    ("""void llm_rs_prepare(const llama_memory_recurrent_context * mctx, bool direct) {
""",
     """void llm_rs_prepare(const llama_memory_recurrent_context * mctx, bool direct) {
    if (!direct && llm_rs_i8()) {
        GGML_ABORT("RAPIDLI: the 8-bit state is read and written in place only: LLAMA_RS_DIRECT=1, every lane in every pass, lanes in order");
    }
"""),
])

patch("llama-model.cpp", [(
    """/* recurrent_type_v  */ llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,""",
    """/* recurrent_type_v  */ llm_rs_i8() ? GGML_TYPE_I8 : llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,""", 2)])

patch("llama-memory-recurrent.cpp", [(
    """        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);""",
    """        // RAPIDLI i8: the codes, then one float32 scale per column of 128 codes (33/32 bytes per state number)
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, type_s == GGML_TYPE_I8 ? hparams.n_embd_s() + hparams.n_embd_s() / 32 : hparams.n_embd_s(), n_rows);""")])

patch("models/qwen35.cpp", [(
    """    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);""",
    """    // RAPIDLI i8: the 8-bit state stays a [bytes per cell, n_seqs] view of the cache, the kernel knows its layout
    const bool state_i8 = ssm_states_all->type == GGML_TYPE_I8;
    GGML_ASSERT(!state_i8 || head_v_dim == 128);
    ggml_tensor * state = build_rs(inp, ssm_states_all, state_i8 ? (int32_t) ssm_states_all->ne[0] : (int32_t) hparams.n_embd_s(), n_seqs);
    if (!state_i8) {
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    }""")])

patch("models/delta-net-base.cpp", [
    ("""    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);
""",
     """    GGML_ASSERT(s->type == GGML_TYPE_I8 || (s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs));
"""),
    ("""        s_arg = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);""",
     """        s_arg = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S_v, S_v, H_v, n_seqs);"""),
    ("""    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);
""",
     """    const int64_t S_v          = v->ne[0];
    const int64_t H_v          = v->ne[1];
    const int64_t n_seqs       = v->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (s->type == GGML_TYPE_I8) {
        // RAPIDLI i8: the kernel writes the new state into the cache itself (in place), nothing to copy
        GGML_ASSERT(!keep);
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        cb(attn_out.first, "attn_output", il);
        return attn_out.first;
    }

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);
"""),
])
print("patched: 8-bit gated-delta-net state (llama-graph, llama-model, llama-memory-recurrent, qwen35, delta-net-base)")
