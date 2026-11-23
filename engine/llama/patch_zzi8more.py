"""Patch for llama.dll: the 8-bit recurrent state (patch_i8.py, LLAMA_RS_I8) for the other models built on the gated
delta net the way qwen35 is: qwen35moe (Qwen3.6/3.8-35B-A3B), qwen4exp (Qwen3.8-Flash-Next) and qwen3next. Their graph
reshaped the state to [S_v, S_v, H, n_seqs] like qwen35's did, which stops an 8-bit state on GGML_ASSERT(ggml_nelements(a)
== ne0*ne1) (20 Nov 2026); the delta net (delta-net-base.cpp, patched by patch_i8.py) already takes the 8-bit view.
The 8-bit state is half the bytes of the 16-bit one: Qwen3.8-35B-A3B fitted 96 lanes at 16 bits (3.4 GB of state).
  python patch_zzi8more.py        (after patch_i8.py: named to come last)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

OLD = """    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);"""
NEW = """    // Rapidli (patch_zzi8more.py, as patch_i8.py for qwen35): the 8-bit state stays a [bytes per cell, n_seqs] view
    const bool state_i8 = ssm_states_all->type == GGML_TYPE_I8;
    GGML_ASSERT(!state_i8 || head_v_dim == 128);
    ggml_tensor * state = build_rs(inp, ssm_states_all, state_i8 ? (int32_t) ssm_states_all->ne[0] : (int32_t) hparams.n_embd_s(), n_seqs);
    if (!state_i8) {
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    }"""

done = []
for name in ("qwen35moe.cpp", "qwen4exp.cpp", "qwen3next.cpp"):
    p = os.path.join(HERE, "src", "models", name)
    s = open(p, encoding="utf8").read()
    if s.count(OLD) == 1:
        open(p, "w", encoding="utf8", newline="\n").write(s.replace(OLD, NEW))
        done.append(name)
print("patched: 8-bit gated-delta-net state also for " + ", ".join(done))
