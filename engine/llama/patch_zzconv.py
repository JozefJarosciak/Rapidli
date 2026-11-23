"""Rapidli: in qwen35 the convolution of the gated-delta-net layers is put into the graph right after its input (the
state and the new token's channels joined), before the state is stored back. The three operations then follow one
another, and ggml-cuda (engine/cuda/patch_convfuse.py) runs them as one kernel that never writes the joined input:
the same arithmetic in the same order, without two passes over memory. Nothing else changes: the same operations on
the same tensors, only their order in the graph."""
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


patch("models/qwen35.cpp", [
    ("""    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);
""",
     """    // RAPIDLI: the state joined with the new channels, the convolution (with its silu) and only then the store of the
    // new state, one after another in the graph, so that ggml-cuda can run the three as one kernel
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);
    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);
    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, ggml_transpose(ctx0, qkv_mixed), 0);
    cb(conv_input, "conv_input", il);
    ggml_tensor * conv_output_early = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    ggml_build_forward_expand(gf, conv_output_early);
    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);
""", 1),
    ("""    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);
""",
     """    ggml_tensor * conv_output_silu = conv_output_early;
    cb(conv_output_silu, "conv_output_silu", il);
""", 1),
])
