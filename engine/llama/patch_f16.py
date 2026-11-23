"""Patch 2 for llama.dll (Rapidli): keep the gated-delta-net recurrent state in 16-bit floats.

The state of one stream is 48 blocks x 48 heads x 128 x 128 numbers: 144 MiB as float32. Stored as IEEE half it is
72 MiB, so a pass moves half the bytes per stream and many more streams fit on the card. The arithmetic stays in
float32: the CUDA kernel (our ggml-cuda.dll, cuda-build/patch_f16.py) converts on load and on store. LOSSY: every
update rounds the state to 11 significant bits. Opt-in: environment variable LLAMA_RS_F16=1 (needs our ggml-cuda.dll).
  python patch_f16.py        (run after patch_direct.py on the pristine source in src/)"""
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


patch("llama-model.cpp", [
    ("""#include "llama-memory-recurrent.h"
""",
     """#include "llama-memory-recurrent.h"
#include "llama-graph.h" // RAPIDLI: llm_rs_f16()
"""),
    ("""/* recurrent_type_s  */ GGML_TYPE_F32,""",
     """/* recurrent_type_s  */ llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,""", 2),
    # the plain hybrid memory (the one qwen35 gets) names the same two arguments "k" and "v": the second is the state
    ("""/* recurrent_type_v  */ GGML_TYPE_F32,""",
     """/* recurrent_type_v  */ llm_rs_f16() ? GGML_TYPE_F16 : GGML_TYPE_F32,""", 2),
])

patch("models/delta-net-base.cpp", [(
    """    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);""",
    """    // RAPIDLI: a 16-bit state (a view of the cache) is read by the kernel directly; the op builder only takes float32,
    // so it is given a float32 stand-in of the same shape (never part of the graph) and the real state is put in after
    ggml_tensor * s_arg = s;
    if (s->type != GGML_TYPE_F32) {
        s_arg = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
    }
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s_arg, /*K=*/1);
    if (s_arg != s) {
        result->src[5] = s;
    }""")])
print("patched: 16-bit recurrent state (llama-model, delta-net-base)")
