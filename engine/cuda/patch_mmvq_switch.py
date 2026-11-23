"""Patch for ggml-cuda (llama-src): GGML_CUDA_MMVQ_Q2_0=<n> sets the largest batch for which Q2_0 uses the vector kernel
(MMVQ) on an RTX 40 card; larger batches use MMQ. The default stays 8: measured 20 Nov 2026 on an RTX 4090 with
Ternary-Bonsai-2-27B, MMQ from 3 or 5 lanes up made a pass of 3 to 6 lanes 1 to 3 ms slower. A switch for measuring.
  python patch_mmvq_switch.py"""
import io, os
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"),
                 "ggml", "src", "ggml-cuda", "mmvq.cu")
s = io.open(P, encoding="utf8").read().replace("\r\n", "\n")
if "GGML_CUDA_MMVQ_Q2_0" in s:
    raise SystemExit("already patched")
old = """        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;"""
new = """        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_0: {      // RAPIDLI: GGML_CUDA_MMVQ_Q2_0=<n> sets where MMQ takes over (measured, see the speed notes)
                static const int n_q2_0 = getenv("GGML_CUDA_MMVQ_Q2_0") ? atoi(getenv("GGML_CUDA_MMVQ_Q2_0")) : MMVQ_MAX_BATCH_SIZE;
                return ne11 <= n_q2_0;
            }
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;"""
assert s.count(old) == 1, "the place to patch was not found"
io.open(P, "w", encoding="utf8", newline="\n").write(s.replace(old, new))
print("patched", P)
