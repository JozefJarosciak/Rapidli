"""Patch for ggml-cuda (llama-src): MMQ tiles of 136 to 160 columns for Q2_0, so that a pass of up to 160 lanes unpacks each weight tile once
(with at most 128 columns, 156 lanes were two tiles of 80, and every tile of 2-bit weights was unpacked twice). Measured 19 Nov 2026 on an RTX 4090, Ternary-Bonsai-2-27B: a pass of 144 lanes 66.8 -> 62.7 ms, 156
lanes 69.2 -> 67.4 ms, 128 lanes unchanged. GGML_CUDA_MMQ_NARROW=1 at run time goes back to tiles of at most 128.
  python patch_mmq_wide.py"""
import io, os
D = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"), "ggml", "src", "ggml-cuda") + os.sep
WIDE = [136, 144, 152, 160]


def edit(name, pairs):
    p = D + name
    s = io.open(p, encoding="utf8").read().replace("\r\n", "\n")
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:70])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf8", newline="\n").write(s)


# the configurations
edit("mmq-config-ampere.cuh", [(
    "    CASE(GGML_TYPE_Q2_0, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);\n",
    "    CASE(GGML_TYPE_Q2_0, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);\n"
    + "".join(f"    CASE(GGML_TYPE_Q2_0, 256, 1, 128, {j}, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);   // RAPIDLI-WIDE\n" for j in WIDE))])

# the choice of the width
edit("mmq.cu", [(
    "    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {",
    "    // (RAPIDLI-WIDE: Q2_0 also has tiles of up to 160 columns)\n"
    "    const int J_max = (type == GGML_TYPE_Q2_0 && getenv(\"GGML_CUDA_MMQ_NARROW\") == nullptr) ? 160 : 128;\n"
    "    for (int J = 8; J <= J_max && ntiles_J_best > 1; J += 8) {")])

# the switch: the wide cases exist for Q2_0 only
cases = "".join(f"        case {j}:\n            if constexpr (type == GGML_TYPE_Q2_0 && !fallback) {{ launch_mul_mat_q<type, {j}, fallback, prec_src1, has_fusion>(ctx, args, stream); break; }}\n            GGML_ABORT(\"no wide MMQ tile for this type\");\n" for j in WIDE)
edit("mmq.cuh", [('(J_)              <= 128, "bad J"', '(J_)              <= 160, "bad J"'), (
    "        case 128:\n            launch_mul_mat_q<type, 128, fallback, prec_src1, has_fusion>(ctx, args, stream);\n            break;\n",
    "        case 128:\n            launch_mul_mat_q<type, 128, fallback, prec_src1, has_fusion>(ctx, args, stream);\n            break;\n" + cases)])
print("wide MMQ tiles for Q2_0:", WIDE)
