"""Patch for ggml-cuda (llama-src): the vector kernel (MMVQ) takes 4 rows of a Q2_0 matrix per block instead of 2 when
2 to 8 tokens are multiplied (RTX 30/40 and newer NVIDIA cards, the "generic" table), so each block reads the q8_1
input once for twice as many rows. Each row is summed in the same order as before, so the results are the same.
Measured 20 Nov 2026 on an RTX 4090 with Ternary-Bonsai-2-27B, alternating, a pass with 5 / 6 / 7 / 8 lanes:
14.7-15.2 / 15.0 / 17.1 / 16.8-17.1 ms before, 13.9-14.0 / 13.4-13.8 / 14.9-15.1 / 14.9-15.1 ms after; 1, 2 and 12
lanes unchanged. Tried and left: 2 rows for one token (11.2 -> 11.3 ms), 8 rows for 5 to 8 tokens (16.5 ms at 8).
  python patch_zmmvqrows.py"""
import io, os
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama-src"),
                 "ggml", "src", "ggml-cuda", "mmvq.cu")
s = io.open(P, encoding="utf8").read().replace("\r\n", "\n")
if "RAPIDLI-MMVQROWS" in s:
    raise SystemExit("already patched")
old = """                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;"""
new = """                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return (type == GGML_TYPE_Q2_0 && table_id == MMVQ_PARAMETERS_GENERIC) ? 4 : 2;   // RAPIDLI-MMVQROWS"""
# (b12631: the line for one token also names RDNA1; the lines patched are the same)
first_old = "                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;"
first_new = "                return (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_RDNA1) && mmvq_two_rows_ncols1(type) ? 2 : 1;"
if s.count(old) != 1 and s.count(old.replace(first_old, first_new)) == 1:
    old, new = old.replace(first_old, first_new), new.replace(first_old, first_new)
assert s.count(old) == 1, "the place to patch was not found"
io.open(P, "w", encoding="utf8", newline="\n").write(s.replace(old, new))
print("4 rows a block for Q2_0 with 2 to 8 tokens")
