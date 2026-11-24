"""Patch for ggml-cuda (llama-src/): the kv_rows flash attention takes a mask with a head dimension.

The MMA kernel already reads the mask of each head (nb32*(zt_Q % ne32)) and of each slice (nb33*(sequence % ne33)), and
K/V through kv_rows; only the kernel choice refused the combination ("the kv_rows mask is compacted per sequence, it has
no head dimension"). Inkling 2 needs it: its relative-position bias goes into a per-head mask, and with kv_rows that mask
covers only each sequence's own cells instead of the whole unified cache (engine/llama/patch_zzzrows.py). A mask with
a head dimension turns the GQA grouping off (ncols2 = 1), as it does without kv_rows. Nothing changes for a mask
without a head dimension, so for every other model.
  python patch_zrowsmask.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.environ.get("LLAMA_SRC") or os.path.join(HERE, "..", "llama-src"), "ggml", "src", "ggml-cuda", "fattn.cu")
NL = chr(10)
s = open(P, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI: kv_rows with a per-head mask" in s:
    raise SystemExit("already patched")
a = """    if (mask && mask->ne[2] != 1) {
        // the kv_rows mask is compacted per sequence, it has no head dimension
        if (kv_rows) {
            return BEST_FATTN_KERNEL_NONE;
        }
"""
b = """    if (mask && mask->ne[2] != 1) {
        // RAPIDLI: kv_rows with a per-head mask (Inkling 2's relative-position bias over each sequence's own cells): the
        // MMA kernel offsets the mask by head and by slice; GQA grouping is off for a mask with a head dimension
"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
open(P, "w", encoding="utf8", newline=NL).write(s)
print("patched", P)
