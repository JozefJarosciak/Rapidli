"""The attention cache a model needs per token (fit.kv_cell_mb), looped models included, and what the parked states and
the drafting context may take (fit.probe_budget). Needs nothing running.

    python tests/test_fit_budget.py"""
import os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import fit     # noqa: E402
import models  # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:240] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


HEADERS = {
    # Inkling-2-Mini Q4_K_M: 48 blocks, attention in all of them
    "unrolled": {"general.architecture": "inkling2", "inkling2.block_count": 48, "inkling2.attention.head_count": [24] * 48,
                 "inkling2.attention.head_count_kv": [8] * 48, "inkling2.embedding_length": 2304,
                 "inkling2.attention.key_length": 128, "inkling2.attention.value_length": 128},
    # Inkling-2-Mini Q2_K_XL: 24 blocks stored, blocks 6..18 run 3 times
    "looped": {"general.architecture": "inkling2_loop", "inkling2_loop.block_count": 24, "inkling2_loop.attention.head_count": [24] * 24,
               "inkling2_loop.attention.head_count_kv": [8] * 24, "inkling2_loop.embedding_length": 2304,
               "inkling2_loop.attention.key_length": 128, "inkling2_loop.attention.value_length": 128,
               "inkling2_loop.loop.count": 3, "inkling2_loop.loop.start": 6, "inkling2_loop.loop.end": 18},
    # a hybrid with attention in every fourth of 40 layers and 2 KV heads (a Qwen3.8-35B-like model)
    "hybrid": {"general.architecture": "qwen35moe", "qwen35moe.block_count": 40, "qwen35moe.attention.head_count": 16,
               "qwen35moe.attention.head_count_kv": 2, "qwen35moe.embedding_length": 2048, "qwen35moe.attention.key_length": 256,
               "qwen35moe.attention.value_length": 256, "qwen35moe.full_attention_interval": 4},
}
tmp = tempfile.mkdtemp()
paths = {}
for name in HEADERS:
    paths[name] = os.path.join(tmp, name + ".gguf")
    with open(paths[name], "wb") as f:
        f.write(b"\0" * 1024)
models.header = lambda path, want_template=False: HEADERS[os.path.basename(path)[:-5]]

cell = {k: fit.kv_cell_mb(p) for k, p in paths.items()}
check("an unrolled 48-layer model: 48 layers of cache", abs(cell["unrolled"] - 48 * 8 * 256 * 2 / 2**20) < 1e-9, cell)
check("a looped model counts each run of its looped blocks (24 stored, 48 run)", abs(cell["looped"] - cell["unrolled"]) < 1e-9, cell)
check("a hybrid counts only its attention layers", abs(cell["hybrid"] - 10 * 2 * 512 * 2 / 2**20) < 1e-9, cell)

fit.card = lambda: (0, 10**6)                 # a card with room for anything: only what the states may cost decides
common = lambda: dict(vault=80, ctx_extra=8192, spec_ctx=8192, lookup=8)
big = fit.probe_budget(paths["looped"], common(), 1000)
check("a large attention cache gets fewer parked states, whatever the card has free", 8 <= big.get("vault", 80) <= 12, big)
check("... and a 4k drafting context", big.get("spec_ctx") == 4096, big)
check("... the same for the looped and the unrolled file", fit.probe_budget(paths["unrolled"], common(), 1000) == big)
small = fit.probe_budget(paths["hybrid"], common(), 1000)
check("a small attention cache keeps its 80 parked states and 8k drafting context", small == {}, small)
c = dict(vault=80, ctx_extra=8192, spec_ctx=8192)   # no drafting
check("without drafting, the drafting context is left alone", "spec_ctx" not in fit.probe_budget(paths["looped"], c, 1000))

fit.card = lambda: (20000, 24564)             # a nearly full card: the probe itself must give up more
tight = fit.probe_budget(paths["looped"], common(), 1000)
check("on a nearly full card the parked states go down to 8", tight.get("vault") == 8, tight)
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
