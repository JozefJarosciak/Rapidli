"""Starts the engine on any downloaded model with as many lanes as the card can hold next to it.

  python app/fit.py <model path> [ctx_extra=8192] [margin=700]

1. a probe start with 8 lanes: the engine's log says what one lane costs on this model (recurrent state per sequence,
   attention cache per cell) and nvidia-smi what the card has left (other programs use part of it);
2. lanes = 8 + free memory / cost of a lane (at most 256 sequences in all);
3. the engine is started with that many lanes and a short pass is timed: if narrow passes got much slower than in the
   probe, part of the memory went to the PC (WDDM spill) and the lanes are reduced by 15% until it is fast again.
The result is kept in data/fit.json per model file; a later start of the same model begins from it (and is checked)."""
import json, os, re, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eng

FIT = os.path.join(os.path.dirname(HERE), "data", "fit.json")
MAX_SEQ = 256
CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144]
AUTO_MIN_LANES = 48        # "auto" context: the largest that still leaves this many lanes
SPEC_SPILL_S = 3.0         # the drafting context reads ~800 tokens in 0.3 to 0.6 s; when it spilled, in 10 s and more
# Free graphics memory a started engine must leave, as a share of the margin: below it the card is overfull, Windows
# pages the engine's memory in and out of the PC's memory, and a pass can take minutes (18 Nov: 0 passes in 9 min).
# Checked right after a start, before the workout puts load on the card.
FREE_SHARE = 0.5
WORKOUT_SPILL_S = 20       # seconds: a first pass with all lanes slower than this has spilled (it takes ~1 s when it fits)
LANE_EXTRA = 1.25          # what a lane really takes, against its states alone (see lanes_for in fit)
VAULT_MB = 640             # the most the parked states may take (80 on a 35B; 10 on Inkling-2-Mini, see probe_budget)
SPEC_CTX_MB = 1024         # the most an 8k drafting context may take; above it, 4k
CTX = 300


import plat
NOWIN = plat.NOWIN
SPILL_MB = 300             # the engine's memory in the PC's memory grew by more than this since the fit: it spilled


def card():
    """(used MiB, total MiB) of the graphics card (the first one)."""
    out = plat.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).strip().splitlines()
    if not out:
        raise RuntimeError("nvidia-smi did not answer: Rapidli needs an NVIDIA graphics card and its driver")
    try:
        a, b = [int(float(x)) for x in out[0].split(",")[:2]]
    except ValueError:
        raise RuntimeError("nvidia-smi gave no memory figures: " + out[0][:120])
    return a, b


def spilled_mb(pid):
    """MB of a program's graphics memory that Windows keeps in the PC's memory instead of on the card ("shared GPU
    memory"). An engine that fits has some there too, more with more lanes (384 MB with 63 lanes and 6 GB of the card
    free, 19 Nov), so the number alone says nothing: the fit notes it, and growth beyond that means Windows moved engine
    memory off an overfull card, where every pass that touches it crawls. 0 when it cannot be read, and on Linux (where
    an overfull card fails an allocation instead)."""
    if os.name != "nt" or not pid:
        return 0
    try:
        out = subprocess.run(["typeperf", rf"\GPU Process Memory(pid_{int(pid)}_*)\Shared Usage", "-sc", "1"], capture_output=True,
                             text=True, timeout=20, creationflags=NOWIN).stdout
        rows = [l for l in out.splitlines() if l.startswith('"') and not l.startswith('"(PDH')]
        vals = [float(v) for v in rows[0].strip('"').split('","')[1:] if v.strip()]
        return round(sum(vals) / 2**20)
    except Exception:
        return 0


def holders(skip_pid=None, least_mb=500):
    """Other programs that hold graphics memory, [(name, MB)] largest first (another llama-server, a game, ...). Windows'
    desktop manager is left out: its counter adds up the desktop's shared surfaces, tens of GB that are not on the
    card. [] when it cannot be read."""
    return [(x["name"], x["mb"]) for x in plat.gpu_programs(least_mb) if x["pid"] != skip_pid and x["name"].lower() != "dwm"]


def log_numbers(port):
    """What the engine log says about its first (main) context: sequences, cells, KV and recurrent buffers.
    kv is the sum of all its KV caches: a model with sliding-window layers (Gemma 4) has two, a full-size one for the
    SWA layers too ("using full-size SWA cache"). Counting only the first made a cell look 3.5 times cheaper than it
    was (NVIDIA-Nemotron-3.5-Bolt-E4B, 18 Nov): 248 lanes and 128k of context filled the card and the passes stalled."""
    t = open(os.path.join(eng.OUT, f"engine-{port}.log"), encoding="utf8", errors="replace").read()
    starts = [m.start() for m in re.finditer(r"llama_context: n_ctx\s+=", t)]
    main = t[starts[0]:starts[1]] if len(starts) > 1 else t[starts[0]:] if starts else t    # the first context only
    seqs = [int(x) for x in re.findall(r"llama_context: n_seq_max\s+=\s+(\d+)", t)]
    cells = [int(x) for x in re.findall(r"llama_context: n_ctx\s+=\s+(\d+)", t)]
    kv = [float(x) for x in re.findall(r"CUDA0 KV buffer size =\s+([\d.]+) MiB", main)]
    rs = [float(x) for x in re.findall(r"CUDA0 RS buffer size =\s+([\d.]+) MiB", main)]
    model = [float(x) for x in re.findall(r"CUDA0 model buffer size =\s+([\d.]+) MiB", t)]
    spec_off = "no multi-token-prediction head" in t
    return {"seqs": seqs[0] if seqs else 0, "cells": cells[0] if cells else 0, "kv": sum(kv),
            "rs": sum(rs), "model": sum(model), "spec_off": spec_off}


def stats(port):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10).read())


def pass_ms(port, width=8, tokens=60, limit=300):
    """Mean time of the engine's passes while `width` lanes write (from its own counters, so lanes that end early do
    not distort it)."""
    names = ["Anna", "Boris", "Clara", "David", "Elena", "Felix", "Greta", "Hugo", "Irene", "Jonas", "Karla", "Leon", "Marta", "Nils", "Olga", "Pavel"]
    body = {"prompt": "Here is a long, detailed list of facts about cities, one per line, never ending:\n",
            "branches": [f"{i + 1}. The city of {names[i % 16]}" for i in range(width)], "n_predict": tokens, "temperature": 0}
    a = stats(port)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    # (a check that takes too long gives the passes done so far, and the load goes on: on a PC short of memory a mixture
    # of experts read most of its experts from the disk, and a check that timed out failed the whole load, 21 Nov;
    # closing the stream frees the lanes)
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=limit) as resp:
            for _ in resp:
                if time.time() - t0 > limit:
                    break
    except urllib.error.HTTPError:
        raise                       # (the engine refused it: an error, not a slow check)
    except (TimeoutError, OSError):
        pass
    b = stats(port)
    n = b["passes"] - a["passes"]
    if n <= 0:
        return max(1000.0 * limit / max(1, tokens), (time.time() - t0) * 1000.0)
    ms = b["mean_pass_ms"] * b["passes"] - a["mean_pass_ms"] * a["passes"]
    return ms / max(1, n)


def responds(port, timeout=60):
    """-> True when the engine writes a few tokens on one lane within `timeout` seconds. On a card that other programs
    have filled, the engine can start and then never finish a pass (19 Nov: HauhauCS IQ4_XS, 15.5 GB, next to 9.1 GB
    of other programs; the probe waited over ten minutes)."""
    body = {"prompt": "Hello", "branches": [","], "n_predict": 4, "temperature": 0}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for _ in resp:
                pass
        return True
    except Exception:
        return False


def pass_ms_warm(port, width=8, tokens=60, good=None):
    """pass_ms, the best of up to three measurements (it stops at the first that is `good` or better). The first
    measurement after a start with many lanes runs up to twice as long as those after it (19 Nov: 31.5 ms, then
    17.5 ms, on a card with 5 GB free), and was taken for memory that spilled: 96 lanes were cut to 68."""
    best = 1e9
    for _ in range(3):
        best = min(best, pass_ms(port, width, tokens))
        if good is None or best <= good:
            break
    return best


def all_ms(port, lanes, tokens=16, base=None):
    """Time of a pass while every lane writes (the better of two measurements). This is what shows memory that
    spilled to the PC's: with 137 lanes on an overfull card a pass with 8 lanes took 30 ms against 17.5 (no more
    than a first measurement does), with 32 lanes 252 ms against 25 (19 Nov). base: the 8-lane pass time; a first,
    short pass more than 4 times slower than all lanes should take has spilled, and is not measured again (78 lanes
    on a spilled card took 1,385 ms a pass and the check four minutes, 19 Nov)."""
    first = pass_ms(port, lanes, 4)
    if base is not None and first > 4 * (base + 0.5 * lanes):
        return first
    return min(first, pass_ms(port, lanes, tokens), pass_ms(port, lanes, tokens))


def all_ok(lanes, ms_all, ms_8):
    """Whether all lanes together write clearly faster than 8 do (64 lanes write 4 times as much in a second as 8
    when their memory is on the card; spilled, 32 lanes wrote half of what 8 did)."""
    return lanes < 24 or lanes / max(0.1, ms_all) >= 1.5 * 8 / max(0.1, ms_8)


def margin_with(margin, others):
    """The graphics memory left free: the margin set, and more next to other programs that hold much of the card,
    because what they hold changes (lanes fitted to 1.3 GB free next to 8 GB of other programs had spilled half an
    hour later, when one of them had grown by 0.3 GB, 19 Nov)."""
    held = sum(mb for _, mb in others or [])
    return int(margin + (0.15 * held if held >= 2000 else 0))


def spec_s(port):
    """Seconds the drafting context (/spec) takes to read about 800 tokens; None without drafting. Its buffers are
    allocated after the lanes', so it is the first to spill into PC memory (then it reads 20 times slower)."""
    st = stats(port)
    if not st.get("spec"):
        return None
    text = " ".join(f"Line {i}: the river passes the old mill and turns east toward the bridge." for i in range(55))
    body = {"prompt": "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n", "n_predict": 1}
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/spec", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        for _ in resp:
            pass
    return time.time() - t0


def workout(port, lanes, limit=300, items=110):
    """Makes the engine allocate what it allocates only when first needed (all lanes in one pass, a long prompt read in
    bulk), so that a check after it sees the memory as it will be in use. -> seconds it took (None: not done in `limit`)"""
    t0 = time.time()
    try:
        text = " ".join(f"Item {i}: a short line about weather, rivers and bridges in a small town." for i in range(items))
        r = urllib.request.Request(f"http://127.0.0.1:{port}/fork", data=json.dumps({"prompt": text, "branches": [f" {i}." for i in range(lanes)],
                                   "n_predict": 6, "temperature": 0}).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(r, timeout=limit) as resp:
            for _ in resp:
                if time.time() - t0 > limit:      # (in all, not per line: a crawling stream sends a line now and then)
                    return None
    except Exception:
        return None
    return time.time() - t0


# drafts copied from the text (engine --lookup): up to this many tokens, checked in one pass. 0: off. A model without an
# MTP head then gets the drafting route (/spec) too; with one, both drafters, whichever is expected to give more.
# Measured 20 Nov on an RTX 4090 (one answer of 320 tokens): Ternary-Bonsai-2-27B (no MTP head) against a lane's
# 82 tok/s: a rewrite 178, code 196, a summary of a source 95, prose 68 (the app sends prose to a lane, Fan.drafting);
# Qwen3.8-27B Q3_K_P against MTP alone (installed engine): prose 79 -> 83, a summary 113 -> 124, a rewrite 130 -> 134,
# code 133 -> 141. It costs the card a drafting context (Bonsai, which had none: ~1.3 GB) or longer rollback (~650 MB).
LOOKUP = int(os.environ.get("RAPIDLI_LOOKUP", "8"))


def load_fits():
    try:
        return json.load(open(FIT, encoding="utf8"))
    except Exception:
        return {}


def fit_key(path, mtp=False, ctx_extra=8192, mmproj=None, margin=1200, lanes_max=None):
    """The key under which a fit of these settings is kept in data/fit.json (one with a lane maximum from the API is
    kept apart: it must not cap the model's own fit; so is one with a smaller projector than the usual 16-bit one)."""
    pn = os.path.basename(mmproj or "").lower()
    vis = "" if not mmproj else "|vision" + (":" + pn if "mmproj" in pn and "f16" not in pn and pn != "mmproj.gguf" else "")
    return (f"{path}|{os.path.getsize(path)}|{ctx_extra}|{bool(mtp)}" + vis + (f"|margin{margin}" if margin != 1200 else "")
            + (f"|lookup{LOOKUP}" if LOOKUP else "") + (f"|max{int(lanes_max)}" if lanes_max else ""))


def reuses(key, fresh=False, lanes_fixed=None):
    """Whether fit() will start from the kept fit (one engine start and a check) rather than fit again (a probe and a start)."""
    return bool((load_fits().get(key) or {}).get("pass_ms_8")) and not fresh and not lanes_fixed


# a mixture of experts kept in the PC's memory. Measured 20 Nov on an RTX 4090 with an i9-14900KF and 64 GB of RAM
# (about 20-30 GB of it free), Qwen3.8-Flash-Next Q2_0 (31.6 GB of experts): one answer on /spec, warm, 45.3 tok/s
# with none of the misses on the processor against 32.4 with all of them; 53.6 with a 9 GB cache for /spec against
# 45.3 with 6 GB; 7 threads on the 8 P-cores against 16 anywhere: 45.3 against 31.4. llama.cpp's own server, its
# experts in memory with a 6 GB cache and 7 threads: 45.0 (the same question asked twice).
# With the experts' products of 9-64 tokens on the vector kernel in slices (engine/cuda/patch_zmmidslice.py, 20 Nov
# night) the lanes went from 130 to 380-437 tok/s at 32 lanes, and one lane writes 51-56 tok/s on the lanes' own cache
# (as fast as /spec with a cache of its own did): one cache for the lanes, no drafting context (copying gains little
# when every extra row needs experts of its own), /spec only for a model with an MTP head.
MOE_LANES = int(os.environ.get("RAPIDLI_MOE_LANES", "32"))          # lanes of a mixture of experts kept in the PC's memory
MOE_SPEC_SHARE = float(os.environ.get("RAPIDLI_MOE_SPEC_SHARE", "0.5"))  # share of the expert cache for /spec (an MTP head only)
# share of the cache's misses computed by the processor. With the sliced products, 32 lanes: 380 tok/s with none, 436
# with a quarter; one lane: 51-56 with none, 45 with a quarter. None: one answer is what is waited for most
MOE_CPU_SHARE = float(os.environ.get("RAPIDLI_MOE_CPU_SHARE", "0"))
MOE_SPEC_CPU_SHARE = float(os.environ.get("RAPIDLI_MOE_SPEC_CPU_SHARE", "0"))
MOE_CHUNK = int(os.environ.get("RAPIDLI_MOE_CHUNK", "1024"))         # prompt tokens a pass: 411 tok/s at 1,024, 196 at 320


def moe_split(path, margin=1200):
    """{"experts", "table", "rest"} in bytes when `path` is a mixture of experts that does not fit the card whole (its
    experts then stay in the PC's memory), else None."""
    import models
    sp = models.split_bytes(path)
    if not sp or not sp["experts"]:
        return None
    used, total = card()
    whole_mb = (sp["experts"] + sp["rest"]) / 2**20       # (an n-gram table always stays in the PC's memory, see below)
    return sp if whole_mb + margin + 2048 > total else None


MOE_SPARE_MB = int(os.environ.get("RAPIDLI_MOE_SPARE_MB", "6144"))   # the PC's memory left for everything else after a read-ahead


def prefetch_experts(path, sp, say, limit=300):
    """Reads the routed experts of `path` from the disk in order (16 MB at a time, nothing kept) so that Windows has them
    in its cache before the engine maps them. Skipped when the PC's memory has no room for them. -> seconds, or None"""
    import models
    import plat
    r = plat.ram()
    need_mb = sp["experts"] / 2**20
    # (the model's own pages in Windows' cache count as free memory, and reading them again costs little: a reload with
    # 26 GB free for 32 GB of experts, most of them cached, skipped the read-ahead it did not need to, 21 Nov)
    if not r or r["free_mb"] < need_mb * 0.7:
        say(f"the experts are not read ahead: the PC's memory has {r['free_mb'] / 1024:.0f} GB free for {need_mb / 1024:.0f} GB" if r else
            "the experts are not read ahead: the PC's memory could not be read")
        return None
    # (any PC: the experts first - every token uses them - and the n-gram table too when the memory holds both with room
    # to spare, so that a PC with more memory starts with the whole model warm; decided before reading, as Windows counts
    # what it has cached as free and the table would push out the experts just read. 64 GB: the experts; 128 GB: both)
    table_mb = sp.get("table", 0) / 2**20
    kinds = ("experts", "table") if table_mb and r["free_mb"] >= need_mb + table_mb + MOE_SPARE_MB else ("experts",)
    ranges = models.expert_ranges(path, kinds)
    if not ranges:
        return None
    t0, done, total, last = time.time(), 0, sum(n for _, _, n in ranges), 0.0
    say(f"reading the {'experts and the table' if 'table' in kinds else 'experts'} ahead ({total / 2**30:.1f} GB)")
    buf = bytearray(16 * 2**20)
    try:
        for p, a, n in ranges:
            with open(p, "rb", buffering=0) as f:
                f.seek(a)
                left = n
                while left > 0:
                    k = f.readinto(memoryview(buf)[:min(len(buf), left)])
                    if not k:
                        break
                    left -= k
                    done += k
                    if time.time() - last > 2:
                        last = time.time()
                        el = last - t0
                        say(f"reading the experts ahead: {done / 2**30:.1f} of {total / 2**30:.1f} GB, {done / 2**20 / max(0.1, el):.0f} MB/s")
                    if time.time() - t0 > limit:
                        say(f"reading the experts ahead stopped after {limit} s ({done / 2**30:.1f} GB)")
                        return None
    except OSError as e:
        say(f"reading the experts ahead failed: {str(e)[:80]}")
        return None
    el = time.time() - t0
    say(f"the experts read ahead: {total / 2**30:.1f} GB in {el:.0f} s")
    return el


# items of the workout's prompt for a model with an n-gram table read from the disk (each token reads rows of it:
# a cold table made the 110-item prompt, ~2,000 tokens on every lane, take minutes)
MOE_WORKOUT_ITEMS = int(os.environ.get("RAPIDLI_MOE_WORKOUT_ITEMS", "28"))
MOE_WORKOUT_S = int(os.environ.get("RAPIDLI_MOE_WORKOUT_S", "180"))   # the workout after a start (it only makes the card allocate)
MOE_CHECK_S = int(os.environ.get("RAPIDLI_MOE_CHECK_S", "120"))       # each speed check, at most
MOE_SHORT_LANES = int(os.environ.get("RAPIDLI_MOE_SHORT_LANES", "16"))  # lanes when the PC's memory is short for the experts


def fit_moe(path, sp, common, margin, port, say, note, lanes_fixed, key):
    """fit() for a mixture of experts larger than the card. Everything every token needs (attention, the recurrent
    mixers, routers, shared experts, the output) stays on the card with the lanes; the routed experts stay in the PC's
    memory, read from the file as it is mapped (nothing is copied, Windows keeps what is used). The rest of the card
    becomes a cache of the experts used most, filled and changed while the model writes (llama.cpp's MoE cache: recency
    and frequency of use); MOE_CPU_SHARE of the experts it misses are computed by the processor where they are,
    the others copied to the card. An n-gram table (Qwen3.8-Flash-Next's 28 GB) stays in the PC's memory as well: each
    token reads a few rows of it, which Windows keeps in its file cache.
    The processor's threads are kept on the fast cores (plat.fast_cores), one core left to the thread that drives the
    card. The drafted single answer (/spec) gets a cache of its own, MOE_SPEC_SHARE of the room: it writes one token
    at a time, where every expert found on the card counts; the lanes' passes use most experts of every layer at once."""
    import plat
    cores = plat.fast_cores()
    threads = int(os.environ.get("RAPIDLI_MOE_THREADS", "0")) or max(1, len(cores) - 1)
    # (a 16-bit recurrent state: the 8-bit one is not built for qwen4exp's layers, its graph stops on a reshape)
    # (longer prompt passes: each expert copied to the card serves more tokens)
    base = dict(common, experts_ram=1, threads=threads, cores=cores, moe_cpu_share=MOE_CPU_SHARE, moe_spec_cpu_share=MOE_SPEC_CPU_SHARE,
                state=16, chunk=MOE_CHUNK)
    # (few parked states: each costs the card about what a lane does - 71 MB for Qwen3.8-Flash-Next - and the card's room
    # is the expert cache. With 80 of them (the many-lane code pages' default since 22 Nov) the cache fell from 14 GB to
    # 3.3 GB and 32 lanes from ~400 tok/s to a 46-60 peak; RAPIDLI_VAULT_MOE)
    base["vault"] = min(int(common.get("vault", 8)), int(os.environ.get("RAPIDLI_VAULT_MOE", "8")))
    base.pop("lookup", None)                  # (copying: see MOE_LANES; /spec stays for an MTP head)
    if not base.get("spec"):
        for k in ("spec_ctx", "spec_batch"):
            base.pop(k, None)
    if base.get("spec"):
        # /spec reads its prompt in passes of spec_batch tokens: 64 made a question with its offline articles wait
        # 45-69 s for its first words (every pass copied most experts to the card, 20 Nov), so as long as the lanes'
        base["spec_batch"] = MOE_CHUNK
    G = 2**30
    say(f"a mixture of experts larger than the card: {sp['rest'] / G:.1f} GB on the card, {sp['experts'] / G:.1f} GB of experts in the PC's memory"
        + (f", a {sp['table'] / G:.1f} GB table read from the disk" if sp["table"] else "") + f"; {threads} processor threads on {len(cores)} fast cores")
    # 0. the experts read ahead, in order: after Windows had let go of the model file (other programs, another model),
    # the engine read its 32 GB of experts a page at a time as tokens needed them, at about 40 MB/s, and a load took
    # nine minutes and its first answers stalled (21 Nov). Read in order they come at the disk's full speed and stay in
    # Windows' cache, where the engine's mapping finds them. Only when the PC's memory has room for them; a file that is
    # in the cache already is read at memory speed
    if os.environ.get("RAPIDLI_MOE_PREFETCH", "1") == "1":
        prefetch_experts(path, sp, say)
    # 1. the probe: 8 lanes and no cache, what the card holds without the experts
    say("probe: 8 lanes, no expert cache")
    t_probe = time.time()
    eng.start(lanes=8, moe_cache_mib=0, moe_spec_mib=0, wait=600, **base)
    nums = log_numbers(port)
    if not responds(port, timeout=300):
        eng.stop(port)
        raise RuntimeError("the model started but could not write next to what other programs hold on the card")
    workout(port, 8, limit=MOE_WORKOUT_S, items=MOE_WORKOUT_ITEMS if sp.get("table") else 110)
    used, total = card()
    per_lane = (nums["rs"] / max(1, nums["seqs"]) + nums["kv"] / max(1, nums["cells"]) * CTX) * 1.05
    others = holders(eng.pid(port))
    # (more room left on the card than for other models: the expert cache fills it, and when a browser or the desktop
    # briefly needed more of the card Windows moved part of the cache to the PC's memory - one lane went from 40 ms a
    # pass to 1,240 for minutes, Qwen3.8-Flash, 22 Nov; a 12 GB cache instead of 14 costs little, 397 tok/s against 403;
    # RAPIDLI_MOE_MARGIN, MB)
    keep = margin_with(margin, others) + int(os.environ.get("RAPIDLI_MOE_MARGIN", "1536"))
    free = total - used - keep
    # the cache is used for passes of up to LLAMA_MOE_CACHE_MAX_BATCH tokens (llama.cpp: 32; patch_zzmoelog.py makes it
    # a setting, set to the lanes below) whose experts all get a slot (the layers of one layout share one pool of slots,
    # filled by recency and frequency of use). Wider passes go the slow way, every expert they use copied to the card.
    import models
    hd = models.header(path, want_template=False)
    arch = str(hd.get("general.architecture", ""))
    n_layer, n_exp, n_used = hd.get(f"{arch}.block_count") or 48, hd.get(f"{arch}.expert_count") or 1, hd.get(f"{arch}.expert_used_count") or 8
    expert_mib = sp["experts"] / 2**20 / max(1, n_layer * n_exp)      # one expert of one layer (gate, up and down)
    room = free - 400                                               # (the caches' slot maps and buffers)
    spec_on = bool(base.get("spec"))
    want = int(lanes_fixed) if lanes_fixed else MOE_LANES
    lanes = 8
    for n in range(8, max(8, min(want, 64)) + 1):                   # the most lanes whose passes still use the cache
        rest = room - (n - 8) * per_lane * LANE_EXTRA
        spec_mib = int(rest * MOE_SPEC_SHARE) if spec_on else 0
        slots = (rest - spec_mib) / max(0.01, expert_mib)
        if slots >= max(n * n_used, 16 * n_used) and rest - spec_mib >= 0.25 * room:
            lanes = n
    if lanes_fixed:
        lanes = max(8, int(lanes_fixed))
    rest = room - (lanes - 8) * per_lane * LANE_EXTRA
    spec_mib = int(rest * MOE_SPEC_SHARE) if spec_on else 0
    cache_mib = max(0, int(rest) - spec_mib)
    say(f"an expert is {expert_mib:.2f} MB ({n_exp} a layer, {n_used} used a token): {cache_mib / max(0.01, expert_mib):.0f} cache slots for the lanes"
        + (f", {spec_mib / max(0.01, expert_mib):.0f} for the drafted answer" if spec_mib else ""))
    # the experts the card does not keep are read where they are mapped: from the PC's memory while Windows keeps them
    # there, from the disk when other programs leave too little of it (slower, but it runs)
    # (the PC's memory measured with the probe stopped: the experts it had read stay in Windows' cache and count as
    # free; measured beside it, 8 GB were "free" where 35 GB were, 21 Nov)
    eng.stop(port)
    r = plat.ram()
    if r:
        need = sp["experts"] / 2**20 - cache_mib - spec_mib
        # short of memory: fewer lanes (a pass of many lanes uses most experts of every layer, and those not in memory
        # are read from the disk; 32 lanes on a PC with 8 GB free did not finish one pass in ten minutes, 21 Nov)
        if need > r["free_mb"] * 1.25 and not lanes_fixed and lanes > MOE_SHORT_LANES:
            lanes = MOE_SHORT_LANES
            rest = room - (lanes - 8) * per_lane * LANE_EXTRA
            spec_mib = int(rest * MOE_SPEC_SHARE) if spec_on else 0
            cache_mib = max(0, int(rest) - spec_mib)
            need = sp["experts"] / 2**20 - cache_mib - spec_mib
            say(f"the PC's memory is short for the experts: {lanes} lanes, a larger cache on the card")
        if need > r["free_mb"]:
            say(f"the PC's memory has {r['free_mb'] / 1024:.0f} GB free for {need / 1024:.0f} GB of experts the card does not keep: "
                "the rest are read from the disk when they are needed (slower until other programs leave more room)")
    say(f"a lane costs {per_lane:.0f} MB; {total - used:.0f} MB free next to the model -> {lanes} lanes, an expert cache of "
        f"{cache_mib / 1024:.1f} GB for the lanes" + (f" and {spec_mib / 1024:.1f} GB for the drafted answer" if spec_mib else ""))
    base["LLAMA_MOE_CACHE_MAX_BATCH"] = max(32, lanes)
    # 2. the real start; a card left short is given a smaller cache
    for attempt in range(4):
        if attempt:
            note("restart", time.time() - t_probe)
        t_up = eng.start(lanes=lanes, moe_cache_mib=cache_mib, moe_spec_mib=spec_mib, wait=600, **base)
        w = workout(port, lanes, limit=MOE_WORKOUT_S, items=MOE_WORKOUT_ITEMS if sp.get("table") else 110)
        # (where a long load goes: the start, then the first pass of all lanes over a long prompt; 21 Nov a load took
        # six minutes between the fit and the speed checks and nothing said which part)
        say(f"engine started in {t_up:.0f} s; its first pass of all {lanes} lanes " + (f"took {w:.0f} s" if w is not None else f"was cut off after {MOE_WORKOUT_S} s"))
        used, total = card()
        short = margin * FREE_SHARE - (total - used)
        # (not the engine's memory Windows keeps in the PC's memory: for a MoE engine it is its host buffers, ~860 MB with 32
        # lanes, 500 MB more than the 8-lane probe whatever the cache's size - shrinking the cache for it cut 13.4 GB to
        # 8.9 GB and changed nothing, 21 Nov)
        if short > 0 and attempt < 3 and cache_mib + spec_mib > 512:
            cut = short + margin * (1 - FREE_SHARE)
            f = max(0.0, 1 - cut / max(1, cache_mib + spec_mib))
            cache_mib, spec_mib = int(cache_mib * f), int(spec_mib * f)
            say(f"only {total - used} MB of the card is free: a cache of {cache_mib / 1024:.1f} GB" + (f" and {spec_mib / 1024:.1f} GB" if spec_mib else ""))
            continue
        break
    ms = pass_ms(port, 8, 40, limit=MOE_CHECK_S)
    ma = pass_ms(port, lanes, 12, limit=MOE_CHECK_S)
    # (the drafting check too only when the lanes' checks were not cut short: it waited up to 300 s on a PC short of
    # memory and failed the load, a review found, 21 Nov)
    ss = None
    if base.get("spec") and max(ms, ma) < 10000:
        try:
            ss = spec_s(port)
        except (TimeoutError, OSError):
            ss = None
    say(f"pass with 8 lanes: {ms:.0f} ms; with all {lanes}: {ma:.0f} ms, {lanes * 1000 / max(0.1, ma):.0f} tokens a second together")
    st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5).read())
    out = {"lanes": lanes, "per_lane_mb": round(per_lane, 1), "per_cell_mb": round(nums["kv"] / max(1, nums["cells"]) * 1.05, 4),
           "ctx_extra": common["ctx_extra"], "options": [], "model_mb": round(nums["model"]), "pass_ms_8": round(ms, 1), "pass_ms_all": round(ma, 1),
           "spec_s": round(ss, 2) if ss else None, "spec": st.get("spec", 0), "spec_mtp": st.get("spec_mtp", st.get("spec", 0)), "spec_lookup": st.get("spec_lookup", 0), "at": time.strftime("%Y-%m-%d %H:%M"), "used_by_others_mb": None,
           "shared_mb": spilled_mb(eng.pid(port)), "free_mb": total - used, "others": others, "vision": bool(st.get("vision")),
           "moe": {"cache_mib": cache_mib, "spec_mib": spec_mib, "cpu_share": MOE_CPU_SHARE, "threads": threads, "cores": cores,
                   "experts_gb": round(sp["experts"] / G, 1), "table_gb": round(sp["table"] / G, 1), "card_gb": round(sp["rest"] / G, 1)}}
    fits = load_fits()
    # (a fit whose speed checks were cut short is not kept: its "passes" of 120 s were saved, and every later start and
    # the watchdog compared against them, 21 Nov; the next load measures again)
    if not lanes_fixed and max(ms, ma) < 10000:
        fits[key] = out
    elif not lanes_fixed:
        fits.pop(key, None)
        say("the speed could not be measured in time (the PC short of memory?): it is measured again at the next load")
    os.makedirs(os.path.dirname(FIT), exist_ok=True)
    json.dump(fits, open(FIT, "w", encoding="utf8"), indent=1)
    return out


def kv_cell_mb(path):
    """MiB of attention cache per token of context, from the model's own header (f16 keys and values in every layer
    that has attention; in a hybrid with full_attention_interval, only those layers). None when the header says too
    little."""
    try:
        import models
        m = models.header(path, want_template=False)
    except Exception:
        return None
    arch = str(m.get("general.architecture", ""))
    a = lambda k, d=None: m.get(f"{arch}.{k}", d)
    n_layer = int(a("block_count", 0) or 0)
    heads = a("attention.head_count", 0)
    heads = max(heads) if isinstance(heads, list) and heads else int(heads or 0)
    kv = a("attention.head_count_kv", heads)
    kv = max(kv) if isinstance(kv, list) and kv else int(kv or heads or 0)
    embd = int(a("embedding_length", 0) or 0)
    k_len = int(a("attention.key_length", 0) or (embd // heads if heads else 0))
    v_len = int(a("attention.value_length", 0) or k_len)
    every = int(a("full_attention_interval", 0) or 0)
    # a looped model stores some blocks once and runs them loop.count times: each run keeps its own cache (Inkling-2-Mini
    # Q2_K_XL, inkling2_loop: 24 blocks stored, 6..18 run 3 times = 48 layers with attention, 23 Nov; counted as 24, its
    # cache was taken for half its size and 80 parked states filled the card)
    loops, l0, l1 = int(a("loop.count", 0) or 0), int(a("loop.start", 0) or 0), int(a("loop.end", 0) or 0)
    if loops > 1 and 0 <= l0 < l1 <= n_layer:
        n_layer += (l1 - l0) * (loops - 1)
    layers = n_layer // every if every > 1 else n_layer
    if not (layers and kv and k_len):
        return None
    return layers * kv * (k_len + v_len) * 2 / 2**20


def probe_budget(path, common, margin, mmproj=None, lanes=8):
    """What the 8-lane probe must give up to fit next to the model: {"vault", "spec_ctx", "ctx_extra"} with smaller
    values, in that order (the parked states first, down to 8; the drafting context down to 4k; the conversation
    context down to 8k), or {} (it fits, or the model's cache size is not known). Every cell of the probe costs
    attention cache in each layer that has attention: (8 lanes + the parked states) x 300 cells, the conversation
    context and the drafting context. On Qwen 3.8 that is small; Inkling-2-Mini has attention in all 48 layers, and
    its probe (88 x 300 + 8k cells, 6.5 GB) next to 15.5 GB of weights filled the card and crawled for 10 minutes
    (23 Nov)."""
    per_cell = kv_cell_mb(path)
    if not per_cell:
        return {}
    vault, ctx = int(common.get("vault") or 0), int(common["ctx_extra"])
    # (the drafting context is there for MTP drafting ("spec") and for drafting by copying ("lookup") alike)
    spec = int(common.get("spec_ctx") or 0) if (common.get("spec") or common.get("lookup")) else 0
    out = {}
    # Whatever the card has free, the parked states and the drafting context get no more than they cost on a model with a
    # small attention cache: on Qwen3.8-35B 80 parked states take ~0.5 GB, on Inkling-2-Mini (attention in all 48
    # layers, 59 MB a state) 4.7 GB, and its 8k drafting context 1.6 GB - the room of ~80 lanes, and every pass reads
    # attention across all those cells (Q2_K_XL, 23 Nov: 21 lanes, 73 ms a pass with 8 where the Q4_K_M took 37)
    if vault > 8 and vault * CTX * per_cell > VAULT_MB:
        vault = max(8, int(VAULT_MB // (CTX * per_cell)))
        out["vault"] = vault
    if spec > 4096 and spec * per_cell > SPEC_CTX_MB:
        spec = 4096
        out["spec_ctx"] = spec
    try:
        used, total = card()
    except Exception:
        return out
    model_mb = os.path.getsize(path) / 2**20 + (os.path.getsize(mmproj) / 2**20 if mmproj and os.path.exists(mmproj) else 0)
    # cells the attention cache may hold: what is left after the weights, the margin and ~1.5 GB of buffers
    cells = (total - used - margin - model_mb - 1536) / per_cell
    need = lambda v, s, c: (lanes + v) * CTX + s + c
    if need(vault, spec, ctx) <= cells:
        return out
    if vault > 8:
        vault = max(8, min(vault, int((cells - spec - ctx) // CTX) - lanes))
        out["vault"] = vault
    if need(vault, spec, ctx) > cells and spec > 4096:
        spec = 4096
        out["spec_ctx"] = spec
    if need(vault, spec, ctx) > cells and ctx > min(CONTEXTS):
        fitting = [c for c in CONTEXTS if need(vault, spec, c) <= cells]
        ctx = max(fitting) if fitting else min(CONTEXTS)
        out["ctx_extra"] = ctx
    return out


def fit(path, mtp=False, ctx_extra=8192, margin=1200, vault=8, port=11832, say=print, kv=None, fresh=False, ctx_max=262144, mmproj=None, lanes_fixed=None, note=None,
        lanes_max=None):
    """Starts the engine on `path` with as many lanes as fit (or lanes_fixed, when the user set a number; at most
    lanes_max, when the API's load asks for a maximum).
    mmproj: the vision projector to load with the model.
    vault: parked states the engine can hold at once (RAPIDLI_VAULT): many lanes that each go on past their own
    ~300 tokens keep their state parked rather than reading the prompt again (~6 MB a slot on a 35B).
    note(event, seconds): told when the start takes longer than planned: "refit" (the kept fit did not hold, fitting
    again) and "restart" (one more engine start, about `seconds` long). -> {"lanes", "spec", ...}"""
    note = note or (lambda *a: None)
    # the drafting context holds a follow-up with its conversation (drafted from the conversation as /spec's parked
    # prefix): as large as the conversation context, 8,192 at most. At 4,096 a follow-up after one long answer no longer
    # fitted and ran undrafted (~55 tok/s against ~80). 8,192 costs about 270 MB more than 4,096 (KV of 256+16 MB per 4k)
    spec = {"spec": int(os.environ.get("RAPIDLI_SPEC_N", "4")), "spec_ctx": 8192 if ctx_extra == "auto" else max(4096, min(8192, int(ctx_extra))),
            # tokens /spec reads per pass (engine --spec-batch, 19 Nov): larger reads cold prompts faster but takes GPU memory
            # (lanes) and holds the card longer per pass
            "spec_batch": int(os.environ.get("RAPIDLI_SPEC_BATCH", "64"))} if mtp else {}
    if LOOKUP:
        spec.setdefault("spec_ctx", 8192 if ctx_extra == "auto" else max(4096, min(8192, int(ctx_extra))))
        spec.setdefault("spec_batch", int(os.environ.get("RAPIDLI_SPEC_BATCH", "64")))
        spec["lookup"] = LOOKUP
    if mmproj:
        spec["mmproj"] = mmproj
    auto = ctx_extra == "auto"
    contexts = [c for c in CONTEXTS if c <= max(8192, int(ctx_max or 262144))]
    # (a long prompt is read in passes of `chunk` tokens: 320 for a Q2_0 model, two of its 160-column matrix tiles (see
    # engine/cuda/patch_mmq_wide.py): Ternary-Bonsai-2-27B read 3,440-3,470 tokens a second instead of 3,310-3,320, with
    # no more of the card taken, 20 Nov; other types have tiles of up to 128 columns, which 256 fills)
    chunk = 320 if "Q2_0" in os.path.basename(str(path)).upper() else 256
    # (not more for Inkling 2: 512 reads a 2,400-token prompt at 4,140 tok/s instead of 2,800, but its compute buffers
    # cost the 32k conversation context - 16k was fitted instead - and long answers were no faster, 24 Nov)
    # (80: a page of 64 functions keeps 64 lanes' states between their windows, ~0.5 GB on a 35B; RAPIDLI_VAULT)
    vault = int(os.environ.get("RAPIDLI_VAULT", 80))
    common = dict(ctx=CTX, state=8, vault=vault, chunk=chunk, port=port, model=path, ctx_extra=8192 if auto else int(ctx_extra),
                  **({"kv": kv} if kv else {}), **spec)
    fits = load_fits()
    key = fit_key(path, mtp, ctx_extra, mmproj, margin, lanes_max)
    cap = (lambda n: max(8, min(int(n), int(lanes_max)))) if lanes_max else (lambda n: n)
    # a mixture of experts larger than the card: its experts in the PC's memory (fitted every time it loads: what the
    # card and the PC's memory have free decides the cache)
    sp = moe_split(path, margin)
    if sp:
        if auto:
            common["ctx_extra"] = 16384
        return fit_moe(path, sp, common, margin, port, say, note, cap(lanes_fixed or MOE_LANES) if lanes_max else lanes_fixed, key)
    # 0. a model fitted before: start with its lanes and check that nothing spilled
    asked, asked_vault = dict(common), vault
    if reuses(key, fresh, lanes_fixed):
        f0 = fits[key]
        common["ctx_extra"] = int(f0.get("ctx_extra") or common["ctx_extra"])
        # (the parked states and the drafting context as the probe cut them: a kept fit of Inkling started with 80
        # parked states again and filled the card)
        common.update({k: v for k, v in probe_budget(path, common, margin, mmproj, lanes=f0["lanes"]).items() if k != "ctx_extra"})
        vault = int(common.get("vault") or vault)
        say(f"starting with {f0['lanes']} lanes and {common['ctx_extra'] // 1024}k of conversation context (fitted {f0.get('at', '')})")
        eng.start(lanes=f0["lanes"], **common)
        used, total = card()
        if total - used < margin * FREE_SHARE:      # overfull: no workout (its passes would crawl), fitted again
            hs = holders(eng.pid(port))
            say(f"only {total - used} MB of the card is free next to the engine (" + (", ".join(f"{n} holds {mb / 1024:.1f} GB" for n, mb in hs[:3])
                if hs else "other programs hold more of it now") + "): fitting again")
            note("refit")
        else:
            w = workout(port, f0["lanes"], limit=WORKOUT_SPILL_S)
            # (a first pass that crawls has spilled: the fit is redone, rather than waiting on it, see WORKOUT_SPILL_S)
            ms = pass_ms_warm(port, 8, good=f0["pass_ms_8"] * 1.6 + 4) if w is not None and w <= WORKOUT_SPILL_S else 1e9
            ma = all_ms(port, f0["lanes"], base=f0["pass_ms_8"]) if ms < 1e9 else 1e9
            ss = spec_s(port) if f0.get("spec_s") and ms < 1e9 else None
            if ms >= 1e9:                     # (not measured: the first passes crawled)
                say("the first passes took " + (f"{w:.0f} s" if w is not None else "over a minute")
                    + " (memory spilled, or the model was still being read from disk): fitting again")
            else:
                say(f"pass with 8 lanes: {ms:.1f} ms (when fitted {f0['pass_ms_8']:.1f} ms); with all {f0['lanes']}: {ma:.1f} ms"
                    + (f" (when fitted {f0['pass_ms_all']:.1f} ms)" if f0.get("pass_ms_all") else ""))
            used, total = card()
            # (room for a fifth more lanes at least: the lanes are counted with their buffers, so a little room is always
            # left, and a kept fit of 158 lanes was fitted again for 156 on every load, 19 Nov)
            roomy = (total - used - margin_with(margin, holders(eng.pid(port)))
                     > max(10, 0.2 * f0["lanes"]) * f0.get("per_lane_mb", 60) * LANE_EXTRA)
            whole = all_ok(f0["lanes"], ma, ms) and (not f0.get("pass_ms_all") or ma <= f0["pass_ms_all"] * 1.8 + 6)
            # (not when a program's lane maximum is reached: its every load was fitted again, 25 Nov)
            if roomy and f0["lanes"] < cap(MAX_SEQ - vault):
                say("the card has room for more lanes than when fitted: fitting again")
                note("refit")
            elif whole and ms <= f0["pass_ms_8"] * 1.6 + 4 and (not f0.get("spec_s") or (ss <= f0["spec_s"] * 3 + 0.5 and ss <= SPEC_SPILL_S)):
                st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5).read())
                return {**f0, "spec": st.get("spec", 0), "spec_mtp": st.get("spec_mtp", st.get("spec", 0)), "spec_lookup": st.get("spec_lookup", 0), "pass_ms_8": round(ms, 1), "pass_ms_all": round(ma, 1), "cached": True}
            else:
                say("slower than when fitted (other programs hold more of the card now): fitting again")
                note("refit")
        # (fitted again from what was asked for, with the engine's own memory given back: measured beside the engine of
        # the kept fit, the probe cut a 16k conversation context to 8k and the parked states from 34 to 8, 25 Nov)
        eng.stop(port)
        common, vault = dict(asked), asked_vault
    # 1. the probe
    cut = probe_budget(path, common, margin, mmproj)
    if cut:
        say("this model's attention cache is large: " + ", ".join(
            {"vault": f"{cut.get('vault')} parked conversations instead of {common.get('vault')}",
             "spec_ctx": f"{cut.get('spec_ctx', 0) // 1024}k of drafting context",
             "ctx_extra": f"{cut.get('ctx_extra', 0) // 1024}k of conversation context"}[k] for k in cut))
        common.update(cut)
        vault = int(common.get("vault") or vault)
    say("probe: 8 lanes")
    t_probe = time.time()
    eng.start(lanes=8, **common)
    nums = log_numbers(port)
    if not responds(port):
        held = sorted(holders(eng.pid(port)) or [], key=lambda x: -x[1])
        eng.stop(port)
        raise RuntimeError("the model does not fit on the graphics card next to what other programs hold there"
                           + (" (" + ", ".join(f"{n} {mb / 1024:.1f} GB" for n, mb in held[:4]) + ")" if held else "")
                           + ": the engine started but could not write. Close some of them, or choose a smaller model, and load it again.")
    workout(port, 8)
    base = min(pass_ms(port, 8), pass_ms(port, 8))
    sbase = spec_s(port)
    if sbase and sbase > SPEC_SPILL_S:
        # even next to 8 lanes the drafting context does not fit (other programs hold too much of the card)
        say(f"the drafting context reads in {sbase:.1f} s instead of under a second: not enough memory for drafting, it is left off")
        common.pop("spec", None); common.pop("spec_ctx", None); common.pop("lookup", None)
        key = key.replace("|True", "|False")
        note("restart", time.time() - t_probe)
        eng.start(lanes=8, **common)
        nums = log_numbers(port)
        base = min(pass_ms(port, 8), pass_ms(port, 8))
        sbase = None
    used, total = card()
    shared_probe = spilled_mb(eng.pid(port))      # the engine's memory in the PC's memory next to the probe (see below)
    # recurrent state and attention cells of a lane, and about 5% more (output rows, sampling buffers: measured on Bonsai-2)
    per_cell = nums["kv"] / max(1, nums["cells"]) * 1.05
    per_lane = (nums["rs"] / max(1, nums["seqs"]) + nums["kv"] / max(1, nums["cells"]) * CTX) * 1.05
    others = holders(eng.pid(port))          # named in the result: closing them leaves room for more lanes
    keep = margin_with(margin, others)
    if keep > margin:
        say(f"other programs hold {sum(mb for _, mb in others) / 1024:.1f} GB of the card, and that changes: {keep} MB are left free rather than {margin}")
    free = total - used - keep
    probe_ctx = common["ctx_extra"]
    # (a lane takes about a quarter more than its states: the engine's buffers grow with the lane count. Without it the
    # first start spilled every time and was found out minutes later: 195 lanes counted where 156 fit, 138 for 110,
    # 78 for 66, 19 Nov)
    lanes_for = lambda c: cap(int(min(MAX_SEQ - vault, 8 + max(0, free - (c - probe_ctx) * per_cell) // max(1.0, per_lane * LANE_EXTRA))))
    if auto:
        fitting = [c for c in contexts if lanes_for(c) >= AUTO_MIN_LANES] or [8192]
        common["ctx_extra"] = max(fitting)
        say(f"conversation context {common['ctx_extra'] // 1024}k tokens ({common['ctx_extra'] * per_cell / 1024:.1f} GB): the largest that leaves {AUTO_MIN_LANES} lanes or more")
    lanes = lanes_for(common["ctx_extra"])
    say(f"a lane costs {per_lane:.0f} MB ({per_lane * LANE_EXTRA:.0f} with its buffers); the card has {total - used:.0f} MB free next to the model and other programs -> {lanes} lanes")
    if lanes_fixed:
        lanes = max(8, min(int(lanes_fixed), MAX_SEQ - vault))
        say(f"{lanes} lanes, as set in the settings")
    elif key in fits and fits[key]["lanes"] < lanes and not fresh:
        lanes = max(8, min(lanes, int(fits[key]["lanes"] * 1.1)))    # it spilled at more before
    # 2. the real start, checked
    for attempt in range(5):
        if attempt:
            note("restart", time.time() - t_try)
        t_try = time.time()
        say(f"starting with {lanes} lanes and {common['ctx_extra'] // 1024}k of conversation context")
        try:
            t_start = eng.start(lanes=lanes, **common)
        except RuntimeError as e:
            # Linux: an overfull card does not slow the engine down, the engine cannot get its memory and stops
            if "stopped while starting" not in str(e) or lanes <= 8 or attempt >= 4:
                raise
            lanes = max(8, int(lanes * 0.85))
            say(f"the engine could not start with that many lanes (the card is full): {lanes} lanes")
            continue
        # the estimate above can miss what the engine allocates (buffers it sizes by lanes): what the card really has
        # left is checked before any load goes on it - an overfull card stalls the workout itself
        used, total = card()
        short = margin * FREE_SHARE - (total - used)
        if short > 0 and attempt < 4:
            need = short + margin * (1 - FREE_SHARE)          # back to the whole margin
            smaller = [c for c in contexts if c < common["ctx_extra"]]
            if auto and smaller:
                fits_now = [c for c in smaller if (common["ctx_extra"] - c) * per_cell >= need]
                common["ctx_extra"] = max(fits_now) if fits_now else min(smaller)
                say(f"only {total - used} MB of the card is free: {common['ctx_extra'] // 1024}k of conversation context")
                continue
            if lanes > 8:
                lanes = max(8, lanes - int(need / max(1.0, per_lane)) - 1)
                say(f"only {total - used} MB of the card is free: {lanes} lanes")
                continue
        # a workout that crawls is a card that spilled to the PC's memory: fewer lanes. (With the drafting route reading 64
        # tokens a pass, the probe counted room for 91 lanes; their first pass then spilled, and the load stood at
        # "starting with 91 lanes" for over 9 minutes at 100% of the card, 19 Nov. It takes ~1 s when it fits.)
        w = workout(port, lanes, limit=WORKOUT_SPILL_S)
        if (w is None or w > WORKOUT_SPILL_S) and lanes > 8 and attempt < 4:
            lanes = max(8, int(lanes * 0.8))
            say(f"the first pass took {'over a minute' if w is None else f'{w:.0f} s'}: memory spilled to the PC, {lanes} lanes")
            continue
        ms = pass_ms_warm(port, 8, good=base * 1.6 + 4)
        ma = all_ms(port, lanes, base=base)
        ss = None
        if sbase:
            ss = spec_s(port)
        say(f"pass with 8 lanes: {ms:.1f} ms (probe {base:.1f} ms); with all {lanes}: {ma:.1f} ms, {lanes * 1000 / max(0.1, ma):.0f} tokens a second together"
            + (f"; drafting context reads in {ss:.2f} s (probe {sbase:.2f} s)" if ss else ""))
        if (ms <= base * 1.6 + 4 and all_ok(lanes, ma, base) and (not sbase or ss <= sbase * 3 + 0.5)) or lanes <= 8:
            if sbase and ss and ss > SPEC_SPILL_S:     # 8 lanes and drafting still spills: drafting off
                say("the drafting context still spills: drafting is left off")
                common.pop("spec", None); common.pop("spec_ctx", None); common.pop("lookup", None)
                note("restart", t_start + 5)
                eng.start(lanes=lanes, **common)
                ms, sbase, ss = pass_ms_warm(port, 8, good=base * 1.6 + 4), None, None
                ma = all_ms(port, lanes)
            break
        # Slower than the probe with no more of the engine's memory in the PC's memory than next to the probe: nothing
        # spilled, the larger conversation context did it. A model with attention in every layer reads its whole
        # attention cache on every pass (Inkling-2-Mini Q2_K_XL, 23 Nov: 8 lanes 35 ms a pass next to 8k of context, 70 ms
        # next to 32k; taken for spilled memory, it was cut from 23 lanes to 9 while the 32k stayed). Back to the probe's
        # context, with the lanes it leaves room for.
        if auto and attempt < 4 and common["ctx_extra"] > probe_ctx and spilled_mb(eng.pid(port)) - shared_probe < SPILL_MB:
            smaller = [c for c in contexts if probe_ctx <= c < common["ctx_extra"]]
            common["ctx_extra"] = max(smaller) if smaller else probe_ctx
            if not lanes_fixed:
                lanes = max(lanes, min(lanes_for(common["ctx_extra"]), int(lanes * 1.5)))
            say(f"slower than the probe with no memory spilled: the attention cache is read on every pass, "
                f"{common['ctx_extra'] // 1024}k of conversation context, {lanes} lanes")
            continue
        lanes = max(8, int(lanes * 0.85))
        say(f"slower than the probe: memory spilled to the PC, {lanes} lanes")
    st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5).read())
    # the lanes the engine runs with, which the speeds below were measured on: after the last try the count was cut
    # without a start, and 52 lanes were noted for an engine running 62 (23 Nov)
    lanes = int(st.get("lanes") or lanes)
    chosen = common["ctx_extra"]
    options = [{"ctx": c, "lanes": max(0, int(min(MAX_SEQ - vault, lanes + (chosen - c) * per_cell / max(1.0, per_lane))))} for c in contexts]
    out = {"lanes": lanes, "per_lane_mb": round(per_lane, 1), "per_cell_mb": round(per_cell, 4), "ctx_extra": chosen, "options": options, "model_mb": round(nums["model"]), "pass_ms_8": round(ms, 1), "pass_ms_all": round(ma, 1), "spec_s": round(ss, 2) if ss else None,
           "spec": st.get("spec", 0), "spec_mtp": st.get("spec_mtp", st.get("spec", 0)), "spec_lookup": st.get("spec_lookup", 0), "at": time.strftime("%Y-%m-%d %H:%M"), "used_by_others_mb": None,
           "shared_mb": spilled_mb(eng.pid(port)),     # the engine's memory in the PC's memory when it fitted (see spilled_mb)
           "free_mb": total - used,             # what the card had left next to the engine when it was fitted
           "others": others}                    # other programs that held 500 MB or more of the card
    out["vision"] = bool(st.get("vision"))
    # (as for a mixture of experts: a fit whose speed checks were cut short is not kept as the baseline)
    if not lanes_fixed and max(ms, ma) < 10000:
        fits[key] = out
    os.makedirs(os.path.dirname(FIT), exist_ok=True)
    json.dump(fits, open(FIT, "w", encoding="utf8"), indent=1)
    return out


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:])
    print(json.dumps(fit(sys.argv[1], mtp=kv.get("mtp", "0") == "1", ctx_extra=kv.get("ctx_extra", "auto") if kv.get("ctx_extra", "auto") == "auto" else int(kv["ctx_extra"]), margin=int(kv.get("margin", 1200))), indent=1))
