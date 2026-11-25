"""The chat models already on this computer: GGUF files in the Hugging Face cache (where Unsloth Studio and many other
apps download), in the folders of LM Studio, Jan, GPT4All, llama.cpp and text-generation-webui, in Ollama's store, in a
few usual places (a "models" folder, Downloads), in Rapidli's own model folder and in any folder the user adds in
Settings. Only the header of each file is read.

  python app/models.py            a table of the models found
  python app/models.py refresh    ... reading every header again
Each model: {"id", "repo", "file", "path", "bytes", "arch", "name", "layers", "context", "mtp", "quant", "parts",
"source"}"""
import json, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data", "models.json")

# architectures that are not chat models (vision projectors, image/audio models, encoders)
NOT_CHAT = re.compile(r"^(clip|mmproj|qwenimage|qwen_image|flux|sd|stable|wan|whisper|wav|encodec|bert|nomic-bert|jina-bert|t5encoder|"
                      r"gemma-embedding|embeddinggemma|wavtokenizer|snac|vits|parakeet|qwen3asr|qwen3_asr|asr|dflash|dspark)", re.I)
# (dspark: a drafter shipped beside a model, "dspark-Inkling-2-Mini-Q8_0.gguf", architecture dflash; it was listed as
# a chat model of its own and failed to load, 24 Nov)
NOT_CHAT_FILE = re.compile(r"(mmproj|dflash|dspark|draft|embedding|bge-|whisper|asr|tts|vae|clip|image)", re.I)


SKIP_DIRS = {"blobs", ".no_exist", "node_modules", "site-packages", "__pycache__", ".git", "venv", ".venv", "$recycle.bin"}
LAST = {"places": [], "at": 0.0}      # what the last scan looked at: [{"folder", "source", "found"}]


def _user_folders():
    """(the folder downloads go to, the folders added in Settings)"""
    try:
        import settings
        cfg = settings.load()
        return (cfg.get("downloads") or {}).get("folder") or "", list((cfg.get("models") or {}).get("folders") or [])
    except Exception:
        return "", []


def places():
    """Where models are looked for: [(folder, where it belongs to)], only folders that exist, each once."""
    home, env = os.path.expanduser("~"), os.environ
    j = os.path.join
    out = []
    for v in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(v):
            out.append((env[v], "Hugging Face cache"))
    if env.get("HF_HOME"):
        out.append((j(env["HF_HOME"], "hub"), "Hugging Face cache"))
    if env.get("XDG_CACHE_HOME"):
        out.append((j(env["XDG_CACHE_HOME"], "huggingface", "hub"), "Hugging Face cache"))
    out.append((j(home, ".cache", "huggingface", "hub"), "Hugging Face cache"))
    down, added = _user_folders()
    out.append((j(os.path.dirname(HERE), "models"), "Rapidli"))
    if down:
        out.append((down, "Rapidli"))
    out += [(f, "your folder") for f in added]
    local = env.get("LOCALAPPDATA") or j(home, ".local", "share")
    roaming = env.get("APPDATA") or j(home, ".config")
    out += [(j(home, ".lmstudio", "models"), "LM Studio"), (j(home, ".cache", "lm-studio", "models"), "LM Studio"),
            (j(home, "jan", "models"), "Jan"), (j(roaming, "Jan", "data", "models"), "Jan"), (j(home, ".local", "share", "Jan", "data", "models"), "Jan"),
            (j(local, "nomic.ai", "GPT4All"), "GPT4All"), (j(home, ".local", "share", "nomic.ai", "GPT4All"), "GPT4All"),
            (j(local, "llama.cpp"), "llama.cpp cache"), (j(home, ".cache", "llama.cpp"), "llama.cpp cache"),
            (j(home, "Library", "Caches", "llama.cpp"), "llama.cpp cache"),
            (j(home, "text-generation-webui", "user_data", "models"), "text-generation-webui"), (j(home, "text-generation-webui", "models"), "text-generation-webui"),
            (j(home, ".unsloth", "models"), "Unsloth Studio"), (j(home, "models"), "models folder"), (j(home, "Models"), "models folder")]
    if os.name == "nt":                           # a "models" folder or a cache at the top of a drive
        for d in "CDEFGH":
            out += [(f"{d}:\\models", "models folder"), (f"{d}:\\hf\\hub", "Hugging Face cache"), (f"{d}:\\huggingface\\hub", "Hugging Face cache")]
    else:
        out += [("/models", "models folder"), ("/opt/models", "models folder"), ("/srv/models", "models folder")]
    out.append((j(home, "Downloads"), "Downloads"))
    seen, res = set(), []
    for folder, source in out:
        try:
            k = os.path.normcase(os.path.realpath(folder))
        except Exception:
            continue
        if os.path.isdir(folder) and k not in seen:
            seen.add(k)
            res.append((folder, source))
    return res


def roots():
    return [f for f, _ in places()]


def ollama_models():
    """The models in Ollama's store: [(blob path, "name-tag.gguf", "ollama/name", projector blob or None)]. Ollama keeps a
    model as a file named by its hash, and a manifest per name and tag that says which file it is."""
    home, out, seen = os.path.expanduser("~"), [], set()
    for base in [os.environ.get("OLLAMA_MODELS"), os.path.join(home, ".ollama", "models"), "/usr/share/ollama/.ollama/models",
                 "/var/lib/ollama/.ollama/models"]:
        man = os.path.join(base, "manifests") if base else ""
        if not base or not os.path.isdir(man):
            continue
        for dirpath, _, files in os.walk(man):
            for fn in files:
                try:
                    layers = json.load(open(os.path.join(dirpath, fn), encoding="utf8")).get("layers") or []
                except Exception:
                    continue
                blob = lambda kind: next((os.path.join(base, "blobs", str(l.get("digest", "")).replace(":", "-")) for l in layers
                                          if str(l.get("mediaType", "")).endswith("image." + kind)), None)
                model = blob("model")
                if not model or not os.path.isfile(model) or model in seen:
                    continue
                seen.add(model)
                rel = os.path.relpath(dirpath, man).replace("\\", "/").split("/")     # registry / namespace / name
                name = rel[-1]
                owner = rel[-2] if len(rel) >= 2 and rel[-2] != "library" else ""
                out.append((model, f"{name}-{fn}.gguf", "ollama/" + (owner + "/" if owner else "") + name, blob("projector")))
    return out


_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def header(path, want_template=True):
    """The metadata of a GGUF file (arrays are skipped, except short ones). Reads only the start of the file. path may
    also be an open file-like object with read() and seek(n, 1) (a file on Hugging Face read in ranges, downloads.py)."""
    import contextlib
    meta = {}
    with (contextlib.nullcontext(path) if hasattr(path, "read") else open(path, "rb")) as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n:
                raise EOFError
            return b

        def u(fmt):
            return struct.unpack(fmt, rd(struct.calcsize(fmt)))[0]

        def string():
            return rd(u("<Q")).decode("utf8", "replace")

        def value(t):
            if t in _SCALAR:
                return u(_SCALAR[t])
            if t == 8:
                return string()
            if t == 9:
                it, n = u("<I"), u("<Q")
                if it in _SCALAR:
                    size = struct.calcsize(_SCALAR[it])
                    if n <= 64:
                        return [u(_SCALAR[it]) for _ in range(n)]
                    f.seek(size * n, 1)
                    return None
                if it == 8:
                    for _ in range(n):
                        f.seek(u("<Q"), 1)
                    return None
                return [value(it) for _ in range(n)]
            raise ValueError("unknown GGUF value type %d" % t)

        if rd(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        u("<I")
        u("<Q")
        n_kv = u("<Q")
        for _ in range(n_kv):
            key = string()
            v = value(u("<I"))
            if key == "tokenizer.chat_template" and not want_template:
                v = bool(v)
            meta[key] = v
    return meta


EXPERTS_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")     # the routed experts (as the engine's --experts-ram)
TABLE_RE = re.compile(r"^per_layer_token_embd\.")                        # an n-gram lookup table: a few rows read per token


def tensor_split(f, size, ranges=None):
    """Where the bytes of one GGUF file go: {"experts", "table", "rest"} in bytes. f: an open binary file (or anything with
    read(n)), size: the file's size. A tensor's size is the distance to the next tensor's data (the last one: to the end
    of the file), so no table of quantization types is needed. "experts": the routed experts of a mixture-of-experts model,
    which can stay in the PC's memory; "table": an n-gram lookup table (Qwen3.8-Flash-Next's 28.8 GB), read a few rows per
    token, which can stay on the disk; "rest": what every token needs, on the card."""
    def rd(n):
        b = f.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def u(fmt):
        return struct.unpack(fmt, rd(struct.calcsize(fmt)))[0]

    def skip_value(t):
        if t in _SCALAR:
            rd(struct.calcsize(_SCALAR[t]))
        elif t == 8:
            rd(u("<Q"))
        elif t == 9:
            it, n = u("<I"), u("<Q")
            if it in _SCALAR:
                rd(struct.calcsize(_SCALAR[it]) * n)
            else:
                for _ in range(n):
                    skip_value(it)
        else:
            raise ValueError("unknown GGUF value type %d" % t)

    if rd(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    u("<I")
    n_t, n_kv = u("<Q"), u("<Q")
    align, pos = 32, 24
    for _ in range(n_kv):
        key = rd(u("<Q")).decode("utf8", "replace")
        t = u("<I")
        if key == "general.alignment" and t == 4:
            align = u("<I")
        else:
            skip_value(t)
    tensors = []
    for _ in range(n_t):
        name = rd(u("<Q")).decode("utf8", "replace")
        nd = u("<I")
        rd(8 * nd)
        u("<I")
        tensors.append((u("<Q"), name))
    head = f.tell() if hasattr(f, "tell") else None
    if head is None:
        raise ValueError("the reader cannot tell its position")
    data = (head + align - 1) // align * align
    tensors.sort()
    out = {"experts": 0, "table": 0, "rest": 0}
    for i, (off, name) in enumerate(tensors):
        end = tensors[i + 1][0] if i + 1 < len(tensors) else size - data
        kind = "experts" if EXPERTS_RE.search(name) else "table" if TABLE_RE.search(name) else "rest"
        out[kind] += max(0, end - off)
        if ranges is not None and end > off:       # (where in the file: data + offset, for reading them ahead)
            ranges.append((kind, data + off, end - off))
    return out


def expert_ranges(path, kinds=("experts",)):
    """[(file, start, length)] of the routed experts of a model (kinds: also "table", its n-gram table), in every part,
    neighbouring ranges joined; [] when a part cannot be read"""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    paths = [path[:m.start()] + f"-{k:05d}-of-{m.group(2)}.gguf" for k in range(1, int(m.group(2)) + 1)] if m else [path]
    out = []
    try:
        for p in paths:
            rs = []
            with open(p, "rb") as f:
                tensor_split(f, os.path.getsize(p), rs)
            for kind, a, n in sorted((r for r in rs if r[0] in kinds), key=lambda r: r[1]):
                if out and out[-1][0] == p and out[-1][1] + out[-1][2] == a:
                    out[-1] = (p, out[-1][1], out[-1][2] + n)
                else:
                    out.append((p, a, n))
    except (OSError, ValueError, EOFError, struct.error):
        return []
    return out


def split_bytes(path):
    """tensor_split over every part of a model (a split model is named by any of its parts) -> {"experts", "table",
    "rest"}, or None when a part cannot be read."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    paths = [path[:m.start()] + f"-{k:05d}-of-{m.group(2)}.gguf" for k in range(1, int(m.group(2)) + 1)] if m else [path]
    out = {"experts": 0, "table": 0, "rest": 0}
    try:
        for p in paths:
            with open(p, "rb") as f:
                for k, v in tensor_split(f, os.path.getsize(p)).items():
                    out[k] += v
    except (OSError, ValueError, EOFError, struct.error):
        return None
    return out


def describe(path):
    m = header(path)
    arch = str(m.get("general.architecture", "?"))
    a = lambda k, d=None: m.get(f"{arch}.{k}", d)
    nextn = a("nextn_predict_layers", 0) or 0
    base = os.path.basename(path)
    q = re.search(r"((?:UD-)?(?:IQ\d_\w+|Q\d_\w+|Q\d_\d|BF16|F16|F32|Q8_0|MXFP4\w*|TQ\d_\d))", base, re.I)
    parts = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", base)
    return {"arch": arch, "name": m.get("general.name") or base, "layers": a("block_count"), "context": a("context_length"),
            "embd": a("embedding_length"), "heads_kv": a("attention.head_count_kv"), "mtp": int(nextn) > 0,
            "template": bool(m.get("tokenizer.chat_template")), "quant": q.group(1) if q else "",
            "parts": int(parts.group(2)) if parts else 1, "ssm": a("ssm.state_size") is not None,
            "experts": a("expert_count") or 0,
            # a mixture of experts: where its bytes go (experts can stay in the PC's memory, an n-gram table on the disk)
            "split": split_bytes(path) if (a("expert_count") or 0) > 0 else None}


def scan(refresh=False):
    """-> list of chat models, largest file first within a repository. Every place is looked through on each call (a new
    download shows up); a file's header is read once and kept in data/models.json by its size and time, and read again
    when refresh is set."""
    old = {}
    if not refresh and os.path.exists(CACHE):
        try:
            old = {x["path"]: x for x in json.load(open(CACHE, encoding="utf8"))["all"]}
        except Exception:
            old = {}
    found, report, seen = [], [], set()

    def add(path, fn, repo, source, mmproj=None):
        try:
            st = os.stat(path)
            real = os.path.normcase(os.path.realpath(path))
        except OSError:
            return False
        # one file under two names (a hard link, as Ollama's store and some download managers make) is one model:
        # C:/models/x.gguf and C:/models/fusion/x.gguf were counted as two (19 Nov)
        ident = ("id", st.st_dev, st.st_ino) if getattr(st, "st_ino", 0) else None
        if st.st_size < 300 * 1024 * 1024 or real in seen or ident in seen:
            return False
        parts = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", fn)
        if parts and parts.group(1) != "00001":
            return False                          # a model in several files is loaded from its first part
        seen.add(real)
        if ident:
            seen.add(ident)
        rec = old.get(path)
        if not rec or rec.get("bytes0", rec.get("bytes")) != st.st_size or rec.get("mtime") != int(st.st_mtime):
            try:
                d = describe(path)
            except Exception as e:
                d = {"arch": "?", "error": str(e)[:120]}
            rec = {"path": path, "bytes": st.st_size, "bytes0": st.st_size, "mtime": int(st.st_mtime), **d}
            if rec.get("parts", 1) > 1:           # the whole model: all its parts
                folder = os.path.dirname(path)
                rec["bytes"] = sum(os.path.getsize(os.path.join(folder, x)) for x in os.listdir(folder)
                                   if re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", x) == re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", fn))
        rec.update(file=fn, repo=repo, source=source)
        if not rec.get("quant"):                  # (a file named by its hash: the quantization is in the name given to it)
            q = re.search(r"(?i)[-._:]((?:q\d+(?:_[a-z0-9]+)*|bf16|f16|f32|iq\d+(?:_[a-z0-9]+)*))(?=[-._]|$)", fn[:-5])
            if q:
                rec["quant"] = q.group(1).upper()
        rec["chat"] = bool(rec.get("arch") not in ("?", None) and not NOT_CHAT.match(str(rec.get("arch"))) and not NOT_CHAT_FILE.search(fn))
        rec["id"] = repo + ":" + re.sub(r"\.gguf$", "", fn)
        if mmproj and os.path.isfile(mmproj):
            rec["mmproj_given"] = mmproj
        found.append(rec)
        return rec["chat"]

    for root, source in places():
        n = 0
        deep = 2 if source == "Downloads" else 12       # (Downloads: the folder itself and one level down)
        for dirpath, dirnames, files in os.walk(root):
            depth = os.path.relpath(dirpath, root).count(os.sep) + (0 if dirpath == root else 1)
            dirnames[:] = [] if depth >= deep else [d for d in dirnames if d.lower() not in SKIP_DIRS]
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    m = re.search(r"models--([^\\/]+?)--([^\\/]+)", dirpath)
                    n += bool(add(os.path.join(dirpath, fn), fn, (m.group(1) + "/" + m.group(2)) if m else os.path.basename(dirpath), source))
        report.append({"folder": root, "source": source, "found": n})
    n, base = 0, ""
    for path, fn, repo, mm in ollama_models():
        n += bool(add(path, fn, repo, "Ollama", mm))
        base = os.path.dirname(os.path.dirname(path))
    if base:
        report.append({"folder": base, "source": "Ollama", "found": n})
    LAST.update(places=report, at=__import__("time").time())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump({"all": found}, open(CACHE, "w", encoding="utf8"), indent=1)
    chat = [r for r in found if r["chat"]]
    for r in chat:
        r["mmproj"] = r.get("mmproj_given") or projector(r["path"])
        if r["repo"].count("--") == 1 and "/" not in r["repo"]:      # downloaded by Rapidli: <owner>--<name>
            r["repo"] = r["repo"].replace("--", "/")
    chat.sort(key=lambda r: (r["repo"].lower(), -r["bytes"]))
    return chat


def projectors(path):
    """The vision projectors (mmproj-*.gguf) in the model's folder, the preferred one first: 16-bit, then 8-bit, then 32-bit."""
    d = os.path.dirname(path)
    try:
        cands = [f for f in os.listdir(d) if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
    except OSError:
        return []
    rank = lambda f: (0 if re.search(r"f16|bf16", f, re.I) else 1 if re.search(r"q8", f, re.I) else 2 if re.search(r"f32", f, re.I) else 3, f)
    return [os.path.join(d, f) for f in sorted(cands, key=rank)]


def projector(path):
    """The vision projector loaded with a model unless another is chosen, or None."""
    return next(iter(projectors(path)), None)


if __name__ == "__main__":
    for r in scan(refresh="refresh" in sys.argv):
        print(f"{r['source'][:18]:18s} {r['repo'][:46]:46s} {r['file'][:44]:44s} {r['bytes'] / 2**30:5.1f} GB  {r['arch']:12s} layers {r.get('layers')} ctx {r.get('context')} "
              f"mtp {'yes' if r.get('mtp') else 'no '} ssm {'yes' if r.get('ssm') else 'no '} experts {r.get('experts')} tmpl {'yes' if r.get('template') else 'no'}")
