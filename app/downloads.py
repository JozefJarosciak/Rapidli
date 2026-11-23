"""Downloads of GGUF models from Hugging Face, started by the user on the Settings screen.

  files(repo, token)            -> the GGUF files of a repository with their sizes (the Hugging Face API)
  start(repo, file, folder)     -> a download in the background (resumes a .part left over); status() lists them
  cancel(id)
A model lands in <folder>/<owner>--<name>/<file>; Rapidli's model list looks there too."""
import json, os, re, threading, time, urllib.parse, urllib.request, uuid
import fetch

# the hub: huggingface.co, or the mirror named by HF_ENDPOINT (the variable Hugging Face's own tools read)
HUB = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
API = HUB + "/api/models/"
_jobs = {}
_lock = threading.Lock()


def _headers(token):
    h = {"User-Agent": "Rapidli/1.0"}
    if token:
        h["Authorization"] = "Bearer " + token
    return h


def files(repo, token=""):
    repo = repo.strip().strip("/")
    if repo.startswith("https://huggingface.co/"):
        repo = "/".join(repo.split("/")[3:5])
    if repo.count("/") != 1:
        raise ValueError('a repository is written as "owner/name", e.g. unsloth/Qwen3.5-2B-GGUF')
    url = API + urllib.parse.quote(repo, safe="/") + "/tree/main?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers(token)), timeout=20) as r:
        items = json.loads(r.read())
    out = []
    for it in items:
        p = it.get("path", "")
        if it.get("type") == "file" and p.lower().endswith(".gguf"):
            size = (it.get("lfs") or {}).get("size") or it.get("size") or 0
            out.append({"file": p, "bytes": size, "gb": round(size / 2**30, 2), "mmproj": "mmproj" in p.lower()})
    out.sort(key=lambda x: (x["mmproj"], x["file"]))
    return {"repo": repo, "files": out}


def _run(job, token):
    url = HUB + "/" + urllib.parse.quote(job["repo"], safe="/") + "/resolve/main/" + urllib.parse.quote(job["file"])
    fetch.download(url, job["path"], job, headers=_headers(token), connections=job.get("want", 6))


def start(repo, file, folder, token="", connections=6):
    repo, file = str(repo).strip().strip("/"), str(file).strip()
    # (the names come from the page: a file stays inside the model's own folder)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or any(x in (".", "..") for x in repo.split("/")):
        raise ValueError("not a Hugging Face model name")
    parts = file.split("/")
    if not file or "\\" in file or ":" in file or any(x in ("", ".", "..") for x in parts):
        raise ValueError("not a file of the model")
    path = os.path.join(folder, repo.replace("/", "--"), *parts)
    with _lock:
        for j in _jobs.values():
            if j["path"] == path and j["state"] in ("starting", "downloading"):
                return j["id"]
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "repo": repo, "file": file, "path": path, "state": "starting", "done": 0, "total": 0, "mb_s": 0, "cancel": False,
               "error": "", "at": time.time(), "want": connections, "connections": 0}
        if os.path.exists(path):
            job.update(state="done", done=os.path.getsize(path), total=os.path.getsize(path))
        _jobs[jid] = job
    if job["state"] != "done":
        threading.Thread(target=_run, args=(job, token), daemon=True).start()
    return jid


def cancel(jid):
    j = _jobs.get(jid)
    if j:
        j["cancel"] = True
    return bool(j)


def status():
    return [{k: v for k, v in j.items() if k not in ("cancel", "want")} for j in sorted(_jobs.values(), key=lambda x: -x["at"])]


# ---------------------------------------------------------------- finding models on Hugging Face
SORTS = {"trending": "trendingScore", "downloads": "downloads", "likes": "likes", "new": "lastModified"}
# what a GGUF repository is for, when it says so and it is not chat (image, speech and embedding models are left out)
NOT_CHAT_TASK = {"text-to-image", "image-to-image", "text-to-video", "image-to-video", "text-to-speech", "text-to-audio",
                 "automatic-speech-recognition", "feature-extraction", "sentence-similarity", "audio-to-audio", "image-to-3d",
                 "text-to-3d", "unconditional-image-generation", "video-to-video", "text-ranking", "audio-classification"}
QUANT = r"(?i)[-._]((?:i\d-)?(?:q\d+(?:_[a-z0-9]+)*|bf16|f16|f32|fp16|iq\d+(?:_[a-z0-9]+)*|tq\d(?:_\d)?))(?=[-._]|$)"


def _get(url, token="", timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers(token)), timeout=timeout) as r:
        return json.loads(r.read())


def repo_name(text):
    """"owner/name" from a name or a huggingface.co address, or None when the text is something to search for."""
    t = str(text or "").strip().strip("/")
    if t.startswith("https://huggingface.co/"):
        t = "/".join(t.split("/")[3:5])
    return t if t.count("/") == 1 and " " not in t and all(t.split("/")) else None


def search(q="", sort="trending", limit=30):
    """GGUF chat models on Hugging Face that match q (all of them when q is empty), most trending / downloaded / liked /
    recent first. Only the words typed are sent; no token."""
    import re
    url = (HUB + "/api/models?filter=gguf&full=false&direction=-1&limit=" + str(max(5, min(int(limit), 60)) + 10)
           + "&sort=" + SORTS.get(sort, "trendingScore") + ("&search=" + urllib.parse.quote(q.strip()[:80]) if q.strip() else ""))
    out = []
    for it in _get(url):
        rid = it.get("id") or it.get("modelId") or ""
        if "/" not in rid or it.get("private") or it.get("pipeline_tag") in NOT_CHAT_TASK:
            continue
        tags = it.get("tags") or []
        out.append({"repo": rid, "owner": rid.split("/")[0], "name": rid.split("/")[1], "likes": it.get("likes") or 0,
                    "downloads": it.get("downloads") or 0, "updated": it.get("lastModified") or it.get("createdAt") or "",
                    "task": it.get("pipeline_tag") or "", "gated": bool(it.get("gated")),
                    "vision": it.get("pipeline_tag") == "image-text-to-text" or "vision" in tags or "multimodal" in tags})
    return out[:max(5, min(int(limit), 60))]


class _Ranged:
    """read(n) and tell() over a file on Hugging Face, fetched in blocks with Range requests: enough to read a GGUF's
    header (its tensors' names and places) without downloading the file. At most LIMIT bytes."""
    BLOCK, LIMIT = 4 << 20, 96 << 20

    def __init__(self, url, token=""):
        self.url, self.token, self.buf, self.pos = url, token, b"", 0

    def read(self, n):
        while self.pos + n > len(self.buf):
            if len(self.buf) >= self.LIMIT:
                raise EOFError("the header is larger than %d MB" % (self.LIMIT >> 20))
            h = dict(_headers(self.token), Range=f"bytes={len(self.buf)}-{len(self.buf) + self.BLOCK - 1}")
            with urllib.request.urlopen(urllib.request.Request(self.url, headers=h), timeout=30) as r:
                got = r.read()
            if not got:
                raise EOFError
            self.buf += got
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def tell(self):
        return self.pos


SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hf-splits.json")
_splits_lock = threading.Lock()


def file_split(repo, file, token=""):
    """Where a model on Hugging Face would go: {"experts", "table", "rest"} in bytes (see models.tensor_split), read from
    the headers of all its parts (a few MB each; a split model is named by its first part). Kept in data/hf-splits.json
    by repository, file and size."""
    import re
    import models
    repo = repo_name(repo)
    if not repo:
        raise ValueError('a repository is written as "owner/name"')
    it = _get(API + urllib.parse.quote(repo, safe="/") + "?blobs=true", token)
    sizes = {s.get("rfilename", ""): (s.get("lfs") or {}).get("size") or s.get("size") or 0 for s in it.get("siblings") or []}
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", file)
    parts = [file[:m.start()] + f"-{k:05d}-of-{m.group(2)}.gguf" for k in range(1, int(m.group(2)) + 1)] if m else [file]
    if any(p not in sizes for p in parts):
        raise ValueError("the repository does not have all parts of " + file)
    key = f"{repo}|{file}|{sum(sizes[p] for p in parts)}"
    with _splits_lock:
        try:
            kept = json.load(open(SPLITS, encoding="utf8"))
        except Exception:
            kept = {}
    if key in kept:
        return kept[key]
    out = {"experts": 0, "table": 0, "rest": 0}
    for p in parts:
        url = HUB + "/" + urllib.parse.quote(repo, safe="/") + "/resolve/main/" + urllib.parse.quote(p)
        for k, v in models.tensor_split(_Ranged(url, token), sizes[p]).items():
            out[k] += v
    with _splits_lock:
        try:
            kept = json.load(open(SPLITS, encoding="utf8"))
        except Exception:
            kept = {}
        kept[key] = out
        os.makedirs(os.path.dirname(SPLITS), exist_ok=True)
        json.dump(kept, open(SPLITS, "w", encoding="utf8"), indent=1)
    return out


def detail(repo, token=""):
    """One repository: its figures, tags and GGUF files (sizes, quantization; the parts of a split model as one entry)."""
    import re
    repo = repo_name(repo)
    if not repo:
        raise ValueError('a repository is written as "owner/name", e.g. unsloth/Qwen3.5-2B-GGUF')
    it = _get(API + urllib.parse.quote(repo, safe="/") + "?blobs=true", token)
    card, gg = it.get("cardData") or {}, it.get("gguf") or {}
    files, split = [], {}
    for s in it.get("siblings") or []:
        p = s.get("rfilename", "")
        if not p.lower().endswith(".gguf"):
            continue
        size = (s.get("lfs") or {}).get("size") or s.get("size") or 0
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p)
        if m:                                     # a model in several files: listed once, by its first part
            key = p[:m.start()]
            e = split.setdefault(key, {"file": None, "bytes": 0, "parts": int(m.group(2))})
            e["bytes"] += size
            if m.group(1) == "00001":
                e["file"] = p
            continue
        files.append({"file": p, "bytes": size})
    files += [{"file": e["file"], "bytes": e["bytes"], "parts": e["parts"]} for e in split.values() if e["file"]]
    for f in files:
        q = re.search(QUANT, os.path.basename(f["file"])[:-5])
        f.update(gb=round(f["bytes"] / 2**30, 2), mmproj="mmproj" in f["file"].lower(), quant=(q.group(1).upper() if q else ""))
    files.sort(key=lambda x: (x["mmproj"], x["bytes"]))
    lic = card.get("license")
    base = card.get("base_model")
    return {"repo": repo, "likes": it.get("likes") or 0, "downloads": it.get("downloads") or 0, "updated": it.get("lastModified") or "",
            "created": it.get("createdAt") or "", "task": it.get("pipeline_tag") or "", "gated": bool(it.get("gated")),
            "license": lic if isinstance(lic, str) else (lic[0] if lic else ""), "base": base if isinstance(base, str) else (base[0] if base else ""),
            "params": gg.get("total"), "arch": gg.get("architecture") or "", "context": gg.get("context_length"),
            "tags": [t for t in (it.get("tags") or []) if ":" not in t and t not in ("gguf", "region:us")][:14],
            "files": files, "url": HUB + "/" + repo}
