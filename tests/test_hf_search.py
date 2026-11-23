"""Settings > Get models, through the app: searching GGUF chat models on Hugging Face, sorting, and one repository's
files (the page marks which of them fit the card). Needs Rapidli running and the internet (it asks huggingface.co for lists only;
nothing is downloaded).

    python tests/test_hf_search.py"""
import json, sys, urllib.parse, urllib.request

B = "http://127.0.0.1:11842"
FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:240] if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)


def get(path):
    return json.loads(urllib.request.urlopen(B + path, timeout=120).read())


r = get("/api/hf/search?q=" + urllib.parse.quote("qwen") + "&sort=downloads&limit=12")
ms = r.get("models") or []
check("a search by name gives GGUF repositories", len(ms) >= 5 and all("qwen" in m["repo"].lower() for m in ms[:5]), [m["repo"] for m in ms[:4]])
check("each has an owner, a name, likes, downloads and a date", all(m.get("owner") and m.get("name") and "likes" in m and "downloads" in m and m.get("updated") for m in ms), {k: ms[0].get(k) for k in ("owner", "name", "likes", "downloads", "updated", "vision", "gated")} if ms else "")
d = [m["downloads"] for m in ms]
check("sorted by downloads, most first", d == sorted(d, reverse=True), d[:6])
r2 = get("/api/hf/search?q=" + urllib.parse.quote("qwen") + "&sort=likes&limit=12")
l2 = [m["likes"] for m in r2.get("models") or []]
check("sorted by likes, most first", bool(l2) and l2 == sorted(l2, reverse=True), l2[:6])
r3 = get("/api/hf/search?q=&sort=trending&limit=8")
check("with no words: what is trending", len(r3.get("models") or []) >= 5, [m["repo"] for m in (r3.get("models") or [])[:4]])
r4 = get("/api/hf/search?q=" + urllib.parse.quote("unsloth gemma") + "&sort=downloads&limit=8")
check("an owner and a name together narrow the search", any(m["owner"].lower() == "unsloth" and "gemma" in m["name"].lower() for m in r4.get("models") or []), [m["repo"] for m in (r4.get("models") or [])[:4]])
r5 = get("/api/hf/search?q=" + urllib.parse.quote("zzzz-no-such-model-qqqq") + "&sort=downloads&limit=8")
check("nothing found is an empty list, not an error", (r5.get("models") or []) == [] and not r5.get("error"), r5)

if ms:
    repo = next((m["repo"] for m in ms if not m.get("gated")), ms[0]["repo"])
    m = get("/api/hf/model?repo=" + urllib.parse.quote(repo, safe="/"))
    fs = m.get("files") or []
    check(f"a repository's files are listed with their size and quantization ({repo})", len(fs) >= 1 and all(f.get("file") and f.get("bytes") and f.get("gb") and f.get("quant") for f in fs if not f.get("mmproj")), [(f.get("file").rsplit("/", 1)[-1], f.get("gb"), f.get("quant"), f.get("parts")) for f in fs[:3]])
    check("a model in several files is one entry with its whole size", all(f.get("parts", 1) >= 1 for f in fs) and len({f["file"] for f in fs}) == len(fs), [f.get("parts") for f in fs[:6]])
    check("each file says whether it is on this PC already", all("here" in f for f in fs), sum(1 for f in fs if f.get("here")))
    check("the repository's details are given", all(k in m for k in ("likes", "downloads", "updated")), {k: m.get(k) for k in ("likes", "downloads", "license", "params", "context", "base")})
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
