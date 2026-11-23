"""Model discovery through the app: a model file that appears on the PC is found by "Look for new downloads" (the
model menu) and "Refresh" (Settings > Models), which both ask /api/models?refresh=1, and by Rapidli when it starts.
Needs Rapidli running. A temporary folder with two stand-in GGUF files is added to the model folders and taken
out again (about 600 MB while it runs); nothing is loaded.

    python tests/test_discovery_app.py            the refresh on demand
    python tests/test_discovery_app.py leave      ... and leave the folder in place (then restart Rapidli and run:)
    python tests/test_discovery_app.py started    the model list of a freshly started Rapidli holds the files; cleans up"""
import json, os, shutil, struct, sys, tempfile, urllib.request

B = "http://127.0.0.1:11842"
KEEP = os.path.join(tempfile.gettempdir(), "rapidli-discovery-test")
FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:220] if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)


def get(path):
    return json.loads(urllib.request.urlopen(B + path, timeout=120).read())


def post(path, body):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def gguf(path, name):
    """A GGUF file with no tensors: the architecture and a name."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = lambda k, v: s(k) + struct.pack("<I", 8) + s(v)
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kv("general.architecture", "llama") + kv("general.name", name))
        f.truncate(301 * 1024 * 1024)       # (files under 300 MB are not taken for chat models)


def names():
    return [m["path"] for m in get("/api/models")["models"]]


def has(paths, file):
    return any(os.path.basename(p) == file for p in paths)


mode = sys.argv[1] if len(sys.argv) > 1 else ""
A, C = "Discovery-Test-A-Q4_K_M.gguf", "Discovery-Test-B-Q8_0.gguf"
try:
    if mode == "started":
        ms = names()
        check("a freshly started Rapidli lists the files that were put there while it was off or running", has(ms, A) and has(ms, C), [os.path.basename(p) for p in ms if "Discovery" in p])
    else:
        shutil.rmtree(KEEP, ignore_errors=True)
        os.makedirs(KEEP)
        before = names()
        gguf(os.path.join(KEEP, A), "Discovery test A")
        r = post("/api/models/folders", {"add": KEEP})
        check("a folder added in Settings is looked through at once", has(names(), A), r.get("models"))
        gguf(os.path.join(KEEP, C), "Discovery test B")
        check("the list is not searched again on every call", not has(names(), C))
        ms = get("/api/models?refresh=1")
        check("a refresh finds the file that appeared", has([m["path"] for m in ms["models"]], C), [m["label"] for m in ms["models"] if "Discovery" in m["path"]])
        place = [p for p in ms["places"] if KEEP in json.dumps(p).replace("\\\\", "\\") or p.get("source") == "your folder"]
        check("the place is reported with what was found in it", any(p.get("found") == 2 for p in place), place[:2])
        check("the models that were there are still there", all(p in [m["path"] for m in ms["models"]] for p in before), len(before))
finally:
    if mode != "leave":
        post("/api/models/folders", {"remove": KEEP})
        shutil.rmtree(KEEP, ignore_errors=True)
        gone = not has([m["path"] for m in get("/api/models?refresh=1")["models"]], A)
        check("the folder taken out again: its models leave the list", gone)
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
