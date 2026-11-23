"""Model discovery (app/models.py): a model in an Ollama-style store is found, with the name its manifest gives it.
The store is a temporary folder next to a real model (a hard link, nothing is copied); it is removed afterwards.

    python tests/test_discovery.py"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import models      # noqa: E402


def main():
    found = models.scan()
    print(len(found), "chat models found;", ", ".join(f"{p['found']} in {p['source']}" for p in models.LAST["places"] if p["found"]))
    if not found:
        print("no model on this computer to test with"); return 0
    src = os.path.realpath(min(found, key=lambda m: m["bytes"])["path"])
    base = os.path.join(os.path.dirname(src), "_rapidli_ollama_test")
    shutil.rmtree(base, ignore_errors=True)
    man = os.path.join(base, "manifests", "registry.ollama.ai", "library", "testmodel")
    os.makedirs(os.path.join(base, "blobs")); os.makedirs(man)
    ok = False
    try:
        os.link(src, os.path.join(base, "blobs", "sha256-abc123"))
        json.dump({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123"},
                              {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:zzz"}]}, open(os.path.join(man, "4b-q4_k_m"), "w"))
        os.environ["OLLAMA_MODELS"] = base
        real, models.places = models.places, lambda: []       # the store alone (the same file elsewhere is listed once)
        try:
            ms = models.scan()
        finally:
            models.places = real
        got = [(m["source"], m["id"], m.get("quant"), m.get("arch")) for m in ms]
        print("from the Ollama-style store:", got)
        ok = len(got) == 1 and got[0][0] == "Ollama" and got[0][1] == "ollama/testmodel:testmodel-4b-q4_k_m" and got[0][2] == "Q4_K_M" and got[0][3] not in ("?", None)
    finally:
        os.environ.pop("OLLAMA_MODELS", None)
        shutil.rmtree(base, ignore_errors=True)
        models.scan()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
