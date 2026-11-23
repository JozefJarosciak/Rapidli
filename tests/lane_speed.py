"""How fast the loaded model writes with 1, 8 and 32 lanes at once (straight to the engine; no chat is made).

    py tests\\lane_speed.py [tokens]"""
import json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import fan as F       # noqa: E402
import models         # noqa: E402

E = "http://127.0.0.1:11832"


def post(body):
    r = urllib.request.Request(E + "/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    fin = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                m = json.loads(raw[5:])
                if "done" in m:
                    fin = m
    return fin


st = json.loads(urllib.request.urlopen(E + "/stats", timeout=10).read())
F.set_template(*F.template_of(models.header(st["model"]).get("tokenizer.chat_template")))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
prompt = F.chat_head("You are a helpful assistant.") + "Write a long, detailed description of a walk through a forest in autumn." + F.CHAT_TAIL
print(f"{os.path.basename(st['model'])}: {st['lanes']} lanes loaded, spec={st.get('spec')}")
for k in (1, 8, 32):
    t = time.time()
    fin = post({"prompt": prompt, "branches": [{"text": "\n\n", "seed": i, "temperature": 0.7} for i in range(k)], "n_predict": n, "temperature": 0.7})
    dt = time.time() - t
    toks = [d.get("tokens_predicted", 0) for d in fin["done"]]
    print(f"  {k:3d} lanes: {sum(toks):5d} tokens in {dt:5.1f} s -> {max(toks) / dt:5.1f} tok/s per lane, {sum(toks) / dt:6.0f} tok/s together")
