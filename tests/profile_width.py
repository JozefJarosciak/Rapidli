"""How long one pass takes with N lanes writing (version 2 engine): N branches forked from one short prompt.
  python v2/profile_width.py [port=11832] [n=1,2,4,8,16,24,32,48,64]"""
import json, sys, time, urllib.request

kv = dict(a.split("=", 1) for a in sys.argv[1:])
PORT = int(kv.get("port", 11832))
U = f"http://127.0.0.1:{PORT}"


def stats():
    return json.loads(urllib.request.urlopen(U + "/stats").read())


def fork(body):
    r = urllib.request.Request(U + "/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    fin = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                m = json.loads(raw[5:])
                if "done" in m:
                    fin = m
    return fin


prompt = "<|im_start|>user\nWrite a long, detailed life story of an invented person.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThis is the life story of"
first = ["Anna", "Boris", "Clara", "David", "Elena", "Felix", "Greta", "Hugo", "Irene", "Jonas", "Karla", "Leon", "Marta", "Nils", "Olga", "Pavel"]
last = ["Novak", "Berg", "Silva", "Kowal", "Marin", "Dubois", "Rossi", "Lund"]
lanes = stats()["lanes"]
print("lanes  ms/pass  tokens/s  (120 tokens per lane, no early ends)")
for n in [int(x) for x in kv.get("n", "1,2,4,8,16,24,32,48,64,80,96,112,128").split(",")]:
    if n > lanes:
        break
    for rep in range(2):                                   # the first run of a width records its graph
        a = stats(); t0 = time.time()
        fin = fork({"prompt": prompt, "branches": [" " + first[k % 16] + " " + last[k // 16] + "," for k in range(n)], "n_predict": 120})
        dt = time.time() - t0; b = stats()
    passes = b["passes"] - a["passes"]
    ms = (b["mean_pass_ms"] * b["passes"] - a["mean_pass_ms"] * a["passes"]) / max(1, passes)
    toks = sum(d["tokens_predicted"] for d in fin["done"])
    print(f"{n:5d}  {ms:7.1f}  {toks / dt:8.0f}   ({passes} passes, {toks} tokens, {dt:.2f} s, read {fin['timings']['read_ms']:.0f} ms)")
