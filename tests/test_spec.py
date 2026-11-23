"""The /spec route of the version 2 engine (answers drafted by the model's own MTP head): speed against the ordinary
one-lane answer of the same engine, and how much of the text is the same.
  python v2/test_spec.py [port=11832] [n=300]"""
import json, sys, time, urllib.request

kv = dict(a.split("=", 1) for a in sys.argv[1:])
U = f"http://127.0.0.1:{int(kv.get('port', 11832))}"
N = int(kv.get("n", 300))
QS = ["Explain how vaccines train the immune system, for an interested adult.",
      "Write a Python function that returns the n-th Fibonacci number, with a short explanation.",
      "What is the difference between TCP and UDP?",
      "A train leaves a station at 9:00 at 80 km/h. A second train leaves the same station at 10:00 in the same direction at 100 km/h. When does the second train catch up?"]


def stream(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0, first, fin = time.time(), None, None
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                continue
            m = json.loads(raw[5:])
            if m.get("rows") and first is None:
                first = time.time() - t0
            if "done" in m:
                fin = m
    return fin, time.time() - t0, first


def chat(q):
    return "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


rows = []
for q in QS:
    a, ta, fa = stream("/fork", {"prompt": chat(q)[:-2], "branches": ["\n\n"], "n_predict": N, "temperature": 0})
    b, tb, fb = stream("/spec", {"prompt": chat(q), "n_predict": N})
    da, db = a["done"][0], b["done"][0]
    same = 0
    while same < min(len(da["content"]), len(db["content"])) and da["content"][same] == db["content"][same]:
        same += 1
    print(f"ordinary {da['tokens_predicted']:4d} tok {ta:5.2f} s {da['tokens_predicted'] / ta:5.1f} tok/s | drafted {db['tokens_predicted']:4d} tok {tb:5.2f} s "
          f"{db['tokens_predicted'] / tb:5.1f} tok/s, {db.get('drafts_kept', 0)}/{db.get('drafted', 0)} drafts kept | same first {same} of "
          f"{min(len(da['content']), len(db['content']))} chars | {q[:40]}")
    rows.append({"question": q, "ordinary": da, "drafted": db, "t_ordinary": ta, "t_drafted": tb, "same_chars": same})
json.dump(rows, open(__file__.replace("test_spec.py", "out/test_spec.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
