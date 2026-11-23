#!/usr/bin/env sh
# Stops Rapidli and the engine it started. It asks Rapidli to shut down; no process is stopped by name.
# An engine started by hand (python3 app/eng.py start ...) is stopped with: python3 app/eng.py stop
cd "$(dirname "$0")" || exit 1
PY="${RAPIDLI_PYTHON:-python3}"
"$PY" - <<'EOF'
import time, urllib.request
A = "http://127.0.0.1:11842/api/"
try:
    urllib.request.urlopen(urllib.request.Request(A + "shutdown", data=b"{}", headers={"content-type": "application/json"}), timeout=20).read()
except Exception:
    print("Rapidli is not running.")
    raise SystemExit(0)
for _ in range(60):                       # the engine is stopped first: wait until the server itself is gone
    try:
        urllib.request.urlopen(A + "status", timeout=2).read()
        time.sleep(0.5)
    except Exception:
        time.sleep(1)                     # (the port closes a moment before the process ends)
        print("Rapidli and its engine are stopped.")
        raise SystemExit(0)
print("Rapidli was asked to stop and is still at it; look again in a moment.")
EOF
