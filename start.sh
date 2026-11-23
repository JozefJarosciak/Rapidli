#!/usr/bin/env sh
# Starts Rapidli (Rapid Local Inference) at http://127.0.0.1:11842 in the background; its log is data/logs/rapidli.log.
# It loads the model used last with as many lanes as the graphics card holds. stop.sh stops Rapidli and its engine.
#   ./start.sh            start, and open the page in the browser
#   ./start.sh --quiet    start without opening the browser
# RAPIDLI_PYTHON chooses the Python (default: python3).
cd "$(dirname "$0")" || exit 1
PY="${RAPIDLI_PYTHON:-python3}"
URL=http://127.0.0.1:11842
open_page() { [ "$1" = "--quiet" ] || (xdg-open "$URL" >/dev/null 2>&1 || open "$URL" >/dev/null 2>&1 || true); }
if "$PY" -c "import urllib.request; urllib.request.urlopen('$URL/api/status', timeout=2)" >/dev/null 2>&1; then
  echo "Rapidli is already running: $URL"
  open_page "$1"
  exit 0
fi
if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
  echo "Rapidli needs Python 3.10 or newer as '$PY' (install it, or set RAPIDLI_PYTHON to the python to use)."
  exit 1
fi
if [ ! -x engine/bin/rapidli-engine ]; then
  echo "The engine is not installed: unpack rapidli-engine-linux-x64.tar.gz from the Releases page into engine/bin,"
  echo "or build it with engine/build-linux.sh (see README.md and docs/BUILDING.md)."
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || echo "Note: nvidia-smi was not found; Rapidli needs an NVIDIA card and its driver."
mkdir -p data/logs
nohup "$PY" -X utf8 app/rapidli.py --start >data/logs/rapidli.log 2>data/logs/rapidli.err.log &
echo "Rapidli is starting at $URL (the model takes 20 to 60 seconds to load)."
sleep 2
open_page "$1"
