#!/bin/sh
# Rebuild our llama.dll from Unsloth's pristine source plus the patches, and install it into ../bin (never into Unsloth's folder).
#   sh rebuild.sh            patch scripts: patch_*.py, applied in name order
set -e
cd "$(dirname "$0")"
LLAMA_DIR="${LLAMA_DIR:-$HOME/.unsloth/llama.cpp}"      # llama.cpp as Unsloth Studio installs it (read only)
rm -rf src && cp -r "$LLAMA_DIR/src" ./src
for p in patch_*.py; do "${PYTHON:-python}" -X utf8 "$p"; done
powershell -NoProfile -Command "cmd /c '$(cygpath -w "$PWD")\\build.bat' 2>&1 | Select-String -Pattern 'error|BUILD|warning C4' | Select-Object -Last 12"
test -f bin/llama.dll || { echo 'llama.dll was not built: see core.log, models.log, link.log'; exit 1; }
mkdir -p ../bin
cp bin/llama.dll ../bin/llama.dll
ls -l ../bin/llama.dll | awk '{print "installed llama.dll", $5, "bytes"}'
