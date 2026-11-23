#!/usr/bin/env sh
# The tests that need nothing running (no model, no app), on Linux or macOS: sh tests/run_offline.sh
# (on Windows: tests\run_offline.bat). Exits with the number of test files that failed.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-}"
if [ -z "$PY" ]; then                      # (python3, else python, else py: Git Bash on Windows has a python3 that is not one)
    for c in python3 python py; do
        if "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "Python 3.10 or newer was not found (set PYTHON=...)"; exit 1; }
failed=0
for t in test_rich_points test_together test_stuck test_sizing_notes test_hub_split test_fetch test_kiwix_names; do
    out=$(PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf8 "$PY" "tests/$t.py" 2>&1)
    last=$(printf '%s\n' "$out" | tail -n 1)
    case "$last" in
        "ALL PASS"|PASS*) echo "ok    $t" ;;
        *) echo "FAIL  $t: $last"; printf '%s\n' "$out" | grep -E '^FAIL' | head -n 5; failed=$((failed + 1)) ;;
    esac
done
echo "$failed test file(s) failed"
exit "$failed"
