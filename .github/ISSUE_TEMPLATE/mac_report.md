---
name: Mac report
about: Results from a Mac (the two checks in the README's macOS section)
labels: macos
---

**Your Mac**
- Model and chip (e.g. MacBook Pro, M3 Max):
- Memory (GB):
- macOS version:

**1. The app's tests**: the output of `sh tests/run_offline.sh`

```
(paste here)
```

**2. Many at once**: the model file you used, and the output of
`llama-batched-bench -m /path/to/model.gguf -c 32768 -npp 256 -ntg 128 -npl 1,8,32,64`

```
(paste here)
```

**Anything else** (errors, how warm or loud it got, anything that surprised you)
