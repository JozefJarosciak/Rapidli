# Building the engine

The engine is `engine/bin/rapidli-engine` (`.exe` on Windows) with its libraries: a patched `llama` library, `ggml`
with its CUDA backend, and `mtmd` for vision. The built files are not in the repository. Build them, or take them from
a release.

Everything is built from one source: llama.cpp in Unsloth's fork, `unslothai/llama.cpp` at commit
`c2cf5f983a01bc2ea7a5d68a7ce4fb171e6fe335`. Unsloth publishes that source as a release download only
([`llama.cpp-source-commit-c2cf5f9….tar.gz`](https://github.com/unslothai/llama.cpp/releases/tag/b12347-mix-241a0f3),
sha256 `bc280e70bba1a486b8f294ce1b0b4a255d67efbd9fb3db2e58dba920d94d5be0`): the commit is not in the fork's git
history, so a `git checkout` of it fails. The engine uses functions that only this fork has (`llama_batch_ext_*`,
the MTP drafting context) and two that Rapidli's patches add (`llama_fq_fork`, `llama_fq_set_sampling`), so mainline
llama.cpp will not build it as it is.

## Linux

```
engine/build-linux.sh
```

It downloads that source (and checks its sha256), applies the patches in `engine/llama` and `engine/cuda`, builds llama.cpp with CUDA as shared
libraries (only `llama`, `ggml` and `mtmd`), compiles `engine.cpp` and puts everything into `engine/bin`. Needs curl,
cmake, a C++17 compiler, python3 and the CUDA toolkit (`nvcc` on the PATH or in `/usr/local/cuda/bin`; 12.8 or
newer for an RTX 50 card or gcc 14). Without `CUDA_ARCH` it builds for the card in the computer; for a build that
other computers will use, set `CUDA_ARCH="75;86;89;120"` (RTX 20 to RTX 50). The CUDA files
take most of the time: about 15 minutes on 20 cores.

`LLAMA_SRC=/path/to/tree` uses a source tree you already have instead of cloning. The script keeps the tree's original
files in `.orig`, patches a fresh copy of them at every run and copies over only the files that changed, so it can be
run again after a change to a patch without compiling everything once more.

Run on 19 Nov 2026 under Ubuntu 22.04 in WSL 2 with g++ 11.4, CMake 3.22 and CUDA 12.6 on an RTX 4090: the engine
built, `start.sh` loaded Ternary-Bonsai-2-27B Q2_0 in 24 s and fitted its lanes, and `tests/lane_speed.py` measured
70 tokens a second on one lane and 1,231 with 32 lanes. A model file on a Windows drive (`/mnt/c/...`) loads very
slowly from WSL (a 16 GB model took more than four minutes): keep models on the Linux disk.

## Windows

Most Windows users do not build: the release download has `engine\bin` ready. Building it needs Visual Studio 2022
Build Tools (x64), the CUDA 13 toolkit, CMake and Ninja, Python with the `py` launcher and Git for Windows (for `sh`).
The scripts take llama.cpp's headers, vendor code and prebuilt `ggml` from an Unsloth Studio install whose llama.cpp
is the source above (its `UNSLOTH_PREBUILT_INFO.json` says `"source_commit": "c2cf5f98…"`, release
`b12347-mix-241a0f3`); `LLAMA_DIR` names it (default `%USERPROFILE%\.unsloth\llama.cpp`). With another release of
llama.cpp there, the patches stop at their first text check. Nothing in that folder is changed.

```
sh engine/llama/rebuild.sh        a fresh copy of the fork's src, every patch in name order, build.bat,
                                  and the result installed as engine\bin\llama.dll
engine\llama\make_mtmd_lib.bat    the import library for mtmd.dll (vision) from its exports
engine\src\build.bat              compiles engine.cpp and installs rapidli-engine.exe into engine\bin
                                  ("ENGINE BUILD OK")
```

`rebuild.sh` was run on 19 Nov 2026 from the repository's files alone (a copy with no `lib`, `obj`, `bin` or `src`
folder): BUILD OK, a `llama.dll` of 3,316,736 bytes that exports `llama_fq_fork` and `llama_fq_set_sampling`. Two runs
give the same code byte for byte, so a change to a patch can be checked by comparing the libraries.

`ggml.dll`, `ggml-base.dll`, the `ggml-cpu-*.dll` variants and `mtmd.dll` in `engine\bin` are the fork's own build.
The release's `ggml-cuda.dll` was built as below with CUDA 13.0 for 75;86;89;120 (about 5 minutes on 24 threads).
`ggml-cuda.dll` is not: it is built from the fork's source with the patches in `engine/cuda`:

```
rem the source above, downloaded and checked (Windows' own tar.exe; Git's tar reads "C:" as a host name)
curl -fLo llama-src.tar.gz https://github.com/unslothai/llama.cpp/releases/download/b12347-mix-241a0f3/llama.cpp-source-commit-c2cf5f983a01bc2ea7a5d68a7ce4fb171e6fe335.tar.gz
certutil -hashfile llama-src.tar.gz SHA256          must print bc280e70bba1a486b8f294ce1b0b4a255d67efbd9fb3db2e58dba920d94d5be0
mkdir C:\src\llama-src && %SystemRoot%\System32\tar.exe -xzf llama-src.tar.gz -C C:\src\llama-src --strip-components=1
set LLAMA_SRC=C:\src\llama-src
for %p in (engine\cuda\patch_*.py) do py -X utf8 %p     every patch once, in name order, on the fresh source
engine\cuda\build_cuda.bat configure  CUDA_ARCH chooses the cards to build for (default 75;86;89;120: RTX 20 to RTX 50)
```

`build_cuda.bat` finds CMake on the PATH or in Visual Studio (its "C++ CMake tools"), and needs Ninja on the PATH.

Then copy `ggml-cuda.dll` from `engine\cuda\out\bin`, and `ggml.dll`, `ggml-base.dll`, the `ggml-cpu-*.dll` files
and `mtmd.dll` from the Unsloth Studio install's `build\bin\Release`, into `engine\bin`.

The CUDA runtime libraries (cuBLAS) are not copied into `engine\bin`: the engine finds them through `CUDA_PATH`, which
the CUDA installer sets.

## The patches

| Patch (`engine/llama`) | What it does |
|---|---|
| `patch_direct.py` | each lane's recurrent (gated-delta-net) state is read straight from the state cache instead of being gathered and scattered every pass |
| `patch_f16.py` | the recurrent state kept in 16-bit floats (half the bytes per lane) |
| `patch_i8.py` | the recurrent state kept as 8-bit codes (`LLAMA_RS_I8=1`; more lanes) |
| `patch_greedy.py` | one sampling operation for all lanes instead of a sampler chain per lane (`LLAMA_FAST_GREEDY=1`) |
| `patch_pin.py` | a new sequence takes the state cell with its own number (`LLAMA_RS_DIRECT=1`) |
| `patch_passlog.py` | a log line per pass (`LLAMA_PASS_LOG=<file>`), for finding where time goes |
| `patch_zfork.py` | `llama_fq_fork`: one sequence's state copied to many lanes (cell tags + recurrent rows); falls back to `llama_memory_seq_cp` for other memory types |
| `patch_zspec.py` | the 8-bit in-place state becomes a property of a context: the lanes keep it, the drafting context (with rollback snapshots) keeps a full or 16-bit state (`LLAMA_RS_SPEC_F16`) |
| `patch_zzconv.py` | qwen35: the convolution of a gated-delta-net layer comes right after its input in the graph, before the state is stored, so that ggml-cuda can fuse the three (same operations, another order) |
| `patch_zzidx.py` | `llama_fq_fork` also copies the indexer cache of a memory that has one (`llama_memory_hybrid_idx`: Qwen3.8-Flash-Next), so lanes forked from a parked prompt keep their pooled keys |
| `patch_zzmoelog.py` | the MoE cache's figures while it runs (`LLAMA_MOE_CACHE_LOG=<n>`), and `LLAMA_MOE_CACHE_MAX_BATCH`: passes of up to that many tokens use the cache (llama.cpp: 32) |
| `patch_zzi8more.py` | the 8-bit recurrent state (`patch_i8.py`) also for qwen35moe (Qwen3.6/3.8-35B-A3B), qwen4exp (Qwen3.8-Flash-Next) and qwen3next |
| `patch_zzmoepin.py` | the MoE cache's host experts are registered as pinned memory once per process (read only; `LLAMA_MOE_PIN=0` turns it off): misses are uploaded at ~23 instead of ~10 GB/s |

| Patch (`engine/cuda`) | What it does |
|---|---|
| `patch_convfuse.py` | the gated-delta-net convolution, its silu and the store of its state as one kernel that never writes the joined input (same numbers; `GGML_CUDA_CONV_FUSE=0`: one by one) |
| `patch_f16.py` | the gated-delta-net kernel reads and writes a 16-bit state |
| `patch_zmmidslice.py` | a mixture of experts' products (MUL_MAT_ID) of 9 to 64 tokens on the vector kernel, 8 tokens at a time, when the tokens rarely share an expert (`GGML_CUDA_MMID_SLICE=0`: as before) |
| `patch_fwhtsigns.py` | the sign flip before a Hadamard-rotated matrix multiplication done inside the Hadamard kernel (same numbers; `GGML_CUDA_FWHT_SIGNS=0`: two kernels) |
| `patch_zfwhtq8.py` | the Hadamard kernel also writes its output as the q8_1 blocks the vector kernel reads, which then skips its own quantize kernel (same bytes; `GGML_CUDA_FWHT_Q8=0`: off) |
| `patch_zmmvqrows.py` | the vector kernel takes 4 rows of a Q2_0 matrix per block for 2 to 8 tokens (the same sums; 8 lanes 17 -> 15 ms) |
| `patch_i8.py` | the state stored as 8-bit codes and updated in place (experimental: it changes the answers) |
| `patch_memset.py` | the state of a starting lane is cleared in one go instead of with a wait per tensor (`GGML_CUDA_LAZY_MEMSET`) |
| `patch_mmq_wide.py` | matrix tiles of up to 160 columns for Q2_0, so a pass of up to 160 lanes unpacks each weight tile once (`GGML_CUDA_MMQ_NARROW=1`: at most 128) |
| `patch_mmvq_switch.py` | `GGML_CUDA_MMVQ_Q2_0=<n>`: up to which batch Q2_0 uses the vector kernel (default 8, as measured best) |
| `patch_optime.py` | a timer per operation (`GGML_CUDA_OPTIME`), for finding where a pass goes |
| `patch_sample.py` | one fused sampling kernel for all lanes' rows of logits |

## A release

A release download holds `engine/bin` for one platform. Put llama.cpp's `LICENSE` and `THIRD_PARTY_LICENSES.txt` next
to the binaries (see `THIRD_PARTY_NOTICES.md`). Pack it from a list, not the whole folder (which collects backups):
Windows: `rapidli-engine.exe`, `llama.dll`, `mtmd.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu-*.dll`,
`ggml-cuda.dll`; Linux: `rapidli-engine` and the `lib*.so*` files. Leave out the tools the engine does not use
(`llama-server`, `llama-bench`, `llama-quantize`, the `*-impl.dll` files) and any `*.bak*`.
