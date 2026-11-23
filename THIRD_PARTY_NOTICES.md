# Third-party software

Rapidli's own code is under the MIT licence (see `LICENSE`). It builds on the work below.

## In this repository

| What | Where | Licence |
|---|---|---|
| llama.cpp, in Unsloth's fork (`unslothai/llama.cpp`, release `b12631-mix-acc0843`, commit `59f6620`): headers, and the `src/` copy that `engine/llama/rebuild.sh` patches | `engine/llama/include`, `engine/llama/ggml/include`, `engine/llama/gen`, `engine/llama/src` (generated, not committed) | MIT, Copyright (c) 2023-2026 The ggml authors |

The patches in `engine/llama/patch_*.py` and `engine/cuda/patch_*.py` are Rapidli's own (MIT) and apply to that
source.

## Compiled into the engine, or loaded by it

| What | Licence |
|---|---|
| llama.cpp, ggml and mtmd (vision), from the same fork | MIT |
| cpp-httplib (the engine's HTTP server), from llama.cpp's `vendor/` | MIT |
| nlohmann/json, from llama.cpp's `vendor/` | MIT |
| NVIDIA CUDA runtime (cuBLAS) | NVIDIA's CUDA EULA. Not distributed with Rapidli: it comes with the CUDA toolkit or runtime you install. |

A release that ships built binaries (`engine/bin`) must include llama.cpp's `LICENSE` and its
`THIRD_PARTY_LICENSES.txt` next to them.

## Optional Python packages (installed by you, not bundled)

| Package | Used for | Licence |
|---|---|---|
| `ddgs` | web search through several engines | MIT |
| `pypdf` | the text of attached PDFs | BSD-3-Clause |
| `libzim` | reading Kiwix `.zim` files (offline Wikipedia) | GPL-3.0-or-later |

Rapidli imports `libzim` only when it is installed and offline Wikipedia is used. If you redistribute Rapidli together
with `libzim`, the GPL's terms apply to that combination.

## Content and services

- **Wikipedia** editions (the Simple English one Rapidli offers at its first start, and the others in Settings ›
  Offline Wikipedia) are downloaded by the user from the Kiwix library (https://kiwix.org, library.kiwix.org). Their
  articles are written by Wikipedia's contributors and available under the Creative Commons Attribution-ShareAlike 4.0
  licence (https://creativecommons.org/licenses/by-sa/4.0/); each article's authors are listed in its history on
  Wikipedia. Rapidli's MIT licence does not apply to them, and none is in this repository.
- **Models** are downloaded by the user from Hugging Face under each model's own licence.
- The page uses the fonts installed on the computer; it loads none from other servers.
