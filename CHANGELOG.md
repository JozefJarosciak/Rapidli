# Changelog

## API for programs that send many related requests (24 Nov 2026)

For live copilots, agents and document tools (docs/API.md › Many related requests; README › The API):

- **Plain mode** (`"mode": "plain"` on `/v1/chat/completions`): the messages in the model's own chat format (ChatML,
  Gemma 4, Inkling), the whole system prompt uncut as a system turn at the front, one lane per request streamed token
  by token, no planner, notes, look-ups or layout. `max_tokens`, `temperature`, `top_p`, `min_p`, `stop`, `seed`;
  a reply longer than a lane goes on from the lane's parked state; a prompt longer than a lane is read into a parked
  state first; `finish_reason` (`length` when cut) and real token counts; failures as an error event; `image_url`
  parts read when the model sees images; not kept as a chat unless `"store": true`. Plain requests are not held back
  by `parallel_max` (their own limit: `api.plain_max`, none by default). A model load stops them and waits for them.
- **Contexts** (`/api/v1/contexts`, `/append`, `/fork`, `GET`, `DELETE`): text read once into a parked state; a
  request with `"context"` starts from a copy and reads only its new messages, leaving the context unchanged. The last
  message stays open, so appended text goes into it; an append makes a new state and running requests keep the old
  one. The app's clean-up (`Fan.free_slots`, `heal`) never releases a context's state (`fan.PINNED`); an expired,
  deleted or reloaded-away context answers 404, a full one 413, one too many 429. `app/plain.py`.
- **Priority** (`"priority": "high" | "normal" | "low"`): the engine lets waiting requests in by priority, and of the
  long prompts being read it reads the highest priority's first one to its end before the next, instead of in turns
  (four 2,400-token prompts sent together had all started writing at about 2.8 s; one alone starts at 0.7 s).
- **Batches** (`POST /api/v1/batch`): many replies from one context on one connection of server-sent events
  (`{id, delta}`, `{id, done, finish_reason, usage}`), each cancellable by its id (`/api/v1/batch/{batch}/cancel`).
- **Load options** (`POST /api/v1/models/load`): `lanes` (a maximum), `context`, `margin_mb`, `vision`, `drafting`
  for that load only, kept by refits; a fit with a lane maximum is kept apart (`fit.fit_key`).
- **Status**: `GET /api/v1/status` adds free and busy lanes, queued requests, the context limit and the contexts'
  use, free places for parked states, vision, the chat format, top_p, the card's memory and a load's progress;
  `GET /api/v1/status/events` sends it whenever it changes.
- **JSON replies** (`response_format`: `json_object`, `json_schema`): the reply is opened with `{`, ends when the
  object closes, and is checked (JSON, then the schema); written again twice at most, then 422.
- Engine (`engine.cpp`, version 5): `priority`, `tag` and `POST /cancel {"tag"}`; a cancelled request leaves its lane
  at the next pass also while its prompt is still read (it was noticed only after its next written token), and one
  cancelled while it waited leaves the queue; `"stream": true` on `/completion`; `top_p` through
  `llama_fq_set_sampling_p` when `llama.dll` has it (new `engine/llama/patch_ztopp.py` and
  `engine/cuda/patch_ztopp.py`: top-p in the fused sampling kernel); text that continues a parked state no longer
  starts with a second start-of-text token (Gemma); `/stats` adds `busy`, `queued`, `top_p`.
- Settings › API shows plain mode and contexts with an example. `tests/test_plain_api.py` (offline, a fake engine)
  covers the routes; it is in `tests/run_offline`.

## 0.9.1: only the driver to install, fixes from a fresh install (24 Nov 2026)

- The engine downloads carry the CUDA libraries they use, so the CUDA Toolkit (about 3 GB) is no longer needed: only
  the NVIDIA driver (580 or newer on Windows, 525 or newer on Linux). Windows: cuBLAS 13.0 (`cublas64_13.dll`,
  `cublasLt64_13.dll`); Linux: the CUDA 12.6 runtime and cuBLAS (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`), with NVIDIA's EULA as `LICENSE.cuda.txt`. Checked: with no toolkit on the path, the engine
  loads them from its own folder. The engine itself is the 0.9.0 build.
- Stopping Rapidli while a model loaded (stop.sh / stop.bat, or the app's shutdown) left the engine running and
  holding the graphics card: the app counted the engine as its own only once the load had finished. It does so from
  the start of the load now.
- README: the Linux install creates `engine/bin` before unpacking the engine into it (a fresh clone has no such
  folder, and `tar -C` stopped); Inkling-2-Mini is in the measured models; the first start loads a model by itself.

## Settings: larger, and the models as a table (24 Nov 2026)

- Settings opens over nearly the whole window (up to 1,680 px wide, 96% of its height) instead of a 1,040 x 780 box.
- Settings > Models is one box: a toolbar (find a model by name, quant or architecture; show only those that see
  images, draft with MTP, are mixtures of experts or fit the card; the count and total size; Look for new downloads;
  Download a model) over a table that sorts by any column - model, abilities, quant, size, your card, context,
  layers, source - and keeps the order and filters between visits. The loaded model is always the first row, marked,
  with its lanes and context and its own Fit again and Unload; the separate "Unload the model" button is gone. Where a
  large model would go (card, PC memory, disk) is the cell's tooltip. The folders Rapidli looks in are a box of their
  own below, the empty ones dimmed.
- Models and Get models are one tab: Settings > Models has "On this PC" and "Get models" under its heading, and a
  download going on shows as one line on "On this PC". What opened the old tab (the model menu's "Models and
  downloads…") opens Get models there.
- Get models, laid out again: a search bar; labelled filters - Size, Abilities (sees images, MTP drafting, mixture of
  experts), Fits my card, Publisher (a list instead of a row of chips) - with Clear filters; each result with its
  parameters, about how large it is at Q4_K_M and whether that fits, and its abilities; the model opened beside them,
  both scrolling on their own; the downloads and the download settings (folder, token) in boxes below. The filters
  are kept between visits.
- Get models filters by what Hugging Face read from each repository's GGUF (its parameters, architecture and context,
  asked with the search): "Inkling-2-Mini" says no size in its name and had been left out by Size and Fits my card.
  MTP is "MTP" when the name says it (or a tag does, on an architecture with MTP heads) and "MTP?" for an
  architecture whose files usually keep the head; Inkling 2 has none (its "mtp" tag is about a DSpark drafter).
  Opening a model reads one file's header (a few MB, kept in `data/hf-abilities.json`) and says for sure: MTP drafting
  or no MTP head, mixture of experts, sees images, the context. A repository with both kinds of file is read from an
  MTP one, and its MTP files are tagged. The largest file that fits the card is marked "best fit".
- Fixed: the list changed by itself. Each redraw opened its first result, whose file then decided the MTP filter -
  with MTP drafting on, clicking Fits my card and back left none of the earlier results. Now only a new search opens
  a result, and what a file says marks its row without reordering or dropping rows. The file read for a repository's
  abilities is never its drafter (the smallest file of Inkling repositories, which said "reads up to 1024k · dflash").
- Drafters beside a model (`dspark-…gguf`, architecture dflash) are no longer listed as chat models on this PC, and in
  Get models they are tagged "drafter".

## Better Long articles: written without offline Wikipedia's text (24 Nov 2026)

- A Long answer that explains something (an article, how something works, why) is no longer handed offline
  Wikipedia's text. Its writers paraphrased the Simple English articles ("stitched-together encyclopedia
  definitions", a reviewer wrote) and the outline shrank to what those articles covered: a history of aviation planned
  from them stopped at the Wright brothers. Blind review, 15 Long articles a version (Inkling-2-Mini Q2_K_XL,
  Qwen3.8-35B-A3B, Ternary-Bonsai-2-27B; 5 topics), scores out of 10, two separate rounds of reviewers:

  | | Before | Now |
  |---|---|---|
  | All three models | 3.1 / 3.2 | 5.5 / 5.9 |
  | Inkling-2-Mini | 2.0 / 2.1 | 6.0 / 6.1 |
  | Qwen3.8-35B-A3B | 3.6 / 4.0 | 6.0 / 6.8 |
  | Ternary-Bonsai-2-27B | 3.6 / 3.6 | 4.4 / 4.9 |

  Accuracy did not drop (5.7 to 5.8). The articles are also longer and written 2-4x faster, because every lane no
  longer reads a few thousand tokens of articles: Inkling 866 words in 22 s -> 1,719 in 11 s, Qwen 1,250 words in
  5.6 s -> 2,017 in 4.2 s, Bonsai 27B 931 in 7.8 s -> 1,314 in 5.8 s.
  A lookup (who, when, what happened), deep research, and the other lengths still use offline Wikipedia as before.
  `RAPIDLI_EXPLAIN_WIKI=1` restores the old behaviour; `RAPIDLI_EXPLAIN_WIKI_SIZES` names the lengths it covers
  (default `detailed`, the page's Long).
- A second pass for planned answers, `Fan.revise` (every paragraph written again by a lane that has read the whole
  draft), is in the code but off: it took 2-7 s and did not raise the score in the same review (`RAPIDLI_REVISE=1`
  switches it on; `tests/test_revise.py`). An outline rule asking for the subject's whole scope did not help either and
  was not kept.

## Inkling 2 at speed (24 Nov 2026)

- In all: Inkling-2-Mini UD-Q2_K_XL (the looped layout, `inkling2_loop`) on an RTX 4090 loads with 58-59 lanes at a
  32k conversation context instead of 21, its lanes write 1,340 tok/s together instead of ~230, and long planned
  answers come in at 150-185 tok/s (an article) and 250-305 tok/s (an in-depth explanation) where the latter had run at
  ~84. UD-Q4_K_M: 29 lanes instead of 12, a long answer at 230 tok/s.
  Qwen and Bonsai models run as before. Details below.
- Inkling 2's lanes attend to each sequence's own cells (kv_rows) with the relative-position bias in a per-head mask
  (`engine/llama/patch_zzzrows.py`, `engine/cuda/patch_zrowsmask.py`). Before, every pass built that bias over the
  whole shared cache - every lane's cells - and the pass time grew with lanes x cells. Inkling-2-Mini UD-Q2_K_XL on an
  RTX 4090, all lanes writing one greedy answer each: 62 lanes 231 -> 888 tok/s (246 -> 63 ms a pass); UD-Q4_K_M,
  48 lanes 253 -> 811. A prompt read whose cells are out of order takes the same path: with a 32k conversation
  context the layouts' reads had timed out after 30 s, now 0.3-1.1 s.
- When the lanes write (one token each), Inkling's 3 query heads per K/V head are the 3 queries of one attention
  (`engine/llama/patch_zzzzgqa.py`): each K/V head's cells - the shared prompt among them - are read once for the three
  instead of once per head. 83 lanes: 1,259 -> 1,331 tok/s with a short shared prompt, 783 -> 1,006 with a 2,400-token
  one, which is what the lanes of a planned answer share.
- ggml-cuda: a concat of non-contiguous tensors with a short first dimension runs one thread per element
  (`engine/cuda/patch_zconcat.py`; it was a block per row with all but a few threads idle, 24% of an Inkling pass).
  Q2_K_XL 124 lanes 1,196 -> 1,497 tok/s, 62 lanes now 1,131. Qwen3.8-35B-A3B (96 lanes ~1,620) and
  Ternary-Bonsai-2-2B (128 lanes ~11,750) are unchanged.
- ggml-cuda: `ADD -> SCALE` and `TANH -> MUL` run as one kernel each (`engine/cuda/patch_zfuse2.py`; bit-identical
  results). A pass of one Inkling token is ~5,400 kernels of ~2 us, so their number sets a single lane's speed:
  12.0 -> 11.7 ms a token (1 lane), 22.5 -> 22.1 ms (8 lanes).
- ggml-cuda: an F32 matrix on 4-8 columns takes the matrix-vector kernel instead of the tiled one
  (`engine/cuda/patch_zmmvf32.py`). Inkling keeps its expert router and loop controls in F32, and with 8 lanes the
  tiled kernel read them at ~80 GB/s, 11% of the GPU time. 4 lanes 18.3 -> 16.5 ms a pass, 8 lanes 22.1 -> 20.8 ms -
  where the passes of a long answer mostly are (5-17 lanes writing on average). Qwen and Bonsai unchanged.
  `GGML_CUDA_MMVF_F32_MAX=3` restores the stock choice.
- Fitting: a looped model's attention cache counts each run of its looped blocks (Q2_K_XL: 24 blocks stored, 48 run);
  the parked conversations and the drafting context take at most 640 MB and 1 GB of cache (11 and 4k on Inkling,
  unchanged on Qwen); a start slower than the probe with nothing spilled goes back to the probe's conversation context
  instead of cutting lanes; the lanes noted are the ones the engine runs. Q2_K_XL: 21 lanes -> 55 (32k context) or
  116 (8k).
- An answer ended as stuck on a nearly full card (under 3 GB free) fits the lanes again at once; the watchdog did that
  only for answers still under way after 60 s, so the next answer hung the same way.

## Engine on llama.cpp b12631, Inkling 2 (23 Nov 2026)

- The engine is built on Unsloth's llama.cpp release b12631-mix-acc0843 (commit 59f6620) instead of b12347
  (c2cf5f98). It reads Inkling 2's looped layout: Unsloth's Inkling-2-Mini GGUFs stopped with "key not found in
  model: inkling2.d_rel" before. Three patches were adapted (docs/BUILDING.md); engine.cpp is unchanged. The headers
  in engine/llama are the new release's. Measured on an RTX 4090, three runs of `tests/lane_speed.py 800`:
  Qwen3.8-35B-A3B IQ3_M 1 / 8 / 32 lanes 175 / 818 / 1,625 -> 191 / 858 / ~1,700 tok/s; Ternary-Bonsai-2-27B Q2_0
  80 / 485 / 1,180 -> 85 / 500 / 1,250. The build scripts and docs name the new source and its checksums;
  build-linux.sh refuses a source tree of the old release.
- Inkling 2's chat format (`<|message_user|><|content_text|>…<|end_message|>`): its answers are written without
  thinking (effort 0); "Think per part" works as with any model. Inkling-2-Mini UD-Q4_K_M with its vision projector
  on a 24 GB card: 23 lanes, a short answer 1.5 s, Medium 11 s on 7 lanes, Longest 2,616 tokens in 21 s on 23 lanes
  (126 tok/s together).
- Fitting: the probe's attention cache is budgeted from the model's header (keys and values of every layer with
  attention). Where 8 lanes, the parked conversations, the drafting context and the conversation context would not fit
  beside the weights, the parked conversations are cut first (down to 8), then the drafting context (4k), then the
  conversation context (8k); a kept fit is started with the same cuts. Inkling-2-Mini's probe (80 parked conversations,
  6.5 GB of cache next to 15.5 GB) had filled the card and crawled.
- An API request whose answer stalled got an event line instead of an HTTP response; it gets a 504 with the API's
  error now (a streamed OpenAI-style one, an error chunk and `[DONE]`).
- Downloads: a .part without notes starts over (one of 2 GB, left from an older download, failed its checksum after
  12.7 GB).
- Not yet done on the new engine: Qwen3.8-Flash-Next (experts in the PC's memory), the Linux build, images with
  Inkling, and Inkling's other layout (Mainline_Support, `inkling2_loop`).

## API: models and answer options (23 Nov 2026)

- `GET /api/v1/models` lists every model on the PC; `POST /api/v1/models/load` (`model`: id or label, `wait`,
  `refit`) and `POST /api/v1/models/unload` load and unload them.
- `/api/v1/ask` and `/v1/chat/completions` take the composer's choices: `size` (also by the names the app shows),
  `think` (the model's levels and per-part thinking; `reasoning_effort` on the OpenAI route), `wiki`, `web`,
  `first` and `deep`. A wrong value is refused with the values that can be given; no model ready answers 503.
- `GET /api/v1/options` says what can be chosen now; `/api/v1/status` adds the label and the thinking levels.
- docs/API.md rewritten for other programs, with examples; Settings › API opens it in the app.
- No model loaded: the engine's port is tried every 2 s at most, not on every poll (the polls had piled up).
- Settings › Models: an unloaded model shows Load and Delete once the engine has stopped; the cards' text wraps.
- "Look for new downloads" says it is looking, then what it found (new models, ones no longer there, how many in all);
  the new ones are tagged. The model menu stays open for it.
- Downloads: a download stopped before its first notes were written was later kept whole, nearly all zeros (the .part
  is made full length first). The notes are now written at once, a full-length .part without notes starts over, and a
  model is checked against Hugging Face's sha256 (or at least its GGUF start) before it is kept. Unfinished downloads
  left when Rapidli stopped are listed in Get models with Resume and Delete; the checksum step is shown and can be
  cancelled. tests/test_fetch.py covers both.
- A finished download asked for a new look through the model folders every second while Get models was open (5-8 s
  each); "Look for new downloads" waited behind them. It is asked for once per finished download now, looks asked for
  while one runs share it, the menu shows the last result (not a "looking" menu opened late), it names the models still
  downloading, and a finished vision projector says it is not a model of its own.

## Eject and delete models (23 Nov 2026)

- Eject on the loaded model's row in the model menu unloads it and frees the graphics card.
- Delete in Settings > Models removes a model's files from the PC. The confirmation lists every file it removes and its
  size: all parts of a split model, the file behind a Hugging Face cache link, and the vision projector when no other
  model in the folder uses it. It warns when the files are in another app's folder. The loaded model, a model still
  downloading, and Ollama models (its store shares files between models) are refused.

## Pictures in answers (22 Nov 2026)

- A message that asks to see something ("show me the cover of this book", "what does X look like", "a photo of...")
  gets pictures from the web with Search on: found through ddgs's image search for the query the web search wrote
  (so "this book" is the book of the conversation), fetched by the app (only public addresses; the format told by
  the first bytes; the thumbnail when a site refuses the picture), kept in data/images and shown above the answer
  with their sites. The model is told what is shown: it had answered "I can't display images" with a description.
  Without Search, the answer says that Search shows pictures.

## Ready for a public release (22 Nov 2026)

- The app answers only to its own loopback names (127.0.0.1, localhost, [::1]): a web page renamed to 127.0.0.1 (DNS
  rebinding) is treated as another computer. Without an API key, another site's page cannot post questions. The
  engine refuses browser requests and other host names too.
- A page linked in a question, or a redirect, cannot reach this computer itself (the app, its engine, link-local
  addresses). Model downloads stay inside their folder; offline Wikipedia articles are shown in a sandbox, never run.
- Linux: `engine/build-linux.sh` downloads the pinned llama.cpp source (sha256-checked); the pinned commit was never in
  the fork's git history. Other programs on the card are now seen on Linux too; the page check finds Chromium there.
- The Windows CUDA build includes RTX 50 (compute 12.0) by default.
- The engine's OpenAI-style answers name the loaded model; `/spec` checks its request before taking its lock.
- Python 3.10 and 3.11 work again (three f-strings had a backslash inside their braces, which only 3.12 allows; the
  app did not start on Ubuntu 22.04's Python).
- `start.bat` and `start.sh` say what is missing: Python 3.10+, the engine (with where to get it), the NVIDIA driver.
- Release archives of the engine for Windows and Linux; `PORTABLE=1` builds the Linux engine for any x86-64 with AVX2.
  The Windows CUDA build configures from the top of the published source (it has no `ggml.pc.in`) and finds CMake in
  Visual Studio.
- Research notes, experiments and local test scripts are no longer part of the repository. New README with
  screenshots, install steps for Windows and Linux, hardware requirements and the measured results.

## Big mixtures of experts faster (22 Nov 2026)

- The experts of a model kept in the PC's memory are pinned, so the card copies the ones it misses at ~23 GB/s instead
  of ~10. Qwen3.8-Flash-Next (62 GB) on one RTX 4090: a Longest article in 40 s instead of 55 (195-206 tok/s while the
  lanes write, was 110-140), 300-448 tok/s on a page of 64 canvas effects. `LLAMA_MOE_PIN=0` turns it off.

## Code on many lanes (22 Nov 2026)

- A page or module of many functions is written by 34-66 lanes at once: a drafted lane lists the functions, every
  function is written in 2-5 copies side by side, each copy is checked (`node --check`, unknown calls) and run in a
  hidden browser (throws? draws? moves?), and the best one of each is kept. Drawing functions go into a gallery
  template (one canvas per card, one animation loop, each function walled off so one bad one cannot stop the page);
  games get keys and pointer input (each plays itself until its card is clicked). The finished page is tested, failing
  cards are swapped for other copies or written again, and the best page seen is kept. See `docs/MULTILANE-CODE.md`.
- Measured through the app (35B A3B, 91-114 lanes, four runs each): 32 canvas effects in 19-31 s, 31-32/32 (one lane:
  34 s, half the lines); 64 effects in 44-46 s, 63-64/64 (too long for one lane at 8k); a generative art gallery
  in 23-26 s (one lane 37 s); 10 physics simulations in 24-29 s (one lane: 55 s, did not run); 12 dashboard charts in
  18-24 s; 12 playable games in 33-37 s, 9-12 drawing; a page of 20 text functions in 9 s (one lane 14 s, a third of
  the lines); a 20-function Python module with doctests in 8 s. Peaks of 1,450-2,460 tok/s.
- While it works the answer says what it is doing: the functions' names at once, the drafts written, how many copies
  draw and move, what is written again.
- Lists with a number asked are numbered and go on from their own lines when the model starts repeating names; copies
  that took their label for an object or left out the function line are put right by their text; a card that makes
  more drawing calls in a frame than 0.43 per pixel of its canvas is stopped (a runaway effect, or one painting a full
  card pixel by pixel, had hung whole pages and their tests); numbered variants of a listed name count as repeats; a
  copy that stands still ranks lower when the request asks for animation; simulations are told to start out of
  balance; every function with no working copy is written again (up to 32); cards off the screen are not drawn; a game
  that stands still while nobody plays it is started again and played by a bot.
- A question about functions ("What are the 4 functions of a compiler?") stays a text answer.
- An engine whose process has ended (a CUDA error, twice in the night) is loaded again at once: answering again ~26 s
  after the crash instead of 50-70 s.
- "Expand on the third section only" is answered with the detail (it got "The third section is complete; no further
  expansion is needed." in 7 of 10 runs: the note for "continue" offered that way out); an answer to such a request
  that stops after a lead-in sentence goes on under a heading, whichever way it was written.
- The engine keeps 80 parked states (was 8), and the app reads that number from it.
- The watchdog reloads the model when the engine stops answering; a page's test gives up after 8 s (a hang is a
  failure); a fix prompt over the drafting route's 8k is not sent to it.
- A follow-up in English to a chat with Czech in it is answered in English.
- Regression probes (all): 131-132/138 turns pass (115-118/125 before this work). Every case that failed was run on the
  code from before this work (0334fe5) with the same model: rank_countries, doc_summary, adj_onelane, vocab_15 fail
  there too, py_tests and section_ref half the time (section_ref now passes 5/5); sk_code and adj_shorter passed 3/3
  on a rerun, cz_chain 2/3 (it failed the same way on 21 Nov). On the final code: smoke 15/15, code 17/17, fast 57/58 (doc_summary, as always).

## Unreleased (first public version)

**Fast local inference**
- An engine that writes many answers ("lanes") in one pass over the model: 156 lanes write about 2,400 tokens a second
  together on one RTX 4090 with Ternary-Bonsai-2-27B (one lane: about 90, eight: about 535). Its CUDA patches fuse
  the gated-delta-net convolution with its state store and the Hadamard sign flip and the vector kernel's input
  quantization with the transform (the same numbers, fewer passes over memory), give Q2_0 matrix tiles of up to 160
  columns, and let the vector kernel take 4 rows per block for 2 to 8 lanes.
- The lanes are fitted to the card at load: as many as fit, checked with every lane writing so that nothing spills into
  the computer's memory, with more left free next to programs that hold much of the card. A model that does not fit
  next to them stops with a message that names them.
- Long answers are planned once and written by many lanes at once: a Report or Longest answer gives every point a lane
  of its own (a bullet with its details as sub-bullets), Longest uses as many lanes as the card holds. Long articles
  are written at 1,700 to 1,900 tokens a second while the lanes write, with peaks of 2,000 to 2,170; a large model
  writes the outline beside the brief answer instead of after it.
- Stories are planned once (voice, cast, what each scene tells and where it stops) and every scene is written by a
  lane of its own: 5,890 words in 23 seconds. On models with an MTP head the plan is drafted, which takes a
  1,100-word story from 20 to 15 seconds.
- Short answers are drafted by the model's own MTP head where it has one.
- Several questions at once: side by side on the free lanes, or one after another with the place in line and the wait.
- While a question is typed, the engine keeps the card at speed (nothing shows), so a question asked after a pause
  starts as fast as one asked right after another: a one-paragraph definition in 0.98 s instead of 1.32.

**The app**
- Long articles: offline Wikipedia is used when it is about the subject (a broad request was turned down as "not
  answered" and written from memory); article lanes write at temperature 0.4 (looping sentences 52 -> 10 over 13
  articles, the same speed); the tidy-up takes out the plan's template, retold events, filler paragraphs, word-for-word
  repeats, citation-only lines and fragments, keeps a cited source passage the first time it is used, and keeps code
  blocks in place; an answer asked to be chronological has its lists in the order of their years. A question about
  how something works gets a Mermaid diagram of its main steps from one more lane, at no extra time.
- Mermaid diagrams are drawn in the answer (in a sandboxed frame), the code a click away.
- Read-ahead for a mixture of experts on any PC: the experts, and the n-gram table too when the memory holds both.
- Probes in tiers: smoke (under 2 minutes), code (code, diagrams, fixes), fast, all.
- Coding: Ask to fix quotes the lines the error points at (a Python traceback's line, or a page's line found in its
  inline script by the name in the error) and asks to keep what works; a Python error about network, files or modules
  says the code runs in Pyodide in the browser; an address inside code or an error is not read as a linked page. Code
  blocks have Changes (a line diff against the code's last version in the chat), and code that leaves parts out
  ("// ... rest unchanged") is flagged with a way to ask for all of it. A page built on request opens by itself.
- Qwen3.8-Flash-Next: its experts are read ahead before the engine starts (a cold load took nine minutes), its workout
  prompt is short (its 26.8 GB table is read from the disk), and its answers get four times longer before they count
  as stuck. "Continue" draws once a frame.
- Loading a mixture of experts kept in the PC's memory (Qwen3.8-Flash-Next) no longer fails on a slow PC: the speed
  checks are time-limited and give what they measured; the PC's memory is measured with the probe stopped; with too
  little of it the model gets 16 lanes; a load that fails stops its engine (one left running held 24 GB). The load
  says each step in a few words and, when the PC is short of memory for the experts, says so under the bar; the
  model menu says it before the load.
- The context menu lists only the sizes that fit on the card (and the one in use); its heading says how much fits.
- The Run frame is safe however it is opened: a CSP sandbox gives it an opaque origin, only the app may frame it, and
  it takes a page only from the app (any web site could open it as a window and run a page as the app). A page that
  reloads itself ("Play again") is sent again; a slow frame falls back to the in-page preview for that run only.
- Second review fixes: German and Dutch questions with "was", "will" or "an" are no longer taken for English; short
  Spanish and French questions are no longer taken for Slovak; "make a good snake game" builds a page; "Make a list
  ...", "Let's talk about ..." after a page or code are new questions, not edits; camel-case site names in result
  titles stay whole; Chart reads "8,849 m" as metres, "1,5" as 1.5 and "5 400 000" whole; a slow search engine has
  its health set when it answers late; "Prelož ... do angličtiny" is answered in English; a table cell repeated side
  by side is translated in every place.
- Every table in an answer has Copy (tab-separated: it pastes into a spreadsheet as cells); the Run pane has a full
  window toggle, and its Save keeps a program as its own code file.
- Run takes CSV: its rows as a table, with the chart of its figures above it.
- A table of figures in an answer has Chart: its numeric columns drawn in the Run pane (a line over years, else bars;
  "5.3 million", "$1.2bn" and "0.4%" read as numbers; a far smaller column on its own axis).
- Run takes SQL: the statements run on an empty SQLite database in the page (sql.js), each result as a table, an
  error in the console after what ran before it.
- A Run preview runs in a process of its own (loaded from the app's other loopback name, localhost or 127.0.0.1): a
  page that never stops working, such as a game's computer player thinking for seconds, no longer freezes the app.
  Pages are asked to keep such work short or in a Web Worker.
- Fixes from a review of these changes: a follow-up counts as a change to the code or page before only when it names
  them ("How to make sourdough bread?" after a program was taken for one); "a good website to learn Spanish" is not
  a page to build; the tidy-up leaves fenced code as it is (it took a closing fence and "=======" lines out) and does
  not cut parallel sentences with other figures or names as loops; an engine is not rested for a query nobody has
  results for; "the 3 main reasons" stays in an offline query; the engine's loop check reads only the lane's current
  request (20 tokens in other languages, 12 in English).
- Planned answers: an outline's introduction and conclusion parts are left out (the brief answer opens it and the
  bottom line closes it), and a short outline is topped up before its conclusion without repeats: a Slovak Longest
  article had nine "Záver: Záver" parts and 1,856 words; now 15 parts and 4,654 words.
- Search: at most four engine requests at once; videos and social networks take no page-reading place; a split
  model is listed by its name without the part number.
- While an answer is written its live card (speed, lanes, phases) stays at the top of the conversation; the engine
  panel fits the window without scrolling (from 700 px high), with all its figures.
- Run: an open preview runs the new version when a changed or fixed page, program, diagram or component arrives;
  a fix or change to code gets the room for the whole code again; Run and Save wait until the code is complete.
- Settings > Web search shows each search engine's state (answered when, resting and for how long). Engines that
  answered lately are asked first, up to three at once when the first do not answer, and a second query is sent
  only in another language: searches in 1-7 s where they took 12-22 s. The query writer knows today's date.
- Run takes React (JSX/TSX) and TypeScript too, without a build step (Babel and React from a CDN). The start screen
  offers "Build and run"; a chat's address opens it; a page without a title is named by its first heading.
- A long answer that cites sources lists them at its end; Copy and Export carry the list.
- Search answers in 1-2 s again when engines refuse: the next engine is asked after 1.5 s, the fastest first.
  Page markup (an article's "(picture)" links, wiki heading marks) is left out of what is read.
- Planned answers, measured on Longest articles: reworded duplicates are dropped
  (near-duplicate pairs 52 -> 5 in four articles, terms defined twice 10 -> 2); every paragraph that opens with
  It/This/They is rewritten to name its subject; a lane cut at its limit keeps only whole sentences; the plan's notes
  are stop strings; short 'Label: fact' paragraphs become a list; a question that names its parts gets sub-sections.
  The engine can end a lane caught in a loop (request option "loop"; needs the engine rebuilt).
- Run also takes Python (in the browser through Pyodide, loaded once from the internet: printed lines in the console,
  matplotlib charts on the page, the error as the program's own line) and Mermaid diagrams. Long code is shown folded
  with Show all, and a block worth keeping has Save (named from the page's title or a file comment).
- A citation shows, when pointed at, the passage the answer was written from, not only the page's title.
- The line under an answer says how many of the loaded lanes it used ("125 of 175 lanes"): a planned answer takes a
  lane per point, not all of them.
- Planned answers: the opening paragraph is never removed as a duplicate (a Longest article had begun at its first
  heading); a lane that writes out the plan's notes or copies the sources word for word has those paragraphs left
  out, and the lanes are asked to use the sources in their own words; sub-bullets that only say something is
  important, common or a best practice, with no figure or name, are left out.
- Search: each web engine is asked on its own, in turn (one query, one engine; the next only when it gave too
  little), and one that stops answering rests while the others go on: after about 20 quick searches every query
  had found nothing while two engines still answered. Results of several engines are merged by rank. What was
  found and read is kept for 30 minutes (an answer written again does not search again); Wikipedia that says "too
  many requests" is let rest. Pages with accented addresses are read (they failed).
- The passages of all the pages read are ranked together (BM25, the engine's snippet as a pointer, those at half
  the best score or more), from two more pages than before in the same room: on 32 questions the answer reached
  the model as often as before; offline Wikipedia 22 of 22 answers against 21, in less text.
- Offline Wikipedia searches the question without what it says about the answer's form ("About 200 words",
  "in 5 bullet points", "briefly"): "About Elly" was cited for a question on heat pumps, "State" for one on
  Michael Jackson. The last word of a title is searched alone only for a person's name, and must be in the title.
- Run: code an answer writes for the browser (an HTML page, a script, an SVG) runs beside the conversation, in a sandbox
  that can read and change nothing of the app (the server refuses requests from other pages), with its console, its
  errors and "Ask to fix", which puts them in the message box. A page in an answer gets a card under it that runs it;
  it opens in a new tab or saves as an .html file. "Make a snake game", "build a pomodoro timer" are written by one
  lane as one complete HTML file sized to the conversation's context (an 8k context: 246 lines, then the whole page
  again with a change asked for), and a change asked for after it gets the whole page again, or only the parts that
  change when the page no longer fits.
- Finds the GGUF models already on the computer (Hugging Face cache, LM Studio, Jan, GPT4All, llama.cpp,
  text-generation-webui, Ollama, folders of your own) at every start and on Refresh.
- Get models: search Hugging Face, filter by size, images, what fits the card and publisher, and download the file
  that fits, over several connections.
- Offline Wikipedia from Kiwix files, the Simple English edition offered with one click at the first start; web search
  when asked. Every search term an answer's research used is listed under the answer (offline and online, with the
  time), and all of them in Settings › Search log, which can be filtered and cleared.
- Long chats stay in view: long answers (and the web results kept with questions) get a short note in the background,
  and a chat longer than the context reads its older long texts as their notes instead of cutting them. Each answer's
  ⋯ menu keeps it whole, always uses its note, or shows the note; the context menu does it for all of them; Settings ›
  Chat turns notes off. The gauge shows what is read of the chosen context and how long the chat is.
- Sources show where they came from: offline Wikipedia or which engine online, and the search terms that found them.
  The search log also records the engines that were skipped or blocked, and why offline Wikipedia was not used.
- Thinking fits the context: its budget leaves room for the web results and the plan (a long article with "Think
  hard" and web search was refused by the engine before).
- "Think per part" (any model): for Medium and Long answers every part's lane gathers the facts of its own section side
  by side (a few seconds), and every writer reads all of them; other lengths think briefly on one lane.
- Offline Wikipedia is searched for the articles the model names (exact titles in the file's language, names spelled
  right), then for the names in the message; lone letters are no longer taken for names, and articles that only share
  a letter or a side subject are checked by the model before they are used.
- Answers that explain something (how it works, a guide, an article) state what is well established plainly and cite
  the sources for specifics; only questions that look something up (a person, a date, a figure, what is current) say
  what the sources did not cover. A planned answer is tidied after it is written: talk about the sources, the plan's
  own marks ("…: effect", "Example:"), paragraphs that are only a citation, stacked citations, a Wikipedia article's
  words about itself and paragraphs said twice are taken out.
- After a planned answer is written, a section that opens with "This" or "It" (pointing at another lane's words) gets
  its first sentence written again, and a conclusion about one corner of the answer is written again from the section
  titles (a lane each, in one pass, under half a second). A figure stated a third time is left out, a paragraph that
  stopped after a sentence is written once more, and a paragraph whose sentence plainly comes from one source is cited
  to it.
- Once a long answer's outline is written, the parts that the offline articles found do not cover get an article of
  their own (Routing, Packet switching, OSI model …), read by every writer and listed with the sources.
- A Medium article has 6 or 7 parts. With thinking on, the outline in the thought is not copied into the plan, and a
  plan with far fewer parts, or a part with fewer points, than the length asks for is completed by the lane that wrote
  it (Longest with thinking: 7,300-10,600 words instead of 1,400-3,500).
- A long article ("Long" and up) has sections with headings; "Long" was written like "Medium" before. With thinking on,
  the plan reads only the end of the thought, so a Longest answer keeps its length.
- Long answers are researched more widely: up to six offline articles (the subject and its main parts) and four web
  queries side by side, each read shorter. Each offline source names the title it was found by. Web search no longer
  waits for a slow query once another has results (4 seconds more), or for slow pages (5 seconds for all of them).
  An engine that keeps failing rests for a few minutes (DuckDuckGo 20 minutes after it asks for a captcha), so a
  throttled engine no longer costs every question its timeout.
- Models larger than the card: a mixture of experts (Qwen3.8-Flash-Next and others) keeps its experts in the PC's
  memory and only what every token needs on the card. The card keeps the experts used most in a cache that follows
  what is written, the processor helps on its fast cores only, and an n-gram table stays on the disk. Qwen3.8-Flash-Next
  Q2_0 (66 GB) writes one answer on an RTX 4090 at 51-56 tokens a second, and 32 lanes write 380-440 tokens a second
  together (the experts' products of up to 64 tokens run on the vector kernel in slices of 8: three times what 32
  lanes wrote before); a long article of 3,600 words takes under a minute. It runs best with about 35 GB of free RAM. Models and Get models show for every file what goes on the card,
  into the PC's memory and onto the disk (read from the file's header, also before downloading); the engine panel shows
  the card's load, heat and power and the PC's memory; Settings › Memory shows where the model is.
- Smaller mixtures of experts that fit the card (Qwen3.6/3.8-35B-A3B) keep their recurrent state in 8 bits like
  Qwen3.5: Qwen3.8-35B-A3B IQ3_M writes 207 tokens a second on one lane and 4,131 on 232 lanes together on an RTX 4090.
- One answer is drafted by copying when it repeats its prompt: a rewrite, code given back with changes, a summary that
  quotes its source. Any model can do it, also one without an MTP head (Ternary-Bonsai-2-27B: a rewrite 82 -> 178
  tokens a second, code 82 -> 196), and next to an MTP head it adds 3-10%. The engine measures what a check of every
  number of tokens costs and chooses, for every few tokens, the drafter that is expected to give the most.
- Images for models with a vision projector, attached files, thinking for models that think.
- Chats in folders, the Top 10 fastest answers (asked again with one click), the engine's panel live.
- An HTTP API for other programs, including an OpenAI-style chat route.
- Windows and Linux.
