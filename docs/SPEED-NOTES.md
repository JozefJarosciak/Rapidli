# Speed work, 19-20 Nov 2026 (from 21:39, seven hours)

Ternary-Bonsai-2-27B Q2_0 on one RTX 4090 (24 GB), other programs holding 8 to 9 GB of the card, 156 lanes fitted at
8k of conversation context. Checkpoint before the work: commit `3e2c0f1`, tag `before-speed-7h`. Every number is
the best of several runs (the card is shared with the desktop, and single runs vary by up to 30%).

## Result

The engine, the original binaries (`bin-before-speed-7h`) and the final ones measured alternately on 20 Nov at 01:50,
twice each (`fit.pass_ms`, best of 3; the two runs agreed within 0.6 ms):

| Lanes | Original | Final | |
|---|---|---|---|
| 1 | 11.6 ms (86 tok/s) | 11.3 ms (88); 11.1-11.2 ms (90) with 20 | -4% |
| 8 | 17.5 ms | 16.9-17.5 ms; 14.9-15.1 ms with 21 | -14% |
| 32 | 24.7-24.9 ms | 24.4 ms | -2% |
| 64 | 34.3 ms | 33.3-33.5 ms | -2.6% |
| 128 | 57.9 ms | 54.7-55.1 ms | -5% |
| 156 | 69.3 ms (2,251 tok/s) | 64.6-65.1 ms (2,410) | -6.4% |
| reading a prompt | 3,248 tok/s | 3,337 tok/s | +2.7% |

Through the app (`tests/perf_bench.py`, REPEAT=2, questions asked one after another), b7base against final9 (with
everything kept, up to 21):

| Case | Baseline | Final | |
|---|---|---|---|
| one lane: define / explain / code | 61 / 57 / 79 tok/s; 1.3 / 1.8 / 6.3 s | 66 / 60 / 87; 1.2 / 1.6 / 5.8 s | 1.05-1.10x |
| Medium: how-to / causes / comparison | 148 / 100 / 149; 4.3 / 4.0 / 5.3 s | 221 / 138 / 226; 3.2 / 3.0 / 3.4 s | 1.38-1.52x |
| Detailed | 176, 6.5 s | 225, 4.3 s | 1.28x |
| Report | 413 (34 lanes, 9.7 s) | 378 (31 lanes, 9.7 s); 260-480 over about 20 runs as the plan comes out | 0.92x |
| "everything about" | 436 (8,100 tokens, 18.6 s) | 673 (11,276 tokens, 16.8 s) | 1.54x |
| long article / long article 2 | 805 / 754 (14,493 / 12,089 tokens) | 1,042 / 1,072 (21,369 / 19,853 tokens) | 1.29x / 1.42x |
| peak of a long answer | 1,913-2,011 | 1,915-2,220 | |

A short answer after a pause, typed in the box (14): "Define entropy." 1.32 s -> 0.98 s, first text 0.42 -> 0.15 s
(1.35x); "Explain in one paragraph how vaccines work." 1.79 -> 1.51 s.

The targets: 3,000-5,000 tokens a second on long answers was not reached. The engine's best is about 2,400 with all
156 lanes, and a whole long answer, with its plan written first by one to a few lanes and its paragraphs ending one
after another, about 850-1,050. Twice the speed for a single short answer was not reached either: one lane is bound
by reading the 7 GB of weights once per token (the matrix-vector kernels at about 90% of the card's memory bandwidth
take 7.9 of 11.3 ms), and drafting with a small model did not pay (1); what was gained for short answers is the time
before the first word after a pause (1.35x for a one-paragraph definition).

Quality: the engine changes 12 and 13 give the same text on all 156 lanes; 5, 7, 15 and 16 change the order or width
of passes, which can tip near-ties (121 of 156 lanes the same with 5). The fast regression with the final binaries:
57/58 (only continue_long, which failed before the work too). One run before 16 had 55/58: no_fuel then passed 3 of 3
reruns, and internet_history failed on the order of bullets inside one section (also, sometimes, with the old
narrowing).

## Baseline

The engine alone (`fit.pass_ms`, best of 3, 48 tokens per lane; measured at 21:45 with other programs on the card;
see the Result above for the original binaries measured again beside the final ones):

| Lanes writing | Pass | Tokens a second |
|---|---|---|
| 1 | 12.1 ms | 83 |
| 2 | 12.4 ms | 161 |
| 5 | 15.3 ms | 326 |
| 8 | 17.8 ms | 450 |
| 16 | 20.5 ms | 779 |
| 64 | 35.8 ms | 1,787 |
| 112 | 56.0 ms | 2,001 |
| 156 | 63.4 ms | 2,459 |

Through the app (`tests/perf_bench.py b7base`, REPEAT=2):

| Case | Whole answer | While the lanes write | Peak |
|---|---|---|---|
| one lane: define / explain / code | 61 / 57 / 79 tok/s | | 83-84 |
| Medium (5 to 7 lanes) | 100-149 tok/s | 244-330 | 285-359 |
| Detailed (45 lanes) | 176 tok/s | 937 | 1,144 |
| Report (34 lanes) | 413 tok/s | 1,003 | 1,134 |
| Longest, "everything about" (104 lanes) | 436 tok/s | 1,479 | 1,736 |
| Longest, "a long article" (107-108 lanes) | 754-805 tok/s | 1,590-1,653 | 1,913-2,011 |

Where a pass goes (the per-operation timer, `GGML_CUDA_OPTIME`, operations synchronised, so the totals are inflated):
the ternary matrix products (MMQ, int8 tensor cores) and the gated-delta-net state update grow with the lanes; between
8 and 156 lanes they make about 55% and 28% of the added time. Normalisations, multiplications, additions and small
copies hardly grow. The products already run at about half the card's int8 peak, and the state update close to its
memory bandwidth, so the engine's ceiling at this lane count is near 3,300 tokens a second (12 ms + 0.3 ms per lane).

Where a long answer's 18 s go ("a long article", 14,493 tokens): question read 0.07 s, brief answer 1.0 s, outline
2.9 s, points 2.0 s (one lane per part), paragraphs 11.4 s (108 lanes).

## What was tried

### 1. A small draft model for one-lane answers (rejected)

Ternary-Bonsai-2-27B has no MTP head, so a sibling of the same family drafted instead:
`prism-ml/Ternary-Bonsai-2-2B-gguf` (Q2_0, 0.51 GB, Apache-2.0, same 248k vocabulary). Built as `--draft <gguf>` for
`/spec` (an engine patch kept outside the repository): sequence 0 of the draft model's context holds what was
accepted, each cycle's guesses are made on a copy (`llama_fq_fork`) that is dropped afterwards, the full model checks
`[last, d1..dK]` in one pass and is rolled back with its snapshots.

| Drafts per cycle | Plain | Drafted | Kept |
|---|---|---|---|
| 4 | 83.5 tok/s | 75.9 (0.91x) | 30-66% |
| 3 | 84.1 | 78.6 (0.93x) | |
| 2 | 82.9 | 74.9 (0.90x) | |

Why: a cycle cost 39 ms, 18.8 ms of drafting (4.7 ms a draft step: a 0.5 GB model, bound by kernel launches) and
20.4 ms to check 5 tokens (a lane pass with 5 tokens costs 15.3 ms; one token 12.1 ms). Breaking even needs about
3.3 kept tokens a cycle; the 2B sibling gave 2.6. Undone; the patch is kept for a card or model where a multi-token
pass costs about what a one-token pass does.

### 2. Long answers: the outline written once (kept, small)

For Detailed, Report and Longest the outline was written twice: once beside the brief answer (to decide the route,
which the length has decided already) and again after it, grounded on the brief answer. Now, when the length forces a
plan, the brief answer is written first and the outline once after it (`FAN_GROUND_FIRST`, on). The outline ends
0.1 to 1.0 s sooner (long article 3.95 -> 3.60 s, 3.52 -> 2.51 s; detailed 2.25 -> 2.15 s; "everything" 4.48 ->
4.01 s). A report is left out: its brief answer is an executive summary of three or four sentences, and the outline
waited for it (2.36 -> 3.30 s).

### 3. Long articles: shorter paragraphs (rejected)

A long article's paragraph lanes write a median of 107 to 154 tokens, the longest 219 to 235, and the answer waits for
the longest. With 120 words a paragraph instead of 150 (`RAPIDLI_LONG_WORDS`) the longest lane wrote 180 tokens and an
article took 14.5 to 14.7 s instead of 14 to 18 s, but it was also 15% shorter: 800 to 877 tok/s against 638 to 853.
Within the noise for whole answers that say less; left at 150.

### 4. Where a pass goes, measured with Nsight Systems (real kernel times, CUDA graphs traced)

| | 1 lane | 156 lanes |
|---|---|---|
| pass (under the profiler) | 13.3 ms | 73.5 ms |
| ternary products | 8.19 ms (`mul_mat_vec_q`, 85% of the memory bandwidth) | 36.9 ms (`mul_mat_q`, ~228 int8 TOPS, about a third of the card's peak) + 2.0 ms stream-k fix-up |
| gated delta net | 0.19 ms | 13.6 ms (at the memory bandwidth: 12 GB of 8-bit state read and written) |
| about 20 kinds of small kernels (copies, concat, norms, Hadamard transforms, attention) | ~3 ms | ~13 ms |
| gaps between kernels | 1.9 ms (2,231 kernels a pass) | 4.6 ms (3,253 kernels) |

One lane is within 35% of reading the weights once a pass (8.2 ms): doubling it is not possible without more than one
token a pass, and §1 shows that drafting does not pay here. For many lanes the products are the lever: at 156 lanes
the kernel cuts the lanes into two tiles of 80 columns, so every tile of 2-bit weights is unpacked twice a pass.

### 5. Wide matrix tiles for Q2_0 (kept, small)

`engine/cuda/patch_mmq_wide.py`: tiles of 136 to 160 columns for Q2_0, so that a pass of up to 160 lanes unpacks every
tile of 2-bit weights once instead of twice. Measured twice, alternating the libraries: 144 lanes 66.8 -> 62.7 ms a
pass (-6%), 156 lanes 69.2 -> 67.4 ms (-2.5%), 128 lanes unchanged. Same text on 121 of 156 lanes (the others differ
where two tokens were nearly tied: the sums are added in another order), one lane unchanged (another kernel).
Tried and rejected: tiles of 64 rows for the wide widths (the engine failed with a CUDA error on the first wide pass).

### 6. Reading a short prompt when the lane is alone (kept)

Prompts of up to 96 tokens were fed one token per pass so that other lanes keep generating; a lane alone read a
45-token prompt in 45 passes. Now a short prompt is read in one bulk pass when its lane is the only one with a request
(`--alone-bulk`, on): 30 tokens in 36 ms instead of about 370.

### 7. Narrowing the pass after a long answer (kept)

The pass narrowed only after 48 passes with fewer lanes, so a short answer after a long one ran its first 48 tokens in
156-lane passes (65 ms a token instead of 12). Now it narrows after 3 passes when the lanes at work fit a quarter of
the width or less.

### 8. Smaller things measured

- Chunks of 512 tokens for long prompts instead of 256: 3,484 against 3,249 tokens a second read, for 89 MB more of
  compute buffer (about one lane). Left at 256.
- The gap between two passes on the card: 0.04 ms (the engine already launches the next pass at once); gaps between
  kernels inside a pass: 1.1 ms (1 lane), 3.4 ms (156 lanes).
- Up to 10 points per section for Longest instead of 8 (`RAPIDLI_LONG_POINTS_MAX`): long articles used 138 to 146
  lanes (20,624 tokens, 976 tok/s) instead of 95 to 115, but "everything about" got a 10-section outline and a 7 s
  round of points written again (301 tok/s). Not kept until it holds on several runs.

Fast regression after 2, 5, 6 and 7: 57 of 58 turns (the known "continue" case).

### 9. Medium answers: the second outline only for small models (kept)

An answer the router decides to plan wrote its outline twice: once to decide the route, and again after the brief
answer, grounded on it (added because NVIDIA-Nemotron-3.5-Bolt-E4B outlined Carl Sagan for Peter Sagan). The 27B models
kept to the subject of their brief answer on five ambiguous names (Peter Sagan, Mercury, Michael Jordan, Jaguar, Java),
so the second outline is now written for small models only (width under 4,096 or fewer than 40 layers;
`RAPIDLI_GROUND=1` always). Two runs each:

| Case | With the second outline | Without |
|---|---|---|
| How-to (6-7 lanes) | 148 tok/s | 184 tok/s |
| Causes (5 lanes) | 110 | 124 |
| Comparison (7 lanes) | 172 | 191 |

Fast regression 55/58; the two new failures (a vegetarian plan that named chicken, a schedule whose times went back)
passed on 3 and 2 reruns: they come and go (one-lane answers, whose prompts are now read in one batch).

### 10. Also measured, not kept

- MMQ instead of MMVQ for small batches of Q2_0 (`GGML_CUDA_MMVQ_Q2_0`, a switch in our ggml-cuda): with MMQ from 3
  or 5 lanes up, a pass of 3 to 6 lanes took 17.5-18.9 ms instead of 14.4-16.4. MMVQ up to 8 lanes stays.
- A shorter offline-Wikipedia check (600 instead of 1,200 characters per article, `RAPIDLI_KCHECK_CHARS`): 0.15 s
  instead of 0.24 on average, but 1 of 12 decisions changed (the boiling point of water lost its article). Left at 1,200.
- Nsight Compute could not read the card's performance counters (ERR_NVGPUCTRPERM: needs an administrator setting),
  so the matrix kernel's limiter (memory, shared memory or tensor cores) is not measured.

### 11. Longest articles: up to 10 paragraphs a section (kept, for articles only)

Three runs each of the long cases, up to 8 or 10 points a section (`RAPIDLI_LONG_POINTS_MAX`):

| Case | 8 points | 10 points |
|---|---|---|
| long article | 887 tok/s, 15,321 tokens, 102-117 lanes | 973 tok/s, 18,068 tokens, 123-141 lanes |
| long article 2 | 788 tok/s, 12,200 tokens | 858 tok/s, 15,437 tokens |
| "everything about" (bullets with sub-bullets) | 540 tok/s | 421 tok/s: two of three runs fell to 36 and 44 lanes |

So a long article gets up to 10 paragraphs a section where 140 lanes or more are fitted (`RAPIDLI_ARTICLE_POINTS_MAX`),
and bullet answers keep 8. Two runs after the change: long articles 971 and 958 tok/s (up to 155 lanes, 17,600-19,600
tokens), "everything about" 576 tok/s (115-119 lanes). Same structure as with 8 points (17-19 sections, paragraphs of
87-114 words, no sentence said twice, none unfinished); the paragraphs of a section open on different subtopics.

### 12. The gated-delta-net convolution as one kernel (kept)

In each of the 48 gated-delta-net layers the stored convolution state (3 columns a channel) was joined with the new
channels (CONCAT, written out), convolved (SSM_CONV + SILU, reading it back), and its last 3 columns copied back as the
new state (CPY, reading it a third time): 2.1 + 2.5 ms of a 156-lane pass went to the join and the copy alone.
`engine/llama/patch_zzconv.py` puts the convolution right after the join in the graph (same operations), and
`engine/cuda/patch_convfuse.py` runs the four as one kernel: it reads the state and the new channels once, adds in the
same order as ssm_conv, and writes the output and the new state. The concat's inputs are kept allocated until the copy
(an allocation dependency), so the output cannot share their memory.

| Lanes | One by one | Fused |
|---|---|---|
| 1 | 11.7 ms | 11.6 ms |
| 8 | 18.4 | 18.2 |
| 64 | 35.2 | 33.9 |
| 128 | 59.1 | 56.6 |
| 156 | 68.8 | 66.2 (2,357 tok/s) |

156 of 156 lanes wrote the same text with and without it; reading a 2,750-token prompt: 845 ms against 852.

### 13. The sign flip inside the Hadamard kernel (kept)

Bonsai rotates the input of most matrix multiplications: MUL(x, signs), then a fast Walsh-Hadamard transform. The MUL
was a kernel of its own, about 258 a pass. `engine/cuda/patch_fwhtsigns.py` has the transform read x and the signs and
compute (x * sign) * scale, the same two multiplications in the same order.

| Lanes | Two kernels | One |
|---|---|---|
| 1 | 11.6 ms | 11.3 ms (89 tok/s) |
| 2 | 12.6 | 12.3 |
| 8 | 18.4 | 18.2 |
| 64 | 33.9 | 33.4 |
| 128 | 56.5 | 55.8 |
| 156 | 66.0 | 65.2 (2,394 tok/s) |

156 of 156 lanes wrote the same text.

Looked at and left: the fork's fused gate/up/SwiGLU MMQ kernel has no Q2_0 loader and runs without stream-k (one block
per tile): the gate and up matrices are 136 tiles each on 128 SMs, a poor fit without stream-k.

### 14. Short answers after a pause: the card warmed while the user types (kept)

A question asked after a few seconds of nothing started at idle clocks: the card drops to P3/P8 (memory at 405-5,001
MHz), and the first prompt read after 8 s idle took 0.59-0.67 s against 0.39 s back to back. Decoding was at full
speed at once (11.2 ms a pass); reading a prompt, compute-bound, was not. The offline-Wikipedia check, the first thing
a question reads, paid it every time (0.45-0.56 s inside the app against 0.25 s in a loop).

The engine's `/warm` runs a 4096x4096x1024 f16 multiplication on a backend of its own, repeated for up to 400 ms or
until a request comes in; the page calls `/api/warm` while the user types (at most every 1.5 s, only with nothing being
answered). No lane, no counter, nothing in the panel; the backend takes about 200 MB of the card, made at the first
warm-up. An elementwise pass over 32 MB instead (memory-bound) did not bring the clocks up (0.63 s after it).

| 8 s idle, then the question (3 runs each) | Cold: first text, total | 1 s after a warm-up |
|---|---|---|
| "Define entropy." (an article goes with it) | 0.42 s, 1.32 s | 0.15 s, 0.98 s |
| "Explain in one paragraph how vaccines work." (the check says no) | 0.62 s, 1.79 s | 0.35 s, 1.51 s |

A request sent during a warm-up is not slowed (0.40 s against 0.39). The benchmarks below ask their questions back to
back, so they do not show this.

### 15. The pass narrows sooner (kept, small)

When lanes finish, the pass narrowed only after 48 passes in a row that a narrower width would have covered (about 3 s
at 156 lanes). Measured on the bare engine with a paragraph-like phase, 150 requests at once of 60 to 250 tokens each
(greedy, end of text ignored), two runs each: waiting 48 passes 14.61 and 14.64 s, 8 passes 14.33 s, 3 passes 14.45 s;
after the change 14.39 s. Now 8 (`--narrow-wait`, or `RAPIDLI_NARROW_WAIT`). In the app the long answers vary too much
from run to run (12,400-20,300 tokens, 116-153 lanes) for a 2% difference to show.

Not pursued: running the Wikipedia check and a speculative answer at the same time. The check's reading is
compute-bound and the answer's lane would share its passes (one token per bulk pass), so the answer would hardly move
while the check runs.

### 16. Bulk reading with a few lanes busy, only for longer prompts (kept, fixes 6)

Measured back to back at the end, the final engine was slower than the original with 4 to 8 lanes (8 lanes 18.3 ms a
pass against 16.9). The cause was `--few-bulk 8` (6): with 8 lanes or fewer at work, every lane's new prompt was read
in a bulk pass of its own, one lane at a time, also the parts of a Medium answer, whose own text is only "[3]\n".
Eight parts then started one after another: on the bare engine, 8 branches of 120 tokens from one prompt took 2.50 s
against 2.43 s read one token a pass together (5 branches: no difference). Now a lane is read in bulk only when it has
more left than two passes for each lane at work (alone: more than 2 tokens, as before). After the change: 8 branches
2.44 s, passes with 4 / 6 / 8 lanes 13.7 / 14.7 / 17.3 ms (before 14.3 / 16.0 / 18.3), a lone prompt still read at
once (0.39 s for the 1,200-token check). In the app, two runs: how-to 209 tok/s, causes 130, comparison 204 (3.8 s),
Detailed 195.

### 17. Planned answers of a large model: the outline beside the brief answer (kept)

An answer planned because of its length (Detailed, Report, Longest) wrote its outline after the brief answer, grounded
on it (Detailed and Longest: once, after it, see 2; Report: beside it and again after it). As in 9, a large model does
not need it, so the outline is now written at once, next to the brief lane (`RAPIDLI_GROUND_PLANNED=1`: as before).
Two runs each:

| Case | Grounded | At once |
|---|---|---|
| Detailed | 5.4 s, 188 tok/s | 4.6 s, 224 tok/s |
| Report (28 / 27-32 lanes) | 9.7 s, 326 tok/s | 10.7 s, 330 tok/s (394 and 267: the plan varies) |
| long article | 17.8 s, 988 tok/s | 15.9 s, 1,050 tok/s |
| "everything about" | 19.1 s, 640 tok/s | 17.5 s, 658 tok/s |

Six ambiguous names at Detailed size, at once: every outline followed the subject of its brief answer (Peter Sagan the
cyclist, Mercury split into planet, element and programme, the jaguar, Michael Jordan the player). "Who was
Washington?" found the article on the state; at once the sections were about the state, grounded they were about the
article ("What the question likely means", "How to find the right article"): neither better. Fast regression 57/58.

### 18. Long prompts read in passes of 320 tokens for Q2_0 (kept)

A long prompt is read in passes of `--chunk` tokens. With the 160-column tiles of 5, 320 tokens are exactly two tiles.
Alternating on the bare engine: 2,750 tokens read at 3,308 / 3,322 tokens a second with 256, at 3,471 / 3,442 with 320
(288: 3,497 once); a 1,200-token check prompt 0.39-0.41 s against 0.37-0.39 s. The card's memory after reading and a
156-lane pass: 21,620 MiB with 256, 21,579 with 320. The app now starts a Q2_0 model with 320 (`app/fit.py`); other
types keep 256, which fills their 128-column tiles.

### 19. Empty points written again at once (rejected)

In a bullet answer, a lane can write a figure or its own label and stop or loop ("1903\n- **The role of the wind
tunnel:** 1904 ..."); such points are written once more in a round after all the others: 1.8 s of a report's 9.1 s,
4.1 s of an 18.8 s "everything about" answer (timeline: "points written, some empty", now marked). Tried: the engine's
`/fork` sent each branch's end as it happened, and a point that ended early with next to nothing was written again at
once, beside the lanes still writing. The lanes that loop run to their token limit and end with the round anyway, and
each early rewrite read the plan again in bulk passes that slowed the main round. Three runs each, off / on:
"everything about" 17.6 / 20.9 s, report 9.8 / 12.9 s (2,700 / 4,000 tokens), Detailed 4.4 / 4.4 s. Reverted
(engine and app); a way that might work is stopping a looping lane while it writes.

Checked and left: stopping a lane that loops while it writes. In an "everything about" answer the longest lanes were
three loops at the 188-token limit (8.4-8.5 s), but bullets that end properly ran 7.7-8.3 s, so at most 0.2-0.5 s.

### 20. The Hadamard kernel writes the vector kernel's input (kept, small)

At one to a few lanes each matrix-vector product first ran a kernel that quantizes its input to q8_1 blocks (332 a pass
at one lane, about 1.1 us each plus the gap). The Hadamard transform before it already holds 32 consecutive values in
one warp, so `engine/cuda/patch_zfwhtq8.py` has it write the same blocks (the same max, d = amax / 127, roundf and sum
as quantize_q8_1) into a buffer of the context, and MMVQ takes them when its input is that output. Alternating, off /
on: 1 lane 11.4 / 11.1 ms and 11.5 / 11.2 ms, 2 lanes 12.3 / 12.0 and 12.2 / 12.0, 4 lanes 13.5 / 13.4 and 13.7 / 13.4;
8 lanes 17.0 / 17.1; 156 lanes 64.3 ms. The same text on 1, 8 and 156 lanes.

### 21. Four rows per block in the vector kernel for 2 to 8 lanes (kept)

A profile at 8 lanes: the matrix-vector kernel took 12.0 of 16.3 ms of kernel time, against 7.9 ms at one lane for
the same weights, so with 8 columns it was no longer bound by memory. Its launch table gives 2 rows per block for 2 to 8
columns; with 4 for Q2_0 (`engine/cuda/patch_zmmvqrows.py`) a block reads the q8_1 input once for twice the rows, and
each row is summed in the same order. Alternating twice, old / new:

| Lanes | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| old | 11.2-11.3 | 12.0 | 13.2-13.4 | 13.0-13.3 | 14.7-15.2 | 15.0 | 17.1 | 16.8-17.1 | 19.4-19.5 |
| new | 11.0-11.2 | 11.9-12.1 | 13.0 | 12.8-12.9 | 13.9-14.0 | 13.4-13.8 | 14.9-15.1 | 14.9-15.1 | 19.3-19.4 |

The same text on 1, 8 and 156 lanes. Tried and left: 2 rows for one column (11.2 -> 11.3 ms), 8 rows for 5 to 8
columns (8 lanes 16.5 ms), and 4 warps instead of 2 for 5 to 8 columns (5-8 lanes 15.1-16.5 ms against 13.6-15.2).
Medium answers (5 to 8 lanes) gain most.

### 22. The vector kernel for 9 to 16 lanes (tried, left; 20 Nov morning)

**Hypothesis:** 9 to 16 lanes run on MMQ (12 lanes 19.4 ms against 15.0 at 8 on the vector kernel); the vector kernel
extended to 16 columns for Q2_0 should continue the 2-8 curve.

**Result, alternating on one build (GGML_CUDA_MMVQ_Q2_0=16 against 8):**

| block shape for 9-16 | 9 lanes | 12 lanes | 16 lanes |
|---|---|---|---|
| MMQ (as kept) | 19.6 ms | 19.5 ms | 20.2 ms |
| 1 warp, 2 rows | 21.4 | 21.3 | 27.7 |
| 2 warps, 4 rows | 20.5 | 21.2 | 25.3 |

**Why (from the machine code):** the weight unpack is already shared across columns (40 byte permutes whatever the
width), but every column costs about 9 four-byte loads of its q8_1 input (blocks of 36 bytes cannot be loaded 16 bytes
at a time) and 8 dot products per row pair, and the warp sum costs 5 shuffles and 5 adds per output: at 16 columns the
kernel issues roughly twice the instructions per weight of the 8-column one.

**Also tried:** the warp sum as a reduce-scatter (the same additions, so the same bits, about one shuffle per output).
It made every width slower (8 lanes 15.0 -> 18.8 ms): the selects between halves kept 9-16 columns at 200-255
registers with a 256-byte stack. Left out.

**What could still work:** a Q2_0 kernel that reads its input in a 16-byte-aligned layout (the Hadamard kernel already
writes the input and could write that layout) so a column costs 3 loads, not 9. The patches of this attempt are not in
engine/cuda; the installed DLL is the one of section 21.

### 23. Drafts copied from the text, chosen by measured cost (kept; 20 Nov evening)

**Idea (prompt-lookup drafting):** a single answer that repeats its prompt (a rewrite,
code given back with changes, a summary that quotes a source) can be drafted by copying. If the last tokens occurred
before, the tokens that followed them then are the draft, and the full model checks them in one pass. Every check's cost is measured for each number of rows, and
the expected tokens a millisecond decide each cycle among three options: MTP drafts of 1 to n tokens, a copied draft of
1 to 8 tokens, or none. Copying needs no MTP head, so `--lookup` gives a model without one (Ternary-Bonsai-2-27B) the
drafting route.

**Result (RTX 4090, one answer of 320 tokens):**

| | prose | summary of a source | rewrite | code given back |
|---|---|---|---|---|
| Bonsai-2-27B, a lane | 82 | 82 | 82 | 82 |
| Bonsai-2-27B, `/spec --lookup 8` | 68 | 95 | 178 | 196 |
| Qwen3.8-27B Q3_K_P, MTP 4 (installed engine) | 79 | 113 | 130 | 133 |
| the same, MTP 4 + lookup 8 | 83 | 124 | 134 | 141 |

**Costs and limits:**
- The longer rollback costs card memory: about 650 MB on the MTP model, and about 1.3 GB on Bonsai, which had no
  drafting context before. Bonsai fitted 107 lanes next to 10 GB held by a screen recorder, against about 123 without it.
- Prose loses on the drafting route without an MTP head: its passes see the whole 8k drafting context.
  `Fan.drafting` therefore sends a single answer there only when the user's own text has something to copy: 1,500
  characters or more, code, or a rework asked for. Search results attached by the app do not count: "Why is the sky
  blue?" with its Wikipedia articles kept 9 of 154 copied tokens, and took 3.7 s against 2.1 s on a lane.

### 24. The chunked tensor-core recurrence for a 16- or 8-bit state (tried, left)

**Idea:** a chunked tensor-core gated-delta-net can be about 3 times faster than token by token. Our llama.cpp has a
chunked kernel too, but it takes a float32 state only. The lanes' 8-bit state and /spec's 16-bit one always ran the
token-by-token kernel. The attempt widened the pass's state to float32 around the chunked kernel and narrowed it after,
with the same rounding.

**Result:**
- The per-operation timer, for one 320-token pass of Bonsai-2-27B: the recurrence took 8.1 ms off and 7.0 ms on, of
  about 95 ms. The Q2_0 matrix products took about 77 ms.
- Prompt reading of 5,900 tokens: 3,255 and 3,276 tok/s installed, 3,262 off, 3,306 and 3,277 on, which is noise.
- Left out: the recurrence is too small a part of this model's prompt reading.

### 25. A mixture of experts larger than the card: experts in the PC's memory (kept; 20 Nov evening)

**Model:** Qwen3.8-Flash-Next, ISTA-DASLab Q2_0: 3.4 GB on the card, 31.6 GB of experts, and a 26.8 GB n-gram table.
**Machine:** RTX 4090, i9-14900KF, 64 GB of RAM, of which other programs left about 20-30 GB free.

**What was built:**
- Engine:
  - `--experts-ram 1`: the routed experts sit in a CPU buffer read from the mapped file.
  - Expert caches on the card, with llama.cpp's own MoE cache (recency and frequency): `--moe-cache-mib` for the
    lanes and `--moe-spec-mib` for /spec.
  - The share of cache misses computed by the processor: `--moe-cpu-share` for the lanes, `--moe-spec-cpu-share` for
    /spec.
- llama: `patch_zzidx.py`.
  - `llama_fq_fork` now copies the indexer cache of `llama_memory_hybrid_idx`.
  - Without it, a forked lane stopped on `GGML_ASSERT(r >= 0)` in llama-memory-hybrid-idx.cpp.
- App:
  - `fit.fit_moe`: probes without a cache and gives the rest of the card to the caches, 70% of it to /spec.
  - The processor threads are kept on the 8 P-cores (`plat.fast_cores`, `plat.pin`), one core left to the thread that
    drives the card.
  - The 16-bit state is used: the 8-bit one stops qwen4exp's graph on a reshape.
  - Prompts are read in passes of 1,024 tokens, /spec's too.
  - `Fan.drafting` sends every single answer to /spec.

**Measured, one answer on /spec, warm (`moe_ab.py`):**

| Setting | tok/s |
|---|---|
| No misses on the processor, 6 GB cache | 45.3 |
| All misses on the processor | 32.4 |
| 16 threads, no pinning | 31.4 |
| No misses on the processor, 9 GB cache | 53.6 |
| llama.cpp's own server, experts in memory (`--cpu-moe`, `--moe-cache-mib 6000`, `-t 7`), the same question asked twice | 45.0 |

**Measured, lanes (`moe_ab2.py`):**
- 16 lanes: 130 tok/s with half the misses on the processor, 104 with none.
- 8 lanes: 136 and 121.
- Prompt reading: 196 tok/s at 320 tokens a pass, 411 at 1,024. Every pass copies most experts to the card.

**Through the app:**
- Before /spec read prompts in long passes, a question with its offline articles waited 41-69 s for its first words;
  after, 16.7 s.
- Writing went at about 27 tok/s. Windows had 7-8 GB free for the 20 GB of experts the card does not keep, so the rest
  came from the SSD.
- With 1.5 GB free (more programs open): 32 s to the first words, 9 tok/s.

**What decides the speed on this PC is free RAM.** The ISTA-DASLab GGUF has no MTP head.

**Not done:**
- Pinned host memory for the experts. It would double the upload rate, but needs about 32 GB that the PC does not have
  free.
- A cache shared by the lanes and /spec. Each llama context has its own.

### 26. Many lanes on a mixture of experts: the experts' products in slices of 8 tokens (kept; 20 Nov night)

**Finding:** with a 15 GB expert cache the hit rate was 96-98%, and the cache's lookups took 5-9 ms a pass, yet 16
lanes took three times as long per pass as 8. A separate attention cache per lane (`--kvu 0`) did not change it.

The cause is in MUL_MAT_ID:
- Up to 8 tokens, llama.cpp uses the vector kernel, which looks up each token's experts.
- From 9 tokens on, it uses MMQ tiles. With the MoE cache, the expert tensor is the cache's bank of about 11,000 slots,
  and the MMQ grid covers every slot, used or not.
- `engine/cuda/patch_zmmidslice.py` runs 9-64 tokens on the vector kernel, 8 at a time, when the experts outnumber the
  token-expert pairs four times or more. With 32 tokens x 10 experts over 512, a token rarely shares an expert, so the
  weights read per slice cost about what MMQ reads once.

**Result (Qwen3.8-Flash-Next Q2_0, a 15 GB cache, 42 GB of RAM free):**

| | before (MMQ) | after (slices) |
|---|---|---|
| 8 lanes | 200 tok/s | 219-229 |
| 16 lanes | 140 | 265-280 |
| 32 lanes | about 130 | 380 (436 with a quarter of the misses on the processor) |
| 48 lanes, 12 GB cache, `LLAMA_MOE_CACHE_MAX_BATCH=48` | - | 397 |
| one lane | 51 | 51-56 |

- 64 lanes with a 15 GB cache overfilled the card, and every width fell to 18-76 tok/s.
- Bonsai-2-27B was unchanged: prompts 3,343 against 3,330 tok/s, one lane 87 against 88. Dense models do not use
  MUL_MAT_ID.

**What changed in the app (`fit.fit_moe`):**
- One cache for the lanes: one lane writes as fast on it as /spec did on a cache of its own.
- No drafting context unless the model has an MTP head, and no processor share.
- 32 lanes, with `LLAMA_MOE_CACHE_MAX_BATCH` set to the lanes. `patch_zzmoelog.py` makes the 32-token limit a setting,
  and `LLAMA_MOE_CACHE_LOG=<n>` prints the cache's figures every n lookups.

**Through the app (14 GB cache, 8 GB of RAM free):**
- The fit measured 403 tok/s with 32 lanes.
- A question with its offline articles: first text in 6.4 s (16.7 s before), 15 s in all.
- A long article: 3,633 words (4,357 tokens) in 53 s, first text at 9.7 s, on 32 lanes.
- Thinking ("Is 391 prime?"): 8.7 s.

### 27. A mixture of experts that fits the card: Qwen3.8-35B-A3B (record; 20 Nov night)

**Model:** empero-ai/Qwen3.8-35B-A3B-Distill IQ3_M: 15.2 GB, `qwen35moe`, 256 experts with 8 used per token, and an
MTP head.

**Slicing (section 26) does not pay here.** The expert tensor has 256 experts and there is no cache bank, so MMQ is
the better kernel. With slicing forced (`GGML_CUDA_MMID_SLICE_RATIO=0`), 32 lanes gave 1,454 tok/s against 1,809
without it. The default rule (4 times more experts than token-expert pairs) never slices for this model.

**The 8-bit state:** `patch_zzi8more.py` gives qwen35moe, qwen4exp and qwen3next the 8-bit recurrent state that
patch_i8.py gave qwen35. Before it, their graph stopped on a reshape.

| state | 1 lane | 32 | 64 | 96 | 128 | 160 | 200 | 232 |
|---|---|---|---|---|---|---|---|---|
| 16-bit (3.4 GB at 96 lanes) | 208 | 1,809 | 2,622 | 2,865 | - | - | - | - |
| 8-bit | 207 | 1,867 | 2,747 | 3,170 | 3,472 | 3,837 | 3,899 | 4,131 |

At 232 lanes, 2.4 GB of the card was still free.

**Through the app:**
- Fitted with the automatic context (128k): 114 lanes, 3,502 tok/s together.
- One answer (MTP drafting): 334 tokens in 3.0 s, first text at 1.6 s.
- A Longest article: 7,647 words (10,016 tokens) in 15.7 s.

### 28. Drafts inside the lanes (batched MTP): measured, not built (20 Nov night)

**Reference implementations:**
- llama.cpp's own server (`--spec-type draft-mtp`, 3 drafts) on Qwen3.8-27B Q3_K_P (HauhauCS) was slower:
  - 8 parallel requests: 257 tok/s without drafts, 207 with them.
  - 16 parallel requests: 339 without, 28 with.
  - It kept 37-47% of its drafts.

**Our engine** (a stand-alone benchmark, the same model, one pass):

| lanes | A: 1 row, 8-bit | C: 1 row, 16-bit | D: 2 rows, 16-bit | B: 2 rows, 16-bit, rollback snapshot, MTP step |
|---|---|---|---|---|
| 8 | 20.1 ms | 20.8 | 23.1 | 34.8 (MTP 2.1) |
| 16 | 23.2 | 24.6 | 32.6 | 56.2 (MTP 2.2) |
| 32 | 33.0 | 35.3 | 37.0 | 85.9 (MTP 2.7) |

**The MTP head's first draft is kept 82% of the time (prose) to 99-100% (rewrite, code)**, measured on /spec with
`--spec 1`.

With about 85% kept:
- llama.cpp's rollback (B) gives 1.07x at 8 lanes, 0.76x at 16 and 0.71x at 32.
- A rollback as cheap as no rollback (D + MTP) would give about 1.48x, 1.23x and 1.53x.

**Why B is slow:** with rollback snapshots (`n_rs_seq` > 0), llama.cpp splits the pass into two ubatches, one token
per lane each (the operation counts per graph halve, the graphs double), and gathers and copies the recurrent state
(CPY, CONCAT, GET_ROWS) instead of updating it in place.

**What would make it pay:** a single ubatch with the state updated in place, where the gated-delta-net kernel also
writes the state after the first of the two tokens to a side slot, and a rejected draft is rolled back by copying that
slot (about 38 MB a lane at 8 bits). That needs a new memory path in llama and a kernel change. Not started.

### 29. A mixture of experts in the PC's memory: the experts pinned (kept; 22 Nov)

**Finding:** through the app, Qwen3.8-Flash-Next wrote a long article at 110-140 tok/s on 30 lanes, where the fit's
test passes gave ~290. The cache log (`LLAMA_MOE_CACHE_LOG=400`) showed why: real prose hit only 81-88% of the
experts while the lanes wrote, 9-12 GB were uploaded per 400 lookups, and the lookups (the uploads included) took
1.1-1.8 s of every 400, nearly the whole pass. The uploads read the mapped model file, pageable memory, which the
driver copies through its own staging buffers.

**Measured (the GGUF mapped read only, 512 MB copies):** pageable 9.8 GB/s, the same pages
registered with `cudaHostRegister` (portable, read only) 23.5 GB/s. Windows accepts the read-only registration of a
file mapping.

**Built:** `engine/llama/patch_zzmoepin.py`. The cache merges its host expert tensors into ranges and registers them
once per process (31.83 GiB in 48 ranges for Flash); a range that cannot be locked stays pageable.

| Qwen3.8-Flash-Next Q2_0, 32 lanes, 12.4 GB cache | before | pinned |
|---|---|---|
| fit pass, 8 lanes | 73 ms | 38.5 ms |
| fit pass, 32 lanes | 111 ms | 65 ms (~490 tok/s) |
| lookups per 400, while writing | 1,100-1,800 ms | 22-47 ms |
| Longest article, steady 5 s windows | 110-140 tok/s | 195-206 tok/s |
| Longest article, all | 55 s | 40 s |
| 64 canvas effects, while the lanes write | - | 300-448 tok/s |

**Tried at the same time, not kept:** 48 lanes (cache 11.3 GB): the article's best 5 s windows 224 tok/s but 55 s in
all. A 512 MB card reserve instead of 1.5 GB (cache 13.7 GB): 186 tok/s, noisier. An article keeps all lanes busy
only between its plan and its join; the code pages' many functions keep them busy for longer.

**Later the same day, Longest articles only (pinned experts):** 512-token prompt passes (`RAPIDLI_MOE_CHUNK`, cache 13.1
GB): 208 tok/s, 45 s. A quarter of the misses on the processor (`RAPIDLI_MOE_CPU_SHARE=0.25`, which gave 436 against 380
before the pinning): 167 tok/s, 50 s. With uploads at ~23 GB/s, copying a miss beats computing it on the processor. Prose
hits the cache 85-94% of the time against 96-99% for code, so articles stay near 200-230 tok/s while code pages reach
435-474; the next step would be uploads that overlap the layer's other work, not a setting.

### 30. Copying the next layer's experts early (measured, not kept; 22 Nov)

**Where an article's pass goes:** with the experts pinned (section 29), a Flash Longest article's wide passes (~28
lanes) hit the expert cache 82-85% of the time and copy 1.3-1.7 GB of missed experts a pass, 55-72 ms of a ~143 ms pass
(48 layers, 400 lookups = 8.3 passes). The copies wait for the router and the experts wait for the copies.

**Tried:** the graph also applied layer il+1's router to layer il's FFN input (top-k kept as an output); the cache read
that guess with its own selection, gave slots to the guessed experts that were not cached (none of layer il's, CRF
untouched, class 0) and copied them on a second CUDA stream while layer il computed; the next lookup waited for that
stream first. Scripts (kept outside the repository: `patch_zzmoeprefetch.py` for llama, `patch_zsidecopy.py`
for ggml-cuda); not applied by the rebuild.

| guessed per token | used guesses | hit rate while writing | articles, steady tok/s (3 runs) |
|---|---|---|---|
| none | - | 82-85% | 191, 163, 203 (avg 186) |
| top 8 | ~43% | ~93% | 120-126 (one run): the wrong guesses' copies took the PCIe link |
| top 2 | 60-75% | ~93-96% | 152, 210, 204 (avg 189) |
| top 4 | - | - | 176, 176, 250 (avg 201), 40-55 s in all |

**Why it does not pay:** a token's first picks are cached already; the misses are its lower picks, which the previous
layer predicts poorly, and the window to hide a copy is one layer's computation (~1.5 ms). The spread between runs
(163-250) is larger than any gain, so the engine was left as it was (llama.dll and ggml-cuda.dll from before, by hash).

