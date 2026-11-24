# What was measured

Measured on one RTX 4090 (24 GB) with an Intel Core i9-14900KF and 64 GB of RAM, Windows 11.

## One answer, many lanes (Qwen3.8-27B UD-Q4_K_M, 72 lanes + MTP drafting)

| Length | Tokens per second |
|---|---|
| One lane (Short), drafted by the MTP head | 86 to 90 (45 to 51 without drafting) |
| Answer (about 1,000 tokens) | 200 to 245 |
| Detailed (about 2,500 tokens) | 377 to 426 |
| Report (about 7,500 tokens) | 638 to 757 |
| Long report | 911 to 915 |
| Long report, fastest mode | 969 to 1,000 (engine peak 1,555 while every lane writes) |

- A pass over the model costs about 19 ms plus 0.33 ms per lane, so the lanes are nearly free; what limits a short
  answer is its plan, written one token per pass by one lane.
- Quality, judged blind against the ordinary answer of the same length: one-level answers 8.4 to 8.6 of 10 against
  8.6 to 8.8; two-level answers 7.8 against 8.5 (with a kind for every point).
- Many questions at once: 1,458 to 1,467 tokens/s, the same as the earlier engine (1,464 to 1,473).

## Inkling-2-Mini (mixture of experts, looped layers)

UD-Q2_K_XL (9.1 GB), engine on llama.cpp b12631 with Rapidli's patches:

| | Tokens per second |
|---|---|
| One lane (11.7 ms a token) | 82 to 85; stock llama.cpp b12631's one stream 81.5 |
| 8 lanes (20.8 ms a pass) | about 385 |
| 58 lanes, all writing (43 ms a pass) | 1,340 |
| 124 lanes, all writing | 1,497 |
| A Longest article, 7,077 tokens in 16.7 s on 83 lanes | 424 over the whole answer, peak 1,116 |
| A Long article, about 1,700 words | written in 11 s |

- Lanes fitted: 108 at an 8k conversation context, 58-59 at 32k (21 before the fitting counted the looped blocks).
- UD-Q4_K_M: 29 lanes at 8k, one lane 75 tok/s (stock 75.1), all lanes 786 tok/s, a long answer 230 tok/s.
- Before the engine work, 62 lanes wrote 231 tok/s together: every pass built the relative-position bias over the
  whole shared cache. The changes one by one are in the [CHANGELOG](../CHANGELOG.md) ("Inkling 2 at speed").
- What limits a whole answer now is its shape, not the engine: a long answer has 5 to 17 lanes writing on average
  (the plan comes first, parts end unevenly). One lane is bound by the number of kernels: a pass of one token is
  about 5,400 small kernels.
- Quality, blind review of Long articles (5 topics, two rounds of reviewers): 6.0 and 6.1 of 10, against
  Qwen3.8-35B-A3B 6.0 / 6.8 and Ternary-Bonsai-2-27B 4.4 / 4.9.

## Rapidli (the chat)

| Model | Lanes fitted | Notes |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_M | 59 + drafting (card free) | 14 lanes and spilling when another program held 5.8 GB of the card |
| Ternary-Bonsai-2-27B Q2_0 | 204 (card free), 113 (5.8 GB held) | Long report 948 tok/s on 158 lanes; with 5.8 GB held: short 50, auto 259, detailed 397, long 843 |
| Ternary-Bonsai-2-27B Q2_0, vision on, 128k context | 52 | image question 0.7 s, follow-up about the image 0.5 s, web-searched answer 2.8 s, Detailed 228 tok/s (peak 393) |
| Nemotron-3.5-Bolt-E4B BF16 (Gemma 4) | 250 | |

- **Context:** Bonsai with automatic context: 8k → 110 lanes, 16k → 101, 32k → 85, 64k → 52 (vision off). A
  38,000-token conversation is read once in 14.2 s; the next turn extends it in 0.04 s. 48 lanes answering over the
  whole conversation: 65.6 ms a pass against 29.1 ms without it. Follow-ups show their first text after 2.8 s, then
  0.22 s.
- **Web search:** 5 DuckDuckGo results in about 0.8 s; three pages read in parallel in about 2 s.
- **Vision:** a small picture is 93 tokens and is read in 0.56 s by Bonsai-2 with its BF16 projector.
