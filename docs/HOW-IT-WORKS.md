# How Rapidli works

Three parts: the **engine** (`engine\bin\rapidli-engine.exe`, C++ on a patched llama.cpp), the **answerer**
(`app\fan.py`), and the **app** (`app\rapidli.py` and the page `app\web\index.html`).

## The engine

The engine keeps a fixed number of **lanes**: sequences that are decoded together in one pass over the model. Reading
the weights from graphics memory is what a pass costs, and it is paid once for all lanes, so 64 lanes write nearly 64
times as many tokens per second as one.

- **Sessions (the vault).** `--vault N` reserves N extra sequences as parking places. A request with `"hold": true`
  parks its state when it ends; `"name"` lets a later request find it again without reading anything. A request with
  `"session": k` starts from a copy of that state. Copies are cheap: `llama_fq_fork` (our patch) tags the attention
  cells of the source with the new lanes (no data moves) and copies the recurrent state rows on the card. For models
  whose memory is not hybrid (e.g. Gemma 4 with sliding-window attention) it falls back to `llama_memory_seq_cp`.
- **Fork.** `POST /fork` reads an optional prompt once, then starts every branch from that state with its own forced
  text (`[3] Battery costs`). The branches stream back together.
- **Conversation context.** `--ctx-extra N` adds N cells that a parked conversation may fill. A lane forked from it
  reads it through shared cells, so 48 lanes can answer over the same 38,000-token conversation without 48 copies.
- **Drafting.** For models with a multi-token-prediction head (Qwen 3.6 and later), `--spec 3` adds a second small
  context pair. One-lane answers are drafted three tokens ahead by the MTP head and checked by the full model in one
  pass (about 86 instead of 51 tok/s). Its state is 16-bit (`LLAMA_RS_SPEC_F16`); its passes and the lanes' passes take
  turns. Two parked prefixes (e.g. the layout text) are kept so a drafted outline does not read them again.
- **Vision.** `--mmproj <file>` loads the model's vision projector with llama.cpp's mtmd library. `POST /media` reads
  text + image + text after a parked session into a new parked session (on the engine's run thread, between passes).
- **Packing.** Lanes that finish early are moved down so the pass narrows at once.
- **What it shows.** `/lanes` (what every lane is doing, tokens written, drafting), `/stats` (model, lanes, vision,
  context, drafting counters).

## Fitting the lanes (`app\fit.py`)

1. Start with 8 lanes. Read from the engine log what the model, the recurrent state and the attention cache take, and
   from `nvidia-smi` what the card has left.
2. Work out a lane's cost and how many lanes fit next to everything else, minus a margin (1,200 MB by default).
   With the context on *Automatic*, take the largest context (8k to 256k) that still leaves 48 lanes or more.
3. Start with that many lanes, run all of them over a long prompt (so every buffer is really used), and time a pass of
   8 lanes against the probe. Slower means memory spilled into the PC: 15% fewer lanes, again. Drafting is left off
   when its context reads in more than 3 seconds.
4. Remember the result in `data\fit.json` (per model, context, drafting, vision). The next load starts with it and only
   checks; when the card has clearly more room than then, it fits again.

## The answerer (`app\fan.py`)

1. **The layout is read once** and parked: a system text describing an answer (outline, parts, bottom line). A
   conversation is parked after it (`conv-<hash>`), and the next turn extends it by the new text only.
2. **The question is read** and parked. A **router lane** writes its first word: DIRECT (a fact, a sum, code, a poem,
   steps that depend on each other) or OUTLINE. Beside it a **brief lane** writes a one-sentence answer.
3. DIRECT: answered by one lane, drafted by the MTP head when the model has one.
   OUTLINE: the planner writes numbered part titles and stops at `===`. Some models (seen with Bonsai-2 Q2_0) leave
   that line out and go straight on with `[END]` or `[1]`; the planner also stops there, its parked state (which holds
   part of that line) is released, and the part lanes read the titles again from the question's state, the same way a
   drafted outline is read. Without this the planner lane wrote the whole answer alone.
4. **Fork**: one lane per part, each given its tag and title, writes its part; one more writes the bottom line. Every
   lane has read the question and the whole outline.
5. **Detailed and longer**: the part lanes first write points; the whole plan is read once and forked again, one lane
   per paragraph.
6. A part is asked to be a quarter longer than it may get and ends with the sentence in which it reaches its length, so
   the lanes stop together. Sentences that repeat an earlier one are removed.
7. **Images** are markers in the conversation text (`\x01IMG:<id>\x01`); `read_into` reads text with `/completion` and
   every image with `/media`, chaining parked states. With images the plain one-lane path answers inside the parked
   conversation (drafting reads plain text only). That state holds the layout text, so such an answer is given the
   stop strings `\n[END]`, `\n[1]` and `\n===`: without them a model can run on into the layout's "[END] Bottom line"
   section. The engine streams text before it has matched a stop string, so the page hides a trailing partial marker
   ("[E", "[END", "==") while it paints; the stored answer is cut by the engine.
8. **Instructions** from Settings › Chat are added to the layout text and, for one-lane answers, given as a system turn.

## Stories (`Fan._story` in `app\fan.py`)

A story's lines follow from one another, so splitting it into parts that share only a list of titles does not work: the
wolf was "Wren" in the opening and "Barnaby" in part one. What the scenes have to agree on can be fixed first, though.

1. **The construct**, by one lane: `TITLE`, `VOICE` (point of view, tense, who it is for), `SETTING` (where, when, how
   time moves), `CAST` (names with a description each, "these names and no others"), `RETURNS` (an object, a sound or a
   phrase that comes back), and the `ARC`. The title is shown the moment it is written.
2. **The arc.** Up to 8 scenes, each line of the arc is a scene: what happens, and `ENDS WITH`, the exact moment it
   stops. That moment is where the next scene begins; one lane planning all of them is what makes the scenes join.
   A longer story's arc is its chapters (about three scenes each, twelve chapters at most), and a lane per chapter then
   plans that chapter's scenes, all chapters at once. Each of those lanes starts from its own chapter's line; with only
   the whole arc in front of it, the third chapter's lane planned the second chapter again.
3. **The scenes.** Every scene is written by a lane of its own, all starting from the whole plan (read once). A lane
   holds about 300 tokens, so a scene is about 170 words: a longer story is more scenes, not longer ones.
4. **Repairs.** A chapter lane that wrote fewer scene lines than asked, and a scene lane that wrote only its plan line,
   are run once more with sampling instead of the likeliest text. A scene cut by its length ends at its last whole
   sentence.

`sizing.story_scenes` decides which requests are written this way and with how many scenes (6 by default, 4 for "a
short story", 12 for "a long story", 12 / 25 / 40 for Long / Extra long / Longest, or the words asked for divided by
170). A poem, a song, a script, a story of a few sentences, and the next part of a story already in the chat are one
lane's work. If no plan can be read, one lane writes the story.

Measured: 5,890 words in 23 s with 34 scenes (Ternary-Bonsai-2-27B Q2_0, RTX 4090), where one lane writes 44 to 60
tokens a second. The price: lanes cannot read each other, so two neighbouring scenes can open with a similar image.

## The app (`app\rapidli.py`)

- `models.py` scans the model folders (GGUF header only, cached in `data\models.json`), drops non-chat files
  (projectors, drafts, embeddings), and finds each model's vision projector in its folder.
- `sizing.py` picks the length; `plan_for` turns it into parts, points and words for the number of lanes there are.
- A conversation that no longer fits the context drops its oldest turns (the page says how many).
- **Web search** (`search.py`): one model lane writes the query from the conversation (a second one only in the
  subject's own language). The engines of `ddgs` are asked one at a time, the ones that answered lately first; the
  next one is asked as well after 1.5 s, at most four requests at once, and an engine that fails twice rests for a few
  minutes (Settings › Web search shows them). Results of several engines are merged by rank, then DuckDuckGo's page
  and Wikipedia are the fallbacks; results and pages are kept for 30 minutes. The best pages are read in parallel
  (two more than are needed, videos and social networks skipped), cut into passages of about 600 characters, and the
  passages of all of them are ranked together (BM25, the engine's snippet as a pointer; `search.pick`), the best kept
  in the room the settings give. The results go in front of the question with an instruction to cite them by number;
  the sources are kept with the answer and listed at the end of a long one.
- **Offline Wikipedia** (`kiwix.py`): the model names the articles to read; the question is searched without what it
  says about the answer's form; the passages of the articles found are ranked together like the web's, each article's
  opening kept.
- **Tidy-up of a planned answer** (`lint.py`, `fan.polish`): copied plan notes and source passages, sentence loops,
  repeated and reworded paragraphs, a heading left empty are taken out; a paragraph that opens with "It" or "This" is
  written again to name its subject.
- **Run** (the page): code in an answer runs in a sandboxed frame beside the conversation; the server refuses requests
  whose Origin is not its own page, so the frame cannot change anything. The frame is loaded from the app's other
  loopback name (`/run-frame` on localhost when the app is on 127.0.0.1, and back), which the browser keeps in a process
  of its own, and is sent the page by a message: a page that never stops working freezes only itself. Python runs
  through Pyodide, SQL through sql.js (SQLite), React and TypeScript through Babel, diagrams through Mermaid, all from a
  CDN; a CSV block and a table of figures in an answer are drawn with Chart.js.
- **Versions**: writing an answer again stores the old answer and the messages after it in `versions`; switching a
  version swaps them back.
- **Images** are stored by content hash in `data\images`, shrunk in the browser to 1,600 pixels at most.
- Load requests and context changes are written to `data\logs\rapidli.log` (with the page that asked). A page closed
  or reloaded while it is being answered stops that answer; the aborted connection is not logged as an error.
- The page polls `/api/lanes` three times a second while an answer is written or the engine panel is open, and
  `/api/system` every four seconds while the panel or the settings are open.
