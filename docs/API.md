# HTTP API

Rapidli answers other programs over HTTP: load and unload models, and ask questions with the same choices as the
composer (length, thinking, offline Wikipedia, web search, deep research). There is Rapidli's own route and an
OpenAI-style chat route that many tools speak. For programs that send many related requests (live copilots, agents,
document tools) there are plain replies, contexts read once and used by many requests, priorities, batches and JSON
replies: see [Many related requests](#many-related-requests).

Contents: [Switching it on](#switching-it-on) · [Models](#models) · [Asking](#asking) · [Answer options](#answer-options) ·
[OpenAI-style](#openai-style) · [Many related requests](#many-related-requests) ([plain mode](#plain-mode),
[contexts](#contexts), [priority](#priority), [batches](#batches), [JSON replies](#json-replies)) ·
[Status and options](#status-and-options) · [Errors](#errors) · [Examples](#examples) ·
[The app's own routes](#the-apps-own-routes-port-11842) · [The engine](#the-engine-port-11832)

## Switching it on

The API is off until it is switched on in **Settings › API**, which also shows the key.

- **Key required** (on by default): every request sends `Authorization: Bearer <key>`, or `X-API-Key: <key>`.
  A new key stops the old one at once.
- **Other computers**: off by default. When it is on (and Rapidli has been started again), the API's paths, and
  nothing else of the app, answer other computers on your network, always with the key. The firewall may need to let
  port 11842 through.
- **Several requests at once**: side by side (written at once, on the lanes that are free) or one after another
  (each waits for the one before it; a streamed request is told its place in line).
- **Keep the API's chats**: the API's conversations appear in the chat list, in a folder called API. (Plain
  requests are not kept unless they ask with `"store": true`.)

The base address is `http://127.0.0.1:11842` (from another computer, this PC's address with the same port).

| Method and path | What it does |
|---|---|
| `GET /api/v1/models` | every model on this PC |
| `POST /api/v1/models/load` | loads a model (with options for this load: lanes, context, margin, vision, drafting) |
| `POST /api/v1/models/unload` | unloads it: the graphics card is free |
| `GET /api/v1/status` | the model, the lanes (busy and free), the context, the parked-state places, the card's memory, the load's progress |
| `GET /api/v1/status/events` | the same as server-sent events, sent again whenever it changes |
| `GET /api/v1/options` | what a question can be asked with now |
| `POST /api/v1/ask` | asks a question (Rapidli's own route) |
| `POST /v1/chat/completions` | asks a question (OpenAI-style); `"mode": "plain"` for one lane and the messages as they are |
| `POST /api/v1/contexts` | reads text once into a context that many requests start from |
| `POST /api/v1/contexts/{id}/append` | adds text (or messages) to a context |
| `POST /api/v1/contexts/{id}/fork` | a second context from the same text, nothing read again |
| `GET /api/v1/contexts`, `GET /api/v1/contexts/{id}` | the contexts, or one of them |
| `DELETE /api/v1/contexts/{id}` | lets a context go |
| `POST /api/v1/batch` | many replies from one context, on one connection |
| `POST /api/v1/batch/{batch}/cancel` | cancels one reply of a batch by its id (or all of them) |
| `GET /v1/models` | the model loaded now, as OpenAI lists models |

## Models

### `GET /api/v1/models`

Every chat model Rapidli found on this PC (its own folder, the Hugging Face cache, LM Studio, Ollama, Jan, GPT4All,
llama.cpp, and the folders added in Settings › Models).

```json
{"models": [
  {"id": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF:Qwen3.8-35B-A3B-IQ3_M", "label": "Qwen3.8-35B-A3B-IQ3_M",
   "repo": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF", "gb": 15.2, "arch": "qwen35moe", "context": 262144,
   "vision": false, "mtp": true, "source": "Rapidli", "loaded": true, "loading": false}
]}
```

`vision`: the model sees images (its projector is next to it). `mtp`: it drafts tokens ahead with its own MTP head.
`loaded` / `loading`: the model in use now.

### `POST /api/v1/models/load`

| Field | |
|---|---|
| `model` | the `id` or the `label` of a model from `GET /api/v1/models` (required) |
| `wait` | `true`: answers when the model is ready. Without it, the answer comes at once (`202`, `"status": "loading"`) and `GET /api/v1/status` tells when the model is `ready` |
| `refit` | `true`: fits the lanes to the card again (when other programs have taken or freed memory on it) |
| `lanes` | the most lanes to start (1 to 256; at least 8 are started). Rapidli still fits to the card: fewer start when fewer fit |
| `context` | tokens of conversation context (`2048` or more), or `"auto"`: the parked conversations and the API's contexts live in it |
| `margin_mb` | graphics memory left free beside the lanes, in MB |
| `vision` | `false`: the vision projector is not loaded (about 0.9 GB more for lanes); `true`: loaded when the model has one |
| `drafting` | `false`: no drafting by the model's MTP head |

`lanes`, `context`, `margin_mb`, `vision` and `drafting` are for this load only: the settings are not changed. A refit
or a reload after an engine crash keeps them; a load chosen in the app uses the settings again. A fit with a lane
maximum is kept apart from the model's own fit. The model already loaded with other options is loaded again.
`GET /api/v1/status` shows them under `load_options`.

Loading fits as many lanes as the card has room for; the fit of a model is kept, so a model loads faster the second
time. A big model can take minutes. Answers being written when another model is loaded are stopped.

```json
{"status": "ready", "model": {"id": "…", "label": "Qwen3.8-35B-A3B-IQ3_M", "loaded": true}, "lanes": 115}
```

The model already loaded answers `{"status": "ready", "already": true, …}` at once. A model that is being loaded
answers `409`; an unknown model `404`; a model that failed to load `500` with the reason.

### `POST /api/v1/models/unload`

Stops the engine: the graphics card's memory is free again. `{"wait": true}` answers when it is done (a few
seconds); without it, `202` with `"status": "unloading"`. With no model loaded: `{"status": "idle", "already": true}`.

## Asking

### `POST /api/v1/ask`

| Field | |
|---|---|
| `question` | the question (required) |
| `chat` | the `chat` of an earlier answer: the question goes on in that conversation |
| `size`, `think`, `wiki`, `web`, `first`, `deep` | the [answer options](#answer-options) |
| `stream` | `true`: the app's own events as they come ([events](#events-of-apisend)) |

```json
{"chat": "3f9c1a2b7d4e", "answer": "…the answer, in Markdown…",
 "stats": {"route": "fan", "size": "detailed", "seconds": 9.1, "tokens": 3140, "tokens_per_second": 1480, "lanes": 14, "first_text_s": 0.6},
 "search": [{"title": "Heat pump", "url": "/kiwix/…", "where": "wiki", "query": "heat pump cold climate"}]}
```

`stats.route`: how it was written: by one lane (`direct`, `serial`; `think` when it thought first), planned and written by many lanes at once (names
starting with `fan`, such as `fan-deep`), a `story`, or a page of code written function by function (`code-lanes`).
`stats.tokens_per_second` is the engine's own timing of the answer, as the app shows it. `search`: the sources the answer
was written from. A stopped answer has `"stopped": true`.

## Answer options

Taken by `/api/v1/ask` and by `/v1/chat/completions`. All are optional. `GET /api/v1/options` lists what can be
chosen now.

| Option | Values | Default |
|---|---|---|
| `size` | `auto` · `short` · `answer` · `detailed` · `report` · `long` | `auto` |
| `think` | `off` · `parts` · the model's own levels (`low`, `medium`, `high`, or `on`) · `true` | `off` |
| `wiki` | `true` / `false`: offline Wikipedia, when installed | `true` |
| `web` | `true` / `false`: web search | `false` |
| `first` | `wiki` · `web`: which is asked first when both are on | `wiki` |
| `deep` | `true` / `false`: deep research | `false` |

**Length (`size`)**, as the composer names it:

| Value | In the app | |
|---|---|---|
| `auto` | Auto length | the length follows the question |
| `short` | Short | one lane, a few sentences |
| `answer` (or `medium`) | Medium | 4 to 8 parts written side by side |
| `detailed` | Long | parts with paragraphs, about 2,000 words |
| `report` (or `extra_long`) | Extra long | about 6,000 words |
| `long` (or `longest`) | Longest | as long as it gets: every lane at work |

**Thinking (`think`)**: what the loaded model offers is in `GET /api/v1/options` and `GET /api/v1/status`.

| Value | In the app | |
|---|---|---|
| `off` | Thinking off | answers at once, fastest |
| `parts` | Think per part | Medium and Long: each part's lane thinks about its own section, side by side; other lengths think briefly on one lane. Works with any model |
| `low` | Think briefly | up to about 1,000 tokens of thinking (models with levels) |
| `medium` | Think | up to about 3,000 tokens |
| `high` | Think hard | up to about 8,000 tokens, the slowest |
| `on` | Thinking on | up to about 4,000 tokens (models that think or not, without levels) |
| `true` | | the model's usual level |

A level the model does not have is refused (`400`, with the levels it has). With thinking and `size` `auto`, one lane
thinks and answers, as a reasoning model does; a length asked for is still planned and written by many lanes.

**Looking things up**: `wiki` checks facts in offline Wikipedia on this PC (Settings › Offline Wikipedia installs
it; without it, `wiki` does nothing). `web` searches the web and reads the best pages; the answer cites them. With
both, `first` says which is asked first. `deep` researches in several rounds (the question's parts first, then what
is still missing) in the sources that are on, and answers as a report unless a `size` is given; it needs `web` or
offline Wikipedia. The old names `search` (for `web`) and `kiwix` (for `wiki`) are still read.

A question that asks to see something ("show me the cover of…") with `web` on gets pictures: they are kept with the
chat, and the answer says what they show.

## OpenAI-style

`POST /v1/chat/completions` with OpenAI's body: base URL `http://127.0.0.1:11842/v1` and the key as the API key.

- The model loaded now answers, whatever `model` is sent (load another with `/api/v1/models/load`).
- The messages before the last are the conversation; the last must be the user's. System (and developer) messages
  are followed as instructions for the answer. Text parts of a message are read; images are not (they are in
  [plain mode](#plain-mode)).
- `"mode": "plain"` answers the messages as they are, on one lane, without the planner: see
  [plain mode](#plain-mode). A request with a `context` or a JSON `response_format` is a plain one.
- The [answer options](#answer-options) can be added to the body; `reasoning_effort` (`none`, `low`, `medium`,
  `high`) is read as `think` when `think` is not given.
- `"stream": true`: `chat.completion.chunk` lines and `data: [DONE]`. A one-lane answer comes word by word; a
  many-lane answer comes whole at its end (its parts are written at once, not in order). While a request waits for
  its turn, SSE comments keep the connection open.

```json
{"id": "chatcmpl-…", "object": "chat.completion", "created": 1795000000, "model": "…",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 12, "completion_tokens": 640, "total_tokens": 652},
 "rapidli": {"chat": "…", "route": "fan", "seconds": 1.9, "tokens": 640, "tokens_per_second": 610, "lanes": 7}}
```

`GET /v1/models` lists the model loaded now (an empty list when none is).

## Many related requests

For programs that send many requests about the same text, often at once: a live copilot that asks about a meeting
transcript as it grows, an agent that runs many steps over one set of instructions and tools, a document tool that asks
twenty questions of one long document. Four things make that fast:

- **[Plain mode](#plain-mode)**: the messages are answered as they are, in the model's own chat format, on one lane,
  streamed token by token. No planner, no added notes, no look-ups, no layout: what you send is what the model reads.
- **[Contexts](#contexts)**: a long text (a system prompt, a document, a transcript) is read once and kept in the
  engine. Every request that names the context starts from it and reads only its own new messages.
- **[Priority](#priority)**: the prompt of a high-priority request is read first; within one priority, one prompt is
  read to its end before the next.
- **[Batches](#batches)**: many replies from one context on one connection, each cancellable by its id.

And [JSON replies](#json-replies) (`response_format`) for replies that a program parses.

### Plain mode

`POST /v1/chat/completions` with `"mode": "plain"`:

```json
{"mode": "plain", "stream": true, "max_tokens": 300, "temperature": 0.7, "top_p": 0.9, "seed": 7, "stop": ["\n\n"],
 "priority": "high",
 "messages": [{"role": "system", "content": "You are a meeting assistant. ..."},
              {"role": "user", "content": "What did Alice decide?"}]}
```

- **The system prompt, whole.** All `system` and `developer` messages, uncut and in order, become one system turn at
  the front, in the model's own format (ChatML for Qwen and Bonsai, Gemma 4's turns, Inkling's messages). `user`,
  `assistant` and `tool` messages follow as turns of their own. The reply is written without thinking.
- **One lane per request**, streamed token by token. Many plain requests at once write side by side on the free
  lanes. They are **not held back by the answer limit** (`parallel_max`, "Several requests at once" in Settings › API):
  their own limit is `plain_max` in `data/settings.json`'s `api` section (`0`, the default: none; extra requests
  wait for a free lane in the engine).
- **Sampling**: `max_tokens` (or `max_completion_tokens`), `temperature` (default `0`: the most likely token;
  OpenAI's default would be 1), `top_p`, `min_p` (default `0.05`), `seed` (with a seed and a temperature, the same
  request gives the same reply), `stop` (a string or up to 8). `n` must be 1 (a [batch](#batches) gives several).
  `top_p` is applied on the card when the status says `"top_p": true` (an engine built with `patch_ztopp`, see
  [docs/BUILDING.md](BUILDING.md)); with an older engine it is accepted and not applied.
- **Longer than a lane.** A lane holds `context.lane` tokens (in the status). A longer reply goes on from the lane's
  parked state on a new lane, seamlessly, up to `max_tokens` (default: up to 8,192, or what is left of the context
  limit). A prompt longer than a lane is first read into a parked state of its own.
- **finish_reason**: `"stop"` (the model ended it, or a `stop` string), `"length"` (cut at `max_tokens`).
  **usage** has the real counts: `prompt_tokens` is everything the model read for this reply (the context's tokens
  plus the new ones), `completion_tokens` what it wrote.
- **Not saved as a chat**: `"store": false` is the default; `"store": true` keeps the request and its reply in the
  chat list (the API folder).
- **Failures** come as an error event (`data: {"error": {...}}`, then `data: [DONE]`), or as an HTTP error when not
  streamed; never as answer text.
- **Closing the connection** frees the lane at once, also while the prompt is still being read.
- **Images**: `image_url` parts (a `data:` URL or an http(s) address) are read when the model sees images
  (`"vision": true` in the status).
- With `"stream_options": {"include_usage": true}` a last chunk with empty `choices` carries the usage too (it is
  always on the chunk with the `finish_reason`).

### Contexts

A context is text read once by the engine and kept there as a parked state. Any number of requests can start from it
at the same time: each one copies the state (on the card, in milliseconds) and reads only its own new messages.

```
POST   /api/v1/contexts              {"messages": [...], "ttl_s": 600}      -> {"id", "tokens", "limit", ...}
POST   /api/v1/contexts/{id}/append  {"text": "..."}  or  {"messages": [...]} -> {"id", "tokens", "limit", ...}
POST   /api/v1/contexts/{id}/fork    {}                                      -> a new {"id", ...}
GET    /api/v1/contexts/{id}                                                 -> {"id", "tokens", "limit", "expires_in_s", ...}
GET    /api/v1/contexts                                                      -> {"data": [every context]}
DELETE /api/v1/contexts/{id}                                                 -> {"id", "deleted": true}
```

- **Using one**: `POST /v1/chat/completions` (or a [batch](#batches)) with `"context": "<id>"` and the new
  `messages` only. The reply starts from the context's state, reads only the new messages, and leaves the context as
  it was. (A request with a context is a [plain](#plain-mode) one.)
- **The last message stays open.** Text appended with `{"text": ...}` goes into the context's last message: make the
  last message the growing part (a transcript, a log, a document being written). `{"messages": [...]}` adds whole
  messages after it instead. A request first closes the open message, then adds its own.
- **Append makes a new state** from the old one; only the appended text is read. Requests already running keep the
  state they started from. Appending is the only change: to change earlier text, build a new context.
- **Full**: a context holds `limit` tokens (the model's conversation context, `context.limit` in the status). An
  append that would pass it answers `413`: build a new context (for a transcript, from a summary and its latest part).
- **Fork** gives a second context on the same state, at once (nothing is read). Appending to one of them then leaves
  the other as it was.
- **Lifetime**: a context lives `ttl_s` seconds after its last use (default 600, at most 86,400; `0`: until it is
  deleted or the model is loaded again). Every request that names it, and every GET, counts as a use.
- **Owned by the app**: Rapidli's own clean-up never lets a context's state go. A context that expired, was deleted,
  or went with a model load or a refit (the engine starts again then) answers `404`: the client builds it again.
- **How many**: the engine's parked-state places are shared with the app's own conversations;
  `context.contexts_max` in the status is the most contexts at once (`contexts_max` in the settings' `api` section
  sets it). One more answers `429`.

```sh
R=http://127.0.0.1:11842; AUTH="Authorization: Bearer $KEY"; JSON="Content-Type: application/json"
# a transcript read once, then questions about it as it grows
curl -s $R/api/v1/contexts -H "$AUTH" -H "$JSON" -d '{"messages": [
  {"role": "system", "content": "You answer questions about the meeting transcript below. Be brief."},
  {"role": "user", "content": "Transcript:\nAlice: We ship on Friday.\n"}]}'
# -> {"id": "ctx-3f2a…", "object": "context", "tokens": 41, "limit": 32768, ...}
curl -s $R/api/v1/contexts/ctx-3f2a…/append -H "$AUTH" -H "$JSON" -d '{"text": "Bob: Then QA needs the build by Wednesday.\n"}'
curl -sN $R/v1/chat/completions -H "$AUTH" -H "$JSON" -d '{"context": "ctx-3f2a…", "stream": true,
  "messages": [{"role": "user", "content": "When must QA have the build?"}]}'
```

### Priority

`"priority": "high" | "normal" | "low"` on `/v1/chat/completions` (plain), a batch (for all its requests, or per
request) and the context routes (for reading the text). Long prompts are read in passes of their own, one prompt at a
time between the passes that write:

- the waiting prompt of the **highest priority** is read first;
- within one priority, a prompt is **read to its end before the next** (first come, first read). The prompts used to
  take turns, so four 2,400-token prompts sent together all started writing at about 2.8 s, where one alone starts
  at 0.7 s; read one after another, the first can start writing as soon as its own prompt is read.
- requests waiting for a free lane are let in by priority too.

A request without a priority is `normal`, as are the app's own.

### Batches

Many replies from one context, on one connection:

```json
POST /api/v1/batch
{"context": "ctx-3f2a…", "priority": "normal",
 "requests": [{"id": "summary",  "messages": [{"role": "user", "content": "Summarize it."}], "max_tokens": 200},
              {"id": "actions",  "messages": [{"role": "user", "content": "List the action items."}]},
              {"id": "decision", "messages": [{"role": "user", "content": "What was decided?"}], "temperature": 0.3, "stop": ["\n\n"]}]}
```

Answered with server-sent events: first `{"batch": "batch-…", "requests": 3, "context": "ctx-3f2a…"}`, then, mixed as
they are written, `{"id": "summary", "delta": "…"}`, and for each request as it ends
`{"id": "summary", "done": true, "finish_reason": "stop", "usage": {...}}` (or `{"id", "done": true, "error": {...}}`),
then `data: [DONE]`.

- Each request has the [plain](#plain-mode) options (`max_tokens`, `temperature`, `top_p`, `min_p`, `seed`, `stop`,
  `priority`, `response_format`) and its own `messages` after the context. Without a `context`, each request's
  `messages` are the whole conversation. 1 to 256 requests; ids are strings (the index when none is given).
- **Cancel one**: `POST /api/v1/batch/{batch}/cancel {"id": "actions"}` (or `{"ids": [...]}`, or `{}` for all). Its
  lane is freed at once; it ends with `"finish_reason": "cancelled"`. Closing the connection cancels them all.
- A batch counts as one plain request against `plain_max`.

### JSON replies

`"response_format": {"type": "json_object"}`, or `{"type": "json_schema", "json_schema": {"name": "…", "schema": {…}}}`
as OpenAI defines them. The model is asked for one JSON object (with the schema, when given), its reply is opened with
`{` and ends as soon as the object closes. It is then checked: that it parses and, for `json_schema`, that it matches
the schema (`type`, `properties`, `required`, `additionalProperties: false`, `items`, `enum`, `const`,
`minItems`/`maxItems`, `minLength`/`maxLength`, `minimum`/`maximum`, `anyOf`/`oneOf`). A reply that fails is written
again (twice at most, at a little temperature); after three tries the request answers `422` with the reason.
This is checked as it is written, not enforced token by token by a grammar, so a streamed JSON reply is sent whole
once it has passed.

## Status and options

### `GET /api/v1/status`

```json
{"status": "ready", "message": "…", "model": "…", "label": "Inkling-2-Mini-UD-Q2_K_XL", "lanes": 108,
 "think": ["off", "parts"], "answers": {"running": […], "waiting": […]}, "together": "parallel",
 "lanes_busy": 12, "lanes_free": 96, "queued": 0,
 "context": {"limit": 32768, "lane": 512, "contexts": 2, "contexts_max": 5, "contexts_tokens": 6100},
 "context_places": {"slots": 11, "used": 6, "free": 5},
 "vision": false, "template": "inkling", "top_p": true, "plain_mode": true, "plain": {"running": 3, "max": 0},
 "gpu": {"name": "NVIDIA GeForce RTX 4090", "total_mb": 24564, "used_mb": 22900, "free_mb": 1664, "model_mb": 8500, "per_lane_mb": 61},
 "load": null, "load_options": {"lanes": 64}}
```

- `status`: `idle` (no model), `loading`, `ready`, or `error` (`message` says why).
- `lanes_busy`, `lanes_free`: lanes with a request now, and without; `queued`: requests waiting for a lane.
- `context.limit`: the most tokens a context, or one request's text, can hold; `context.lane`: what one lane reads
  in one go (longer text is read into a parked state first, by itself). `contexts`, `contexts_max`,
  `contexts_tokens`: the API's contexts now, the most there can be, the tokens they hold.
- `context_places`: the engine's places for parked states (the app's own conversations and layouts use some).
- `vision`: images can be read; `template`: the model's chat format (`chatml`, `gemma4`, `inkling`); `top_p`: the
  engine applies `top_p` (with an older `llama.dll` it is accepted and not applied).
- `plain_mode`: the engine can write plain replies (version 5; an older engine answers them with `501`). `plain`: plain
  requests and batches being written, and their limit (`0`: none of their own).
- `gpu`: the card's memory as measured now, the model's share and a lane's share when it was fitted.
- `load`: while a model loads, `{"seconds", "expected_s", "fraction", "phase", "message"}`.

### `GET /api/v1/status/events`

The same status as server-sent events: one at once, then a new one whenever something in it changes (looked at twice
a second: the model, the lanes busy, the contexts, the load's progress, the card's memory by 256 MB). A comment line
(`: ping`) every 15 seconds keeps the connection open. Close it to stop.

### `GET /api/v1/options`

```json
{"model": "…", "status": "ready",
 "size": [{"value": "auto", "name": "Auto length", "what": "the length follows the question"}, …],
 "think": [{"value": "off", "name": "Thinking off", "what": "answers at once, fastest"}, …],
 "wiki": {"available": true, "files": ["wikipedia_en_all_nopic.zim"]}, "web": {"available": true},
 "first": ["wiki", "web"], "deep": {"available": true, "needs": "web or wiki"}}
```

`think` is empty while no model is ready.

## Errors

Errors are `{"error": {"message", "type": "rapidli", "code"}}` with the HTTP status:

| Code | When |
|---|---|
| `400` | a wrong option (the message lists the values that can be given), an empty question |
| `401` | no key, or a wrong one |
| `403` | the API is switched off, or another computer asks while "Other computers" is off |
| `404` | no such model, or no such path; a context that expired, was deleted or went with a model load (build it again) |
| `409` | a model is being loaded; or an answer is being written in that chat |
| `413` | the text is longer than a context holds (`context.limit` in the status): build a new context |
| `422` | a JSON reply did not parse, or did not match its schema, in three tries |
| `429` | no room for another context: delete one first |
| `501` | the engine is older than the app: plain replies need rapidli-engine version 5 (`"plain_mode": false` in the status) |
| `502` | the engine could not read or write it |
| `503` | no model is ready: load one first; or a load is under way |
| `507` | the engine has no free place for a parked state |

A streamed reply that fails after it started ends with an error event, `data: {"error": {"message", "type", "code"}}`,
then `data: [DONE]`: an error is never sent as answer text.

## Examples

Load a model, ask a long question with offline Wikipedia and the web, then unload (`KEY` is the key from
Settings › API):

```sh
curl http://127.0.0.1:11842/api/v1/models/load -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"model": "Qwen3.8-35B-A3B-IQ3_M", "wait": true}'

curl http://127.0.0.1:11842/api/v1/ask -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How do heat pumps work in cold climates?", "size": "detailed", "think": "parts", "wiki": true, "web": true}'

curl http://127.0.0.1:11842/api/v1/ask -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"question": "Compare solid-state and lithium-ion batteries.", "deep": true, "web": true}'

curl http://127.0.0.1:11842/api/v1/models/unload -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"wait": true}'
```

Python, with nothing but the standard library:

```python
import json, urllib.request

BASE, KEY = "http://127.0.0.1:11842", "<key>"

def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

print([m["label"] for m in call("/api/v1/models")["models"]])
call("/api/v1/models/load", {"model": "Qwen3.8-35B-A3B-IQ3_M", "wait": True})
print(call("/api/v1/options")["think"])
r = call("/api/v1/ask", {"question": "Explain how a heat pump works.", "size": "answer", "think": "parts"})
print(r["stats"]["tokens_per_second"], "tok/s")
print(r["answer"])
r = call("/api/v1/ask", {"question": "And in very cold weather?", "chat": r["chat"]})    # the same conversation
```

With the OpenAI Python library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11842/v1", api_key="<key>")
r = client.chat.completions.create(model="rapidli", messages=[{"role": "user", "content": "Write a long article about heat pumps."}],
                                   extra_body={"size": "long", "web": True})
print(r.choices[0].message.content)
```

## The app's own routes (port 11842)

What the page itself uses, on this PC only (not reachable from other computers, and not with the API's key).

| Method and path | Body / query | Answer |
|---|---|---|
| `GET /` | | the page |
| `GET /api/status` | | `status` (idle, loading, ready, error), `message`, `model`, `lanes`, `spec`, `vision`, `can_see`, `ctx_limit`, `ctx_setting`, `ctx_options`, `fit` |
| `GET /api/models[?refresh=1]` | | models found: `id`, `label`, `repo`, `gb`, `arch`, `mtp`, `vision`, `context`, `current`, `loaded`, `source` (where it was found); also `places` (the folders looked through, with counts) and `folders` (the ones you added) |
| `POST /api/model` | `{"id", "refit"?}` | loads a model (fits the lanes; `refit` fits again) |
| `POST /api/unload` | `{}` | stops the engine |
| `POST /api/context` | `{"ctx": "auto" or tokens}` | sets the conversation context of the current model and reloads it |
| `GET /api/system[?programs=1]` | | card (`gpu`), PC memory (`ram`), `disk`, `engine` (model, lanes and context memory), `programs` using the card |
| `GET /api/settings`, `POST /api/settings` | `{"changes": {section: {key: value}}}` | the settings (the Hugging Face token is never sent back) |
| `GET /api/chats` | | conversations, newest first |
| `GET /api/chats/<id>` | | a conversation with its `context` (`used`, `limit`) |
| `DELETE /api/chats/<id>` | | deletes it |
| `POST /api/chats/<id>/rename` | `{"title"}` | |
| `POST /api/chats/<id>/truncate` | `{"keep": n}` | keeps the first n messages |
| `POST /api/chats/<id>/remove` | `{"index"}` | removes one message |
| `POST /api/chats/<id>/version` | `{"index", "version"}` | shows another version of an answer (with what followed it) |
| `POST /api/upload` | `{"data": "data:image/...;base64,..."}` | `{"id"}` of the stored image |
| `GET /api/images/<id>` | | the image |
| `POST /api/send` | `{"chat"?, "run"?, "text", "size"?, "images"?: [id], "search"?: bool, "kiwix"?: bool, "first"?, "think"?, "deep"?: bool, "regenerate"?: bool, "index"?}` | an event stream (below). Several chats can be answered at once (Settings › Chat: side by side, or one after another); a second answer in the same chat is refused with 409. `run`: an id of the caller's choosing, for `/api/stop` |
| `POST /api/stop` | `{"run"?, "chat"?}` | stops that run's or chat's answer, written or waiting; with neither, every answer |
| `GET /api/folders` | | the chat list's folders: `[{id, name, open}]`, in order |
| `POST /api/folders` | `{"add": name}`, `{"rename": id, "name"}`, `{"remove": id}`, `{"order": [ids]}` or `{"open": id, "value": bool}` | the folders after the change (`remove` leaves the chats, out of any folder) |
| `POST /api/chats/<id>/folder` | `{"folder": id or null}` | puts the chat in a folder, or takes it out |
| `GET /api/top` | | the Top 10: `boards` `tps`, `peak`, `first`, `tokens`, `lanes`, each a list of `{question, at, tokens, tokens_per_second, peak_tps, first_text_s, lanes, seconds, words, route, together, source, settings: {model, model_label, size, size_used, think, search, kiwix, first, temperature, lanes}}` |
| `POST /api/top/clear` | `{"board"?}` | empties one board, or all |
| `GET /api/lanes` | | the engine's `/lanes` |
| `POST /api/search` | `{"q", "n"?}` | web results (for the settings' test) |
| `GET /api/hf/files?repo=owner/name` | | the repository's GGUF files with sizes |
| `GET /api/hf/search?q=&sort=&limit=` | | GGUF chat models on Hugging Face: `repo`, `owner`, `name`, `likes`, `downloads`, `updated`, `vision`, `gated`. `sort`: `trending`, `downloads`, `likes`, `new` |
| `GET /api/hf/model?repo=owner/name` | | one repository: `likes`, `downloads`, `updated`, `license`, `base`, `params`, `context`, `tags`, and `files` (`file`, `gb`, `quant`, `mmproj`, `parts`, `here`: already on this computer) |
| `POST /api/models/folders` | `{add: path}` or `{remove: path}` | a folder of your own where models are looked for; the folders are looked through at once |
| `GET /api/kiwix` | | offline Wikipedia: the files installed and in use, downloads in progress (`state`, `done`, `total`, `mb_s`, `connections`), `partial` (downloads that were cut off and go on when started again), the folder |
| `GET /api/kiwix/updates` | | `{updates: {file: entry}}`: for each file on this PC, the newer edition in the Kiwix library, if there is one (asks library.kiwix.org) |
| `GET /api/kiwix/languages`, `GET /api/kiwix/catalog?lang=` | | the Kiwix library: languages, and the Wikipedia editions of one |
| `POST /api/kiwix/download`, `/cancel`, `/use`, `/remove`, `/add`, `/open`, `/folder` | `{entry, replaces?}` (`replaces`: the older file this one takes the place of once it is whole), `{id}`, `{file, on}` or `{order}`, `{file}`, `{path}`, `{}`, `{folder}` | download an edition, cancel, switch a file on or off or order the files, delete, add a file or folder from elsewhere, show the folder, change it |
| `GET /kiwix/<book>/<path>` | | an offline article, as the sources of an answer link to it |
| `POST /api/downloads` | `{"repo", "file"}` | starts a download (over `downloads.connections` connections, 6 unless set in `data/settings.json`; resumable); `GET /api/downloads` lists them; `POST /api/downloads/<id>/cancel` |
| `POST /api/shutdown` | `{}` | stops the engine (if Rapidli started it) and the app |

### Events of /api/send

`data: {...}` lines: `chat` (its id and title), `search` (`phase`: searching, results,
pages, failed), `size` (length, why, context, lanes, images, drafting), `route` (direct or fan), `plan` (the outline
as it is written), `outline` (part titles), `points`, `text` (`id` of the part, `delta`), `final`, `title`, `stopped`,
`error`, `done` (the stored message and the context). An answer that waits for its turn first gets `queued` about once
a second (`place` in line, `eta` in seconds or null while a model loads, `why`), then `started`. `/api/status` lists the
answers being written and waiting under `answers`.

## The engine (port 11832)

`GET /health`, `GET /stats`, `GET /lanes[?since=n]`, `POST /completion`, `POST /fork`, `POST /spec`,
`POST /media`, `POST /release`, `POST /tokenize`, `POST /cancel`. Request fields: `prompt`, `session`, `name`, `hold`,
`release`, `read_only`, `bulk`, `lookup_only`, `branches`, `n_predict`, `stop`, `soft`, `temperature`, `min_p`,
`top_p`, `seed`, `race`, `prefix`, `priority` (`high`, `normal`, `low`), `tag` (`POST /cancel {"tag"}` ends every request
with it, also one still reading its prompt), `stream` (on `/completion`: server-sent events in `/fork`'s form).
See `docs\HOW-IT-WORKS.md` and the comments in `engine\src\engine.cpp`.

Start it by hand: `python app\eng.py start model=<file.gguf> lanes=64 ctx=300 state=8 vault=8 chunk=256 ctx_extra=8192
[spec=3 spec_ctx=4096] [mmproj=<projector.gguf>]`, stop it with `python app\eng.py stop`.

`POST /api/send` also takes `kiwix` (offline Wikipedia on or off, default on) and `first` (`"wiki"` or `"web"`: which is
asked first when both are on). Its `search` events carry the phase `kiwix` with `{found, answers, seconds, web}`.
A story is answered with the route `story`: a `text` event with the id `title`, `plan` events while the plan is
written, an `outline` (the chapter titles; empty for a short story), `points` (the scene plans of each chapter), then
`text` for every scene (`"3"`, or `"2.1"` in a story with chapters).
