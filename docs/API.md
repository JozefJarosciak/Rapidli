# HTTP API

Rapidli answers other programs over HTTP: load and unload models, and ask questions with the same choices as the
composer (length, thinking, offline Wikipedia, web search, deep research). There is Rapidli's own route and an
OpenAI-style chat route that many tools speak.

Contents: [Switching it on](#switching-it-on) · [Models](#models) · [Asking](#asking) · [Answer options](#answer-options) ·
[OpenAI-style](#openai-style) · [Status and options](#status-and-options) · [Errors](#errors) · [Examples](#examples) ·
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
- **Keep the API's chats**: the API's conversations appear in the chat list, in a folder called API.

The base address is `http://127.0.0.1:11842` (from another computer, this PC's address with the same port).

| Method and path | What it does |
|---|---|
| `GET /api/v1/models` | every model on this PC |
| `POST /api/v1/models/load` | loads a model |
| `POST /api/v1/models/unload` | unloads it: the graphics card is free |
| `GET /api/v1/status` | the model, the lanes, its thinking levels, the answers being written or waiting |
| `GET /api/v1/options` | what a question can be asked with now |
| `POST /api/v1/ask` | asks a question (Rapidli's own route) |
| `POST /v1/chat/completions` | asks a question (OpenAI-style) |
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
  are followed as instructions for the answer. Text parts of a message are read; images are not.
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

## Status and options

### `GET /api/v1/status`

```json
{"status": "ready", "message": "…", "model": "…", "label": "Qwen3.8-35B-A3B-IQ3_M", "lanes": 115,
 "think": ["off", "parts", "on"], "answers": {"running": […], "waiting": […]}, "together": "parallel"}
```

`status`: `idle` (no model), `loading`, `ready`, or `error` (`message` says why).

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
| `404` | no such model, or no such path |
| `409` | a model is being loaded; or an answer is being written in that chat |
| `503` | no model is ready: load one first |

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
`POST /media`, `POST /release`, `POST /tokenize`. Request fields: `prompt`, `session`, `name`, `hold`, `release`,
`read_only`, `bulk`, `lookup_only`, `branches`, `n_predict`, `stop`, `soft`, `temperature`, `race`, `prefix`.
See `docs\HOW-IT-WORKS.md` and the comments in `engine\src\engine.cpp`.

Start it by hand: `python app\eng.py start model=<file.gguf> lanes=64 ctx=300 state=8 vault=8 chunk=256 ctx_extra=8192
[spec=3 spec_ctx=4096] [mmproj=<projector.gguf>]`, stop it with `python app\eng.py stop`.

`POST /api/send` also takes `kiwix` (offline Wikipedia on or off, default on) and `first` (`"wiki"` or `"web"`: which is
asked first when both are on). Its `search` events carry the phase `kiwix` with `{found, answers, seconds, web}`.
A story is answered with the route `story`: a `text` event with the id `title`, `plan` events while the plan is
written, an `outline` (the chapter titles; empty for a short story), `points` (the scene plans of each chapter), then
`text` for every scene (`"3"`, or `"2.1"` in a story with chapters).
