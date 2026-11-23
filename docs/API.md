# HTTP API

Both servers listen on 127.0.0.1 only. With Settings › API › "Other computers" on (and Rapidli started again), the
app's port also answers other computers, but only on the paths of [For other programs](#for-other-programs-settings--api)
and only with the key.

## The app (port 11842)

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
| `POST /api/send` | `{"chat"?, "run"?, "text", "size"?, "images"?: [id], "search"?: bool, "regenerate"?: bool, "index"?}` | an event stream (below). Several chats can be answered at once (Settings › Chat: side by side, or one after another); a second answer in the same chat is refused with 409. `run`: an id of the caller's choosing, for `/api/stop` |
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

**Events of `/api/send`** (`data: {...}` lines): `chat` (its id and title), `search` (`phase`: searching, results,
pages, failed), `size` (length, why, context, lanes, images, drafting), `route` (direct or fan), `plan` (the outline
as it is written), `outline` (part titles), `points`, `text` (`id` of the part, `delta`), `final`, `title`, `stopped`,
`error`, `done` (the stored message and the context). An answer that waits for its turn first gets `queued` about once
a second (`place` in line, `eta` in seconds or null while a model loads, `why`), then `started`. `/api/status` lists the
answers being written and waiting under `answers`.

## For other programs (Settings › API)

Off until switched on in Settings › API. With "Key required" (the default), every call sends
`Authorization: Bearer <key>` (or `X-API-Key: <key>`); from other computers the key is always required. Requests are
answered side by side or one after another, as set there. Errors are `{"error": {"message", "type", "code"}}`.

| Method and path | Body | Answer |
|---|---|---|
| `POST /api/v1/ask` | `{"question", "chat"?, "size"?, "think"?, "search"?, "kiwix"?, "first"?, "stream"?}` | `{"chat", "answer", "stats": {route, size, seconds, tokens, tokens_per_second, lanes, first_text_s}, "search": [sources]}`; with `"stream": true`, the events of `/api/send` |
| `POST /v1/chat/completions` | OpenAI's: `{"model", "messages", "stream"?}`, plus `size`, `think`, `search`, `kiwix` | OpenAI's `chat.completion` (and `rapidli`: the chat and the stats); streamed, `chat.completion.chunk` lines and `data: [DONE]` (a one-lane answer word by word, a many-lane answer as a whole at its end; while it waits for its turn, SSE comments) |
| `GET /v1/models` | | the loaded model, as OpenAI lists models |
| `GET /api/v1/status` | | `status`, `message`, `model`, `lanes`, `answers` (being written and waiting), `together` |

The messages before the last one of `/v1/chat/completions` are kept as the conversation, and system messages are
followed as instructions for the answer. The API's conversations are kept in a folder of the chat list called API
(Settings › API › "Keep the API's chats" off: without a folder).


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
