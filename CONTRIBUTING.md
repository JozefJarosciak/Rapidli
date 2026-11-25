# Contributing to Rapidli

Thank you for looking. Rapidli is about one thing: fast local inference. A change is welcome when it makes answers
faster, makes them better at the same speed, or makes the app simpler to use.

## Getting it running

- Windows or Linux, an NVIDIA card, Python 3.10 or newer. `pip install -r requirements.txt` adds the optional parts.
- The offline Wikipedia is not in the repository: Rapidli offers to download it on its first start (1 GB).
- Windows: `start.bat`. Linux: build the engine once with `engine/build-linux.sh`, then `./start.sh`.
- The app is plain Python and one HTML page: there is no build step for it. Edit, restart, reload.
- Have a Mac? Two short checks in the README's [macOS (help wanted)](README.md#macos-help-wanted) section would help.

## Where things are

`README.md` has the folder map. In short: `app/rapidli.py` is the server, `app/fan.py` writes answers with the lanes,
`app/sizing.py` decides how an answer is written, `app/plat.py` holds everything that differs between Windows and
Linux, `app/web/index.html` is the whole page, `engine/src/engine.cpp` is the engine.

## Before you send a change

- Say what you measured. For speed, `python tests/lane_speed.py` (the engine alone) and `python tests/story_app.py`
  (a story through the app) print tokens and words per second; give the numbers before and after, with the model and
  the card.
- Run the tests that touch what you changed (`tests/`; each file's first lines say what it needs). Most talk to a
  running Rapidli and move the chats they make out of the way; the ones that need nothing running are run by
  `sh tests/run_offline.sh` (Windows: `tests\run_offline.bat`), as GitHub does on every push. For a change to how
  answers are written, also `python tests/probes.py mytag fast` (under 5 minutes, with a model loaded).
- Platform differences go into `app/plat.py`, not into `if os.name` checks scattered through the app.
- Keep comments about why, with the case that showed it. Most rules in `sizing.py` and `fan.py` exist because a real
  request went wrong without them.
- No personal data, no model files, no chats in a commit (`.gitignore` covers the usual places).

## Reporting a problem

Say which model file and quantization, the card and its memory, the operating system, and what the status line under
the model's name said. `data/logs/rapidli.log` and `data/logs/engine-11832.log` usually show the cause; remove
anything private before you attach them.
