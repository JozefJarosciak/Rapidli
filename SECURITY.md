# Security

Rapidli runs models on your own computer. What it exposes, and how to report a problem.

## What listens, and where

- The app listens on `127.0.0.1:11842` and the engine on `127.0.0.1:11832`: only programs on the same computer reach
  them. The app answers requests from this computer only (anything else gets 403).
- **The API for other programs** (Settings › API) is off until you switch it on. With "Key required" (the default),
  every call needs `Authorization: Bearer <key>`. The key is made on this computer and kept in `data/settings.json`;
  "New key" stops the old one at once.
- **Other computers** (Settings › API) opens the app's port to your network after a restart, but only the API's paths
  (`/v1/…`, `/api/v1/…`), and always with the key, whatever "Key required" says. The page, the chats and the settings
  stay on this computer. Use it on networks you trust; the key travels in plain HTTP.
- Programs on the same computer can reach the app's own endpoints (`/api/…`) without a key, as they can reach your
  files. Do not run Rapidli on a computer you share with people you do not trust.

## What leaves your computer

Nothing, unless you ask for it: web search sends the search words to the search engines, Get models and the offline
Wikipedia screen ask huggingface.co and library.kiwix.org for their lists, and downloads come from there. Requests say
`Rapidli/1.0` and nothing about you. A Hugging Face token, if you set one, is kept in `data/settings.json` and sent only
to huggingface.co.

## Reporting a problem

Please do not open a public issue for a security problem. Use the repository's private vulnerability reporting
(GitHub: Security › Report a vulnerability) with what you found, how to reproduce it, and the version (commit) you
used. You can expect a first answer within a week.
