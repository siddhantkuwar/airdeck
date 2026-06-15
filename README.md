# AirDeck

macOS-first webcam gesture controller prototype.

AirDeck uses a bounded camera pipeline and Overshoot-hosted vision inference to translate deliberate hand gestures into native macOS hotkeys.

## V1

The first slice includes local configuration, redacted structured logging, a maxsize-one webcam frame queue, a capture producer, graceful shutdown, and a compact Tkinter app shell.

## V2

The second slice adds the Overshoot safety layer: model discovery, stream lifecycle helpers, latest-frame inference payloads, strict gesture JSON validation, request/budget guards, a capped 480p publisher loop, and a gesture confirmation state machine.

Install the optional LiveKit publisher dependency when testing against a real Overshoot stream:

```bash
.venv312/bin/python -m pip install -e '.[camera,overshoot]'
```

Set `AIRDECK_OVERSHOOT_LIVE=true` only when you want Start to create a real Overshoot stream and consume inference requests. The default remains local demo mode for safe gesture/UI testing.

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[camera]'
cp .env.example .env
.venv312/bin/python -m airdeck.main
```

Run tests:

```bash
.venv312/bin/python -m unittest discover -s tests
```
