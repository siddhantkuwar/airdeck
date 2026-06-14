# AirDeck

macOS-first webcam gesture controller prototype.

AirDeck uses a bounded camera pipeline and Overshoot-hosted vision inference to translate deliberate hand gestures into native macOS hotkeys.

## V1

The first slice includes local configuration, redacted structured logging, a maxsize-one webcam frame queue, a capture producer, graceful shutdown, and a compact Tkinter app shell.

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
