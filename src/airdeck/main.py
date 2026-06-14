from __future__ import annotations

import signal
import sys
import tkinter as tk

from airdeck.config import ConfigurationError, load_settings
from airdeck.logging_setup import configure_logging
from airdeck.ui.app import AirDeckApp


def main() -> int:
    try:
        settings = load_settings(require_api_key=True)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger = configure_logging(settings)
    root = tk.Tk()
    app = AirDeckApp(root, settings, logger=logger)

    def handle_sigint(_signum: int, _frame: object) -> None:
        logger.info("sigint_received")
        try:
            root.after(0, app.close)
        except tk.TclError:
            app.close()

    signal.signal(signal.SIGINT, handle_sigint)
    logger.info("airdeck_started demo_mode=%s", settings.demo_mode)
    root.mainloop()
    logger.info("airdeck_exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
