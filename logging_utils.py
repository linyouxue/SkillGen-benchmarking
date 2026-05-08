"""Central logging setup for SkillGen."""

from __future__ import annotations

import logging
import sys

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


class ColorFormatter(logging.Formatter):
    """Pretty console formatter with ANSI colors."""

    LEVEL_COLORS = {
        logging.DEBUG: DIM,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, "")
        level = f"{color}{record.levelname:<8}{RESET}"
        timestamp = f"{DIM}{self.formatTime(record, '%H:%M:%S')}{RESET}"
        name = record.name.replace("skillgen.", "")
        if name == "skillgen":
            name = "main"
        logger_name = f"{BLUE}{name}{RESET}"
        message = super().format(record)
        return f"{timestamp} {level} {logger_name} {message}"


class TqdmCompatibleHandler(logging.StreamHandler):
    """Stream handler that plays nicely with tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=self.stream)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(verbose_http: bool = False) -> logging.Logger:
    """Configure colorful console logging and suppress noisy SDK loggers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = TqdmCompatibleHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ColorFormatter("%(message)s"))
    root.addHandler(console)

    noisy_level = logging.INFO if verbose_http else logging.WARNING
    for logger_name in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "urllib3",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(noisy_level)
        logger.propagate = True

    return logging.getLogger("skillgen")
