"""Structured-enough logging for a local service: one line per event, key=value details."""

from __future__ import annotations

import logging

LOGGER_NAME = "herbascope"


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
