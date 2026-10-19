"""Dataset adapters for immutable publisher files; NASA C-MAPSS FD001 is registered."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def parse(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (observations, units) in the canonical schema."""


REGISTRY: dict[str, type[DatasetAdapter]] = {}

from .cmapss import CMAPSSFD001

REGISTRY["cmapss_fd001"] = CMAPSSFD001
