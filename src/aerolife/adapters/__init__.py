"""Dataset adapters: raw publisher files -> canonical observations/units tables.

No adapter is registered. Writing one is a dataset-specific step and is
deliberately left undone until the faculty decision in FEASIBILITY.md is
recorded (Decision A or B). An adapter must:

* read files from data/raw/ only, never modifying them;
* emit observations with unit_id, time_index, time_value (+ measured channels);
* emit units with event_observed / failure_time taken from the publisher's
  documented end-of-life definition, NaN failure_time for censored units;
* document every parsing rule and exclusion in the dataset card.
"""

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
