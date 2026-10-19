"""NASA FD001 parser. Official test RUL is used only for retrospective labels."""
from pathlib import Path

import pandas as pd

from . import DatasetAdapter


class CMAPSSFD001(DatasetAdapter):
    name = "cmapss_fd001"

    def parse(self, raw_dir: Path):
        folder = Path(raw_dir) / "CMAPSSData"
        channels = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
        labels = pd.read_csv(folder / "RUL_FD001.txt", sep=r"\s+", header=None)[0]
        observations, units = [], []
        for partition in ("train", "test"):
            frame = pd.read_csv(folder / f"{partition}_FD001.txt", sep=r"\s+", header=None)
            if frame.shape[1] != 26:
                raise ValueError("Expected 26 columns: engine, cycle, 3 settings, 21 sensors")
            frame.columns = ["engine", "cycle", *channels]
            if sorted(frame.engine.unique()) != list(range(1, 101)) or len(labels) != 100:
                raise ValueError("FD001 requires 100 official engines in each partition and 100 labels")
            for engine, g in frame.groupby("engine"):
                uid = f"{partition}_{engine:03d}"
                if g.cycle.tolist() != list(range(1, len(g) + 1)):
                    raise ValueError(f"Non-contiguous cycle history: {uid}")
                out = g[channels].copy()
                out.insert(0, "time_value", g.cycle.to_numpy() - 1)
                out.insert(0, "time_index", g.cycle.to_numpy() - 1)
                out.insert(0, "unit_id", uid)
                observations.append(out)
                offset = int(labels.iloc[engine - 1]) if partition == "test" else 0
                units.append(dict(unit_id=uid, event_observed=True,
                                  failure_time=int(g.cycle.max()) - 1 + offset,
                                  condition_id="sea_level", include=True, exclusion_reason="",
                                  source_partition=partition,
                                  event_basis="official_RUL_label" if offset else "observed_simulated_failure"))
        return pd.concat(observations, ignore_index=True), pd.DataFrame(units)
