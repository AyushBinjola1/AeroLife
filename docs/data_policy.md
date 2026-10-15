# Raw-Data Acquisition and Immutability Policy

1. **Only the original publisher's files are used.** Download them from the URL named in the dataset
   card. Use mirrors only if the publisher lists them, and record which mirror was used.
2. **Raw files are immutable.** Put them in `data/raw/` exactly as downloaded. Never edit, re-save,
   resample or "clean" them in place. All parsing writes new files to `data/canonical/`.
3. **Checksums.** On the first ingest, `scripts/ingest.py` writes `data/provenance_manifest.json` with
   the SHA-256 and byte size of every raw file plus the source block of the dataset spec (URL,
   citation, release, retrieval date, licence or usage terms). Every later ingest re-verifies the
   checksums and stops if any file changed. `aerolife.provenance.make_read_only` can set the files read-only.
4. **Parsing rules** (column order, units, time-index construction, end-of-life rule) are written in
   the dataset card and implemented in exactly one adapter.
5. **Counts.** The ingest summary records units total, included, excluded (with reasons) and censored,
   plus observation counts. The dataset card reports original and retained sample counts.
6. **No synthetic data.** No simulated, augmented, interpolated or noise-generated measurements are
   added. Label columns are never inputs.
7. **Redistribution.** Raw files are not committed to git (`.gitignore`). Reproduce them by
   downloading from the source and checking them against the manifest.
