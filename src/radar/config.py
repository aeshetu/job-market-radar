from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"              # one gzipped JSON per monthly thread (not committed)
INTERIM = DATA / "interim"      # flattened posts before feature extraction (not committed)
PROCESSED = DATA / "processed"  # small, aggregated tables the dashboard reads (committed)

for _p in (RAW, INTERIM, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
