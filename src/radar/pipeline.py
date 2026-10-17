"""Run the whole pipeline: python -m radar.pipeline [--skip-ingest] [--skip-embed]"""
from __future__ import annotations

import argparse
import logging
import time

from . import clusters, embed, features, forecast, ingest, salary_model, trends

STEPS = {
    "ingest": ingest.run,
    "features": features.run,
    "trends": trends.run,
    "forecast": forecast.run,
    "embed": embed.run,
    "salary": salary_model.run,
    "clusters": clusters.run,
}
NEEDS_EMBEDDINGS = {"embed", "clusters"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-ingest", action="store_true", help="reuse downloaded threads")
    ap.add_argument("--skip-embed", action="store_true", help="skip embedding + clustering steps")
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run just these steps")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for name, step in STEPS.items():
        if args.only and name not in args.only:
            continue
        if (args.skip_ingest and name == "ingest") or (args.skip_embed and name in NEEDS_EMBEDDINGS):
            continue
        t = time.time()
        logging.info("== %s", name)
        step()
        logging.info("== %s done in %.0fs", name, time.time() - t)


if __name__ == "__main__":
    main()
