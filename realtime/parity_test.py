"""Parity test (step 7): the live Python features must equal the Stage 1 SQL features.

  python realtime/parity_test.py            # at the main cutoff
  python realtime/parity_test.py 2025-09-30
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ds.catalog import ALL_FEATURES  # noqa: E402
from ds.livefeatures import LiveStore  # noqa: E402
from stage1.extract_dataset import run_query  # noqa: E402


def main(cutoff=config.CUTOFF):
    sql = run_query(cutoff, (pd.Timestamp(cutoff) + pd.Timedelta(days=90)).strftime("%Y-%m-%d")).set_index("client_id")
    store = LiveStore()
    live = store.features(cutoff, clients=sql.index.values).set_index("client_id")
    rows = []
    for f in ALL_FEATURES:
        a, b = sql[f], live.loc[sql.index, f]
        if pd.api.types.is_numeric_dtype(a):
            diff = (~np.isclose(a.astype(float).fillna(-999), b.astype(float).fillna(-999), atol=1e-3)).sum()
        else:
            diff = (a.astype(str) != b.astype(str)).sum()
        rows.append({"feature": f, "rows_different": int(diff)})
    t = pd.DataFrame(rows)
    t.to_csv(config.TABLES / "parity_test.csv", index=False)
    ok = t["rows_different"].sum() == 0
    print(t.to_string(index=False))
    print(f"\nPARITY live vs SQL on {len(sql)} clients at {cutoff}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
