"""External validation ONLY: the public github "Churn Modelling" set (10 000 clients).

It is never the starting point. It goes through the same path as any other source:
raw file -> staging table (stg_github, source 'github') -> cleaning -> query restricted
to the features it actually has (the profile features shared with the simulated bank).

Download it yourself (github account needed):
  https://www.github.com/datasets/shrutimechlearn/churn-modelling   -> Churn_Modelling.csv
and copy it to data/external/Churn_Modelling.csv, then:
  python stage1/github_load.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from stage1 import db  # noqa: E402

github_FILE = config.EXTERNAL / "Churn_Modelling.csv"
COMMON = ["age", "gender", "tenure_months", "credit_score", "num_products", "has_card",
          "is_active_member", "est_salary", "balance"]

QUERY = """
SELECT CAST(customerid AS INTEGER)                 AS client_id,
       CAST(age AS INTEGER)                        AS age,
       CASE WHEN lower(gender) LIKE 'f%' THEN 'F' ELSE 'M' END AS gender,
       CAST(tenure AS INTEGER) * 12                AS tenure_months,
       CAST(creditscore AS FLOAT)                  AS credit_score,
       CAST(numofproducts AS INTEGER)              AS num_products,
       CAST(hascrcard AS INTEGER)                  AS has_card,
       CAST(isactivemember AS INTEGER)             AS is_active_member,
       CAST(estimatedsalary AS FLOAT)              AS est_salary,
       CAST(balance AS FLOAT)                      AS balance,
       geography                                   AS geography,
       CAST(exited AS INTEGER)                     AS churn
FROM stg_github
WHERE CAST(age AS INTEGER) BETWEEN 18 AND 100
"""


def available() -> bool:
    return github_FILE.exists()


def load() -> pd.DataFrame:
    raw = pd.read_csv(github_FILE, dtype=str)
    raw = raw.drop(columns=[c for c in ("RowNumber", "Surname") if c in raw.columns])  # GDPR: no names
    raw = raw.drop_duplicates("CustomerId")
    raw.columns = [c.lower() for c in raw.columns]  # unquoted identifiers work in both engines
    cols = ", ".join(f"{c} TEXT" for c in raw.columns)
    db.run_script(f"DROP TABLE IF EXISTS stg_github; CREATE TABLE stg_github ({cols})", "github")
    db.bulk_insert(raw, "stg_github", "github")
    df = db.read_sql(QUERY, source="github")
    df.to_csv(config.DATASETS / "github.csv", index=False)
    return df


if __name__ == "__main__":
    if not available():
        print(__doc__)
        sys.exit(1)
    d = load()
    print(f"github dataset: {d.shape}, churn share {d['churn'].mean():.3f}")
