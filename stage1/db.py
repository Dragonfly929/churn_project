"""Database access for both engines (SQLite and PostgreSQL).

The DWH has one schema per *source*:
  - "bank"   : the simulated bank (main source)
  - "branch" : the external branch export (pass 2 of XAI)
  - "github" : the public github set (external validation only)
SQLite -> one file per source in data/.  PostgreSQL -> one schema per source
inside database bank_dwh (search_path is set on connect).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text, event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

_ENGINES: dict = {}


def backend() -> str:
    return config.DB_BACKEND


def get_engine(source: str = "bank"):
    key = (config.DB_BACKEND, source)
    if key in _ENGINES:
        return _ENGINES[key]
    if config.DB_BACKEND == "postgres":
        eng = create_engine(
            config.PG_URL,
            connect_args={"options": f"-csearch_path={source},public"},
            pool_pre_ping=True,
        )
        with eng.begin() as c:
            c.execute(text(f"CREATE SCHEMA IF NOT EXISTS {source}"))
    else:
        path = config.DATA / f"dwh_{source}.sqlite"
        eng = create_engine(f"sqlite:///{path}")

        @event.listens_for(eng, "connect")
        def _fk_on(dbapi_conn, _):  # enforce foreign keys in SQLite
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    _ENGINES[key] = eng
    return eng


def run_script(sql: str, source: str = "bank") -> None:
    """Execute a multi-statement SQL script."""
    eng = get_engine(source)
    # drop comment lines first (comments must not contain ';'), then split
    clean = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    with eng.begin() as c:
        for s in clean.split(";"):
            if s.strip():
                c.execute(text(s.strip()))


def read_sql(query: str, params: dict | None = None, source: str = "bank") -> pd.DataFrame:
    eng = get_engine(source)
    # comment lines may mention :params -> strip them before binding
    query = "\n".join(l for l in query.splitlines() if not l.strip().startswith("--"))
    with eng.connect() as c:
        return pd.read_sql(text(query), c, params=params or {})


def bulk_insert(df: pd.DataFrame, table: str, source: str = "bank") -> None:
    """Fast append. PostgreSQL uses COPY; SQLite uses executemany."""
    if df.empty:
        return
    eng = get_engine(source)
    if config.DB_BACKEND == "postgres":
        raw = eng.raw_connection()
        try:
            cols = ",".join(df.columns)
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            with raw.cursor() as cur:
                with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
                    cp.write(buf.getvalue())
            raw.commit()
        finally:
            raw.close()
    else:
        raw = eng.raw_connection()
        try:
            cols = ",".join(df.columns)
            ph = ",".join("?" * len(df.columns))
            rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                          for v in r) for r in df.itertuples(index=False, name=None)]
            cur = raw.cursor()
            cur.executemany(f"INSERT INTO {table} ({cols}) VALUES ({ph})", rows)
            raw.commit()
        finally:
            raw.close()


def sql_file(name: str) -> str:
    """Return the engine-specific SQL file: <name>_<backend>.sql, or <name>.sql."""
    p = config.SQL_DIR / f"{name}_{config.DB_BACKEND}.sql"
    if not p.exists():
        p = config.SQL_DIR / f"{name}.sql"
    return p.read_text(encoding="utf-8")
