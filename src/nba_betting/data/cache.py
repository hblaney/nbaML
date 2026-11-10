"""Read/write local parquet and CSV caches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_frame(df: pd.DataFrame, path: Path, *, also_csv: bool = True) -> Path:
    """Write ``df`` to ``path`` (parquet or csv by suffix). Optionally mirror CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        if also_csv:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported cache format: {path}")
    return path


def read_frame(path: Path) -> pd.DataFrame:
    """Load a cached frame from parquet or csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported cache format: {path}")


def resolve_existing(path: Path) -> Path:
    """Prefer parquet; fall back to sibling csv if parquet missing."""
    path = Path(path)
    if path.exists():
        return path
    if path.suffix == ".parquet":
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"cache miss: {path}")
