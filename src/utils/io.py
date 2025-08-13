import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Any


def save_df_parquet(df: pd.DataFrame | pd.Series, path: str | Path, compression: str = "snappy") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_parquet(path, compression=compression, index=False)


def load_df_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
