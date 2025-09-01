import json
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import pandas as pd

from .tabular_data import TabularSchema, TabularData


def _infer_format(path: Path) -> Literal["csv", "parquet"]:
    """Infer file format from extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in (".parquet", ".pq", ".parq"):
        return "parquet"
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .parquet.")


def _schema_from_dict(d: Dict[str, Any]) -> TabularSchema:
    """Create TabularSchema from a dictionary, ensuring tuples for immutability style."""
    return TabularSchema(
        target=d["target"],
        numeric=tuple(d.get("numeric", ())),
        categorical=tuple(d.get("categorical", ())),
        extra=tuple(d.get("extra", ())),
    )


class TabularDataRepository:
    """
    Repository for loading and saving TabularData.

    load(path, schema=None):
        Loads from CSV/Parquet. If schema is not provided, attempts to read '<stem>.schema.json'.
    save(tab, path, include_schema=True):
        Saves DataFrame to CSV/Parquet. Optionally writes '<stem>.schema.json' if schema is set.
    """

    @staticmethod
    def load(
            path: str | Path,
            schema: Optional[TabularSchema] = None,
            num_dtype="float64",
            cat_dtype="object",
            **kwargs,
    ) -> TabularData:
        p = Path(path)
        fmt = _infer_format(p)

        if fmt == "csv":
            df = pd.read_csv(p, **kwargs)
        else:
            df = pd.read_parquet(p, **kwargs)

        if schema is None:
            schema_path = p.with_suffix("").with_suffix(".schema.json")
            if schema_path.exists():
                with open(schema_path, encoding="utf-8") as f:
                    schema = _schema_from_dict(json.load(f))

        if schema is not None:
            df[list(schema.numeric)] = df[list(schema.numeric)].astype(num_dtype)
            df[list(schema.categorical)] = df[list(schema.categorical)].astype(cat_dtype)

        return TabularData(df, schema)

    @staticmethod
    def save(
            tab: TabularData,
            path: str | Path,
            include_schema: bool = True,
            index: bool = True,
            **kwargs,
    ) -> None:
        p = Path(path)
        fmt = _infer_format(p)
        p.parent.mkdir(parents=True, exist_ok=True)

        df = tab.df

        if fmt == "csv":
            df.to_csv(p, index=index, **kwargs)
        elif fmt == "parquet":
            df_to_write = df.reset_index(drop=not index)
            df_to_write.to_parquet(p, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Only 'csv' and 'parquet' are allowed.")

        if include_schema and tab.schema is not None:
            schema_path = p.with_suffix("").with_suffix(".schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(tab.schema.to_dict(), ensure_ascii=False, indent=2))
