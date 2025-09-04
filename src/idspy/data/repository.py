from pathlib import Path
from typing import Optional, Union, Any, Tuple

import pandas as pd

from src.idspy.data.schema import Schema


class DataFrameRepository:
    """Load/save pandas DataFrames."""

    @staticmethod
    def _resolve_path_and_format(
            base_path: Union[str, Path],
            name: str,
            fmt: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Return resolved file path and format."""
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)

        if fmt:
            return base / f"{name}.{fmt}", fmt

        suffix = Path(name).suffix.lstrip(".")
        if suffix:
            return base / name, suffix

        matches = sorted(base.glob(f"{name}.*"))
        if not matches:
            raise ValueError(
                f"Format not specified and cannot be inferred for '{name}'."
            )

        file_path = matches[0]
        return file_path, file_path.suffix.lstrip(".")

    @staticmethod
    def save(
            df: pd.DataFrame,
            base_path: Union[str, Path],
            name: str,
            fmt: str = "parquet",
            **kwargs: Any,
    ) -> Path:
        """Save DataFrame."""
        file_path, fmt = DataFrameRepository._resolve_path_and_format(base_path, name, fmt)

        if fmt == "parquet":
            df.to_parquet(file_path, **kwargs)
        elif fmt == "csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif fmt == "pickle":
            df.to_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return file_path

    @staticmethod
    def load(
            base_path: Union[str, Path],
            name: str,
            fmt: Optional[str] = None,
            schema: Optional[Schema] = None,
            **kwargs: Any,
    ) -> pd.DataFrame:
        """Load DataFrame; optionally attach schema via df = df.tab.set_schema(schema)."""
        file_path, fmt = DataFrameRepository._resolve_path_and_format(base_path, name, fmt)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        if fmt == "parquet":
            df = pd.read_parquet(file_path, **kwargs)
        elif fmt == "csv":
            df = pd.read_csv(file_path, **kwargs)
        elif fmt == "pickle":
            df = pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # Optionally associate schema using the provided API.
        if schema is not None:
            df.tab.set_schema(schema)

        return df
