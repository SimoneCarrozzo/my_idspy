from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import pandas as pd

from .schema import Schema
from .tab_accessor import TabAccessor

PathLike = Union[str, Path]
Format = Literal["parquet", "csv", "pickle"]


class DataFrameRepository:
    """Load/save pandas DataFrames with format inferred from path."""

    FORMATS = frozenset(["parquet", "csv", "pickle"])

    SAVE_FUNCS: Dict[Format, Callable[[pd.DataFrame, Path, Dict[str, Any]], None]] = {
        "parquet": lambda df, fp, kw: df.to_parquet(fp, **kw),
        "csv": lambda df, fp, kw: df.to_csv(fp, index=False, **kw),
        "pickle": lambda df, fp, kw: df.to_pickle(fp, **kw),
    }

    LOAD_FUNCS: Dict[Format, Callable[[Path, Dict[str, Any]], pd.DataFrame]] = {
        "parquet": lambda fp, kw: pd.read_parquet(fp, **kw),
        "csv": lambda fp, kw: pd.read_csv(fp, **kw),
        "pickle": lambda fp, kw: pd.read_pickle(fp, **kw),
    }

    @classmethod
    def _parse_format(cls, suffix: str) -> Format:
        """Convert file suffix to format."""
        fmt = suffix.lstrip(".").lower()
        if fmt not in cls.FORMATS:
            raise ValueError(f"Unsupported format: {fmt!r}")
        return fmt

    @classmethod
    def _get_metadata_path(cls, file_path: Path) -> Path:
        """Get metadata file path for a given data file."""
        return file_path.with_suffix(".meta")

    @classmethod
    def _resolve_path_and_format(
        cls,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: Optional[Format] = None,
    ) -> Tuple[Path, Format]:
        """Resolve final file path and format from inputs."""
        base = Path(base_path)

        # File path with extension
        if base.suffix:
            resolved_fmt = cls._parse_format(base.suffix)
            if name:
                final_path = base.with_name(Path(name).stem + base.suffix)
            else:
                final_path = base
            return final_path, resolved_fmt

        # Directory path
        if not name:
            raise ValueError("Name required when base_path is a directory")

        name_path = Path(name)

        if name_path.suffix:
            resolved_fmt = cls._parse_format(name_path.suffix)
            final_path = base / name_path.name
        elif fmt:
            resolved_fmt = fmt
            final_path = base / f"{name_path.name}.{fmt}"
        else:
            raise ValueError("Cannot infer format: provide name with suffix or fmt")

        return final_path, resolved_fmt

    @classmethod
    def save(
        cls,
        df: pd.DataFrame,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: Optional[Format] = None,
        save_meta: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Save DataFrame and return file path.

        Note:
            Metadata is saved to a separate .meta file alongside the data file
            and contains schema and partitions information from df.attrs.
        """
        file_path, resolved_fmt = cls._resolve_path_and_format(base_path, name, fmt)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if save_meta:
            meta = df.tab.get_meta()
            # Save metadata to a separate file with .meta extension
            meta_path = cls._get_metadata_path(file_path)

            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

        # metadata will be stored separately if needed
        df.attrs.clear()

        save_func = cls.SAVE_FUNCS[resolved_fmt]
        save_func(df, file_path, kwargs)

        return file_path

    @classmethod
    def load(
        cls,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: Optional[Format] = None,
        schema: Optional[Schema] = None,
        load_meta: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load DataFrame with optional schema.

        Note:
            Metadata is loaded from .meta file if it exists and load_meta=True.
            If no metadata file exists but schema is provided, schema is applied.
        """
        file_path, resolved_fmt = cls._resolve_path_and_format(base_path, name, fmt)

        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        load_func = cls.LOAD_FUNCS[resolved_fmt]
        df = load_func(file_path, kwargs)

        if load_meta:
            # Load metadata from separate .meta file
            meta_path = cls._get_metadata_path(file_path)
            if meta_path.exists():
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                        df.tab.load_meta(meta)
                except Exception:
                    # If metadata loading fails, continue without metadata
                    pass
        if schema:
            df.tab.set_schema(schema)

        return df

    @classmethod
    def has_metadata(
        cls,
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: Optional[Format] = None,
    ) -> bool:
        """Check if metadata file exists for a dataset."""
        file_path, _ = cls._resolve_path_and_format(base_path, name, fmt)
        meta_path = cls._get_metadata_path(file_path)
        return meta_path.exists()
