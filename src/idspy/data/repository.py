from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, Callable, cast

import pandas as pd

from src.idspy.data.schema import Schema

PathLike = Union[str, Path]
Fmt = Literal["parquet", "csv", "pickle"]


class DataFrameRepository:
    """Load/save pandas DataFrames with format inferred from path/name/fmt."""

    _SAVE_FUNCS: Dict[Fmt, Callable[[pd.DataFrame, Path, Dict[str, Any]], None]] = {
        "parquet": lambda df, fp, kw: df.to_parquet(fp, **kw),
        "csv": lambda df, fp, kw: df.to_csv(fp, index=False, **kw),
        "pickle": lambda df, fp, kw: df.to_pickle(fp, **kw),
    }

    _LOAD_FUNCS: Dict[Fmt, Callable[[Path, Dict[str, Any]], pd.DataFrame]] = {
        "parquet": lambda fp, kw: pd.read_parquet(fp, **kw),
        "csv": lambda fp, kw: pd.read_csv(fp, **kw),
        "pickle": lambda fp, kw: pd.read_pickle(fp, **kw),
    }

    @staticmethod
    def resolve_path_and_format(
            base_path: PathLike,
            name: Optional[str] = None,
            fmt: Optional[Fmt] = None,
    ) -> Tuple[Path, Fmt]:
        """
        Infer final file path and format from base_path/name/fmt.

        - If base_path has a suffix → that suffix sets the format.
        - Else (directory-like):
          • If name has a suffix → use it.
          • Else if fmt is given → use it.
          • Else → error.

        Ensures parent directories exist.
        """
        bp = Path(base_path)

        def _suffix_to_fmt(suffix: str) -> Fmt:
            suf = suffix.lstrip(".").lower()
            if suf not in ("parquet", "csv", "pickle"):
                raise ValueError(f"Unsupported format: {suf!r}")
            return cast(Fmt, suf)

        # Case 1: base_path has an explicit suffix => treat as file-like; its suffix wins
        if bp.suffix:
            out_fmt = _suffix_to_fmt(bp.suffix)
            if name is not None:
                # Keep base's directory and suffix; replace the stem with provided name's stem
                name_stem = Path(name).stem
                final = bp.with_name(name_stem + bp.suffix)
            else:
                final = bp

        # Case 2: base_path is directory-like (exists as dir OR has no suffix)
        else:
            base_dir = bp if (bp.exists() and bp.is_dir()) or not bp.suffix else bp.parent

            if name is None:
                raise ValueError(
                    "No file name provided: when base_path is a directory, 'name' is required."
                )

            name_path = Path(name)

            if name_path.suffix:
                out_fmt = _suffix_to_fmt(name_path.suffix)
                final = base_dir / name_path.name
            elif fmt is not None:
                out_fmt = fmt
                final = base_dir / f"{name_path.name}.{str(fmt)}"
            else:
                raise ValueError(
                    "Cannot infer format: provide 'name' with a suffix or pass 'fmt'."
                )

        # Ensure parent directories exist
        final.parent.mkdir(parents=True, exist_ok=True)

        return final, out_fmt

    @staticmethod
    def save(
            df: pd.DataFrame,
            base_path: PathLike,
            name: Optional[str] = None,
            fmt: Optional[Fmt] = None,
            **kwargs: Any,
    ) -> Path:
        """Save DataFrame to disk and return the final file path."""
        file_path, resolved_fmt = DataFrameRepository.resolve_path_and_format(base_path, name, fmt)

        saver = DataFrameRepository._SAVE_FUNCS.get(resolved_fmt)
        if not saver:
            raise ValueError(f"Unsupported format: {resolved_fmt}")
        saver(df, file_path, kwargs)
        return file_path

    @staticmethod
    def load(
            base_path: PathLike,
            name: Optional[str] = None,
            fmt: Optional[Fmt] = None,
            schema: Optional[Schema] = None,
            **kwargs: Any,
    ) -> pd.DataFrame:
        """Load DataFrame; optionally attach schema if provided."""
        file_path, resolved_fmt = DataFrameRepository.resolve_path_and_format(base_path, name, fmt)

        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        loader = DataFrameRepository._LOAD_FUNCS.get(resolved_fmt)
        if not loader:
            raise ValueError(f"Unsupported format: {resolved_fmt}")
        df = loader(file_path, kwargs)

        if schema is not None:
            try:
                df.tab.set_schema(schema)
                df.tab.numerical = df.tab.numerical.astype("float64")
                df.tab.categorical = df.tab.categorical.astype("string")
                df.tab.target = df.tab.target.astype("string")
            except AttributeError:
                pass

        return df
