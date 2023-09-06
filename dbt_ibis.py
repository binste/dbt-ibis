import re
import subprocess
import sys
from abc import ABC, abstractproperty
from collections import defaultdict
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Callable, Final

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.contracts.graph.manifest import Manifest

__version__ = "0.1.0dev"


_REF_IDENTIFIER_PREFIX: Final = "__ibd_ref__"
_REF_IDENTIFIER_SUFFIX: Final = "__rid__"
_SOURCE_IDENTIFIER_PREFIX: Final = "__ibd_source__"
_SOURCE_IDENTIFIER_SUFFIX: Final = "__sid__"
_SOURCE_IDENTIFIER_SEPARATOR: Final = "__ibd_sep__"
_IBIS_MODEL_FILE_EXTENSION: Final = "ibis"


class _Reference(ABC):
    @abstractproperty
    def _ibis_table_name(self) -> str:
        pass

    def to_ibis(self, schema) -> ibis.expr.types.Table:
        if schema is None:
            raise NotImplementedError
        return ibis.table(
            schema,
            name=self._ibis_table_name,
        )


@dataclass
class ref(_Reference):
    model_name: str

    @property
    def _ibis_table_name(self) -> str:
        return _REF_IDENTIFIER_PREFIX + self.model_name + _REF_IDENTIFIER_SUFFIX


@dataclass
class source(_Reference):
    source_name: str
    table_name: str

    @property
    def _ibis_table_name(self) -> str:
        return (
            _SOURCE_IDENTIFIER_PREFIX
            + self.source_name
            + _SOURCE_IDENTIFIER_SEPARATOR
            + self.table_name
            + _SOURCE_IDENTIFIER_SUFFIX
        )


def depends_on(*references):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.depends_on = references
        return wrapper

    return decorator


def _compile_ibis_models(
    dbt_model_folder: Path = Path("models"),
) -> None:
    ibis_model_files = list(dbt_model_folder.glob(f"**/*.{_IBIS_MODEL_FILE_EXTENSION}"))

    # Create empty placeholder file for every Ibis model so that we can run dbt parse
    ibis_dbt_model_files = _create_empty_placeholder_files(ibis_model_files)

    model_infos, source_infos = _get_model_and_source_infos()

    for model_file, dbt_model_file in zip(ibis_model_files, ibis_dbt_model_files):
        model_name = model_file.stem

        model_func = _get_model_func(model_name, model_file)

        # Get Ibis expression
        depends_on = getattr(model_func, "depends_on", [])
        references = []
        for r in depends_on:
            if isinstance(r, source):
                columns = source_infos[r.source_name][r.table_name].columns
            elif isinstance(r, ref):
                columns = model_infos[r.model_name].columns
            else:
                raise ValueError(f"Unknown reference type: {type(r)}")
            schema = {
                c.name: _parse_db_dtype_to_ibis_dtype(c.data_type)
                for c in columns.values()
            }
            references.append(r.to_ibis(schema))
        model = model_func(*references)

        # Convert to SQL and write to file
        dbt_sql = _to_dbt_sql(model)
        dbt_model_file.write_text(dbt_sql)


def _create_empty_placeholder_files(model_file_paths: list[Path]) -> list[Path]:
    dbt_model_file_paths: list[Path] = []
    for model_file in model_file_paths:
        model_name = model_file.stem
        model_folder = model_file.parent
        dbt_sql_folder = model_folder / "__ibis_sql"
        dbt_sql_folder.mkdir(parents=False, exist_ok=True)
        dbt_model_file = dbt_sql_folder / f"{model_name}.sql"
        if dbt_model_file.exists():
            dbt_model_file.unlink()
        dbt_model_file.touch(exist_ok=False)
        dbt_model_file_paths.append(dbt_model_file)
    return dbt_model_file_paths


def _get_dbt_manifest() -> Manifest:
    res: dbtRunnerResult = dbtRunner().invoke(["parse"])
    manifest: Manifest = res.result
    return manifest


def _get_model_func(
    model_name: str, model_file: Path
) -> Callable[..., ibis.expr.types.Table]:
    spec = spec_from_loader(model_name, SourceFileLoader(model_name, str(model_file)))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_func = model_module.model
    return model_func


def _get_model_and_source_infos() -> tuple[dict, dict]:
    dbt_manifest = _get_dbt_manifest()
    nodes = list(dbt_manifest.nodes.values())
    models = [n for n in nodes if n.resource_type.name == "Model"]
    models_lookup = {m.name: m for m in models}

    sources = dbt_manifest.sources.values()
    sources_lookup = defaultdict(dict)
    for s in sources:
        sources_lookup[s.source_name][s.name] = s
    return models_lookup, dict(sources_lookup)


def _parse_db_dtype_to_ibis_dtype(db_dtype: str) -> dt.DataType:
    # Needs to be made dialect specific
    from ibis.backends.duckdb.datatypes import parse

    return parse(db_dtype)


def _to_dbt_sql(model):
    sql = ibis.to_sql(model)
    capture_pattern = "(.+?)"

    # Insert ref jinja function
    sql = re.sub(
        _REF_IDENTIFIER_PREFIX + capture_pattern + _REF_IDENTIFIER_SUFFIX,
        r"{{ ref('\1') }}",
        sql,
    )

    # Insert source jinja function
    sql = re.sub(
        _SOURCE_IDENTIFIER_PREFIX
        + capture_pattern
        + _SOURCE_IDENTIFIER_SEPARATOR
        + capture_pattern
        + _SOURCE_IDENTIFIER_SUFFIX,
        r"{{ source('\1', '\2') }}",
        sql,
    )
    return sql


def main():
    _compile_ibis_models()
    process = subprocess.Popen(
        ["dbt"] + sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr  # noqa: S603
    )
    process.wait()


if __name__ == "__main__":
    main()
