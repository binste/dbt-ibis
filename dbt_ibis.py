import re
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.contracts.graph.manifest import Manifest

__version__ = "0.1.0dev"


_TABLE_IDENTIFIER_PREFIX = "_ibis_dbt__"
_TABLE_IDENTIFIER_SUFFIX = "__identifier_"
_IBIS_MODEL_FILE_EXTENSION = "ibis"


def _compile_ibis_models(
    dbt_model_folder: Path = Path("models"),
):
    model_file_paths = list(dbt_model_folder.glob(f"**/*.{_IBIS_MODEL_FILE_EXTENSION}"))

    # Create empty placeholder files so that we can run dbt parse
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

    dbt_manifest = _get_dbt_manifest()
    nodes = list(dbt_manifest.nodes.values())
    models = [n for n in nodes if n.resource_type.name == "Model"]
    models_lookup = {m.name: m for m in models}

    for model_file, dbt_model_file in zip(model_file_paths, dbt_model_file_paths):
        model_name = model_file.stem

        spec = spec_from_loader(
            model_name, SourceFileLoader(model_name, str(model_file))
        )
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Get Ibis expression
        model_func = model_module.model
        depends_on = getattr(model_func, "depends_on", [])
        references = []
        for r in depends_on:
            schema = {
                c.name: _parse_db_dtype_to_ibis_dtype(c.data_type)
                for c in models_lookup[r.name].columns.values()
            }
            references.append(r.to_ibis(schema))
        model = model_func(*references)

        # Convert to SQL and write to file
        dbt_sql = _to_dbt_sql(model)
        dbt_model_file.write_text(dbt_sql)


def _get_dbt_manifest() -> Manifest:
    res: dbtRunnerResult = dbtRunner().invoke(["parse"])
    manifest: Manifest = res.result
    return manifest


def _parse_db_dtype_to_ibis_dtype(db_dtype: str) -> dt.DataType:
    # Needs to be made dialect specific
    from ibis.backends.duckdb.datatypes import parse

    return parse(db_dtype)


class ref:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_ibis(self, schema) -> ibis.expr.types.Table:
        if schema is None:
            raise NotImplementedError
        return ibis.table(
            schema,
            name=f"{_TABLE_IDENTIFIER_PREFIX}{self.name}{_TABLE_IDENTIFIER_SUFFIX}",
        )


def depends_on(*references):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.depends_on = references
        return wrapper

    return decorator


def _to_dbt_sql(model):
    sql = ibis.to_sql(model)
    dbt_sql = re.sub(
        _TABLE_IDENTIFIER_PREFIX + "(.+?)" + _TABLE_IDENTIFIER_SUFFIX,
        r"{{ ref('\1') }}",
        sql,
    )
    return dbt_sql


def main():
    _compile_ibis_models()
    process = subprocess.Popen(
        ["dbt"] + sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr  # noqa: S603
    )
    process.wait()


if __name__ == "__main__":
    main()
