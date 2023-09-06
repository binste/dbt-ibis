import graphlib
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
from dbt.contracts.graph.nodes import ColumnInfo, ModelNode, SourceDefinition

__version__ = "0.1.0dev"


_REF_IDENTIFIER_PREFIX: Final = "__ibd_ref__"
_REF_IDENTIFIER_SUFFIX: Final = "__rid__"
_SOURCE_IDENTIFIER_PREFIX: Final = "__ibd_source__"
_SOURCE_IDENTIFIER_SUFFIX: Final = "__sid__"
_SOURCE_IDENTIFIER_SEPARATOR: Final = "__ibd_sep__"
_IBIS_MODEL_FILE_EXTENSION: Final = "ibis"

_ModelsLookup = dict[str, ModelNode]
_SourcesLookup = dict[str, dict[str, SourceDefinition]]


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


@dataclass
class _IbisModel:
    ibis_path: Path
    depends_on: list[_Reference]
    model_func: Callable[..., ibis.expr.types.Table]

    @property
    def name(self) -> str:
        return self.ibis_path.stem

    @property
    def sql_path(self) -> Path:
        return self.ibis_path.parent / "__ibis_sql" / f"{self.name}.sql"


def _compile_ibis_models(
    dbt_model_folder: Path = Path("models"),
) -> None:
    all_ibis_models = _get_ibis_models(dbt_model_folder=dbt_model_folder)

    # Order Ibis models by their dependencies so that the once which depend on other
    # Ibis model are compiled after the ones they depend on. For example, if
    # model_a depends on model_b, then model_b will be compiled first and will appear
    # in the list before model_a.
    all_ibis_models = _sort_ibis_models_by_dependencies(all_ibis_models)

    # Create empty placeholder file for every Ibis model so that we can run dbt parse
    _create_empty_placeholder_files(all_ibis_models)
    model_infos, source_infos = _get_model_and_source_infos()

    # Schemas of the Ibis models themselves in case they are referenced
    # by other downstream Ibis models
    ibis_model_schemas: dict[str, ibis.Schema] = {}
    # Convert Ibis models to SQL and write to file
    for ibis_model in all_ibis_models:
        references: list[ibis.expr.types.Table] = []
        for r in ibis_model.depends_on:
            if isinstance(r, source):
                schema = _get_schema_for_source(r, source_infos)
            elif isinstance(r, ref):
                schema = _get_schema_for_ref(
                    r, model_infos, ibis_model_schemas=ibis_model_schemas
                )
            else:
                raise ValueError(f"Unknown reference type: {type(r)}")
            ibis_table = r.to_ibis(schema)

            references.append(ibis_table)
        ibis_expr = ibis_model.model_func(*references)
        ibis_model_schemas[ibis_model.name] = ibis_expr.schema()

        # Convert to SQL and write to file
        dbt_sql = _to_dbt_sql(ibis_expr)
        ibis_model.sql_path.write_text(dbt_sql)


def _get_ibis_models(dbt_model_folder: Path) -> list[_IbisModel]:
    ibis_model_files = list(dbt_model_folder.glob(f"**/*.{_IBIS_MODEL_FILE_EXTENSION}"))
    ibis_models: list[_IbisModel] = []
    for model_file in ibis_model_files:
        model_func = _get_model_func(model_file)
        depends_on = getattr(model_func, "depends_on", [])
        ibis_models.append(
            _IbisModel(
                ibis_path=model_file, depends_on=depends_on, model_func=model_func
            )
        )
    return ibis_models


def _create_empty_placeholder_files(ibis_models: list[_IbisModel]) -> None:
    for model in ibis_models:
        sql_path = model.sql_path
        sql_path.parent.mkdir(parents=False, exist_ok=True)
        # If the file already exists, delete it as this makes it easier for a user if
        # that file contains an error which would prevent dbt from parsing it.
        # For example if the SQL file references a model which no longer exists.
        # This way, we can run dbt parse and then later on replace the errenous SQL file
        # with a new one.
        if sql_path.exists():
            sql_path.unlink()
        sql_path.touch(exist_ok=False)


def _get_dbt_manifest() -> Manifest:
    res: dbtRunnerResult = dbtRunner().invoke(["parse"])
    manifest: Manifest = res.result
    return manifest


def _get_model_func(model_file: Path) -> Callable[..., ibis.expr.types.Table]:
    # Name arguments to spec_from_loader and SourceFileLoader probably don't matter
    # but maybe a good idea to keep them unique across the models
    spec = spec_from_loader(
        model_file.stem, SourceFileLoader(model_file.stem, str(model_file))
    )
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_func = model_module.model
    return model_func


def _sort_ibis_models_by_dependencies(
    ibis_models: list[_IbisModel],
) -> list[_IbisModel]:
    ibis_model_lookup = {m.name: m for m in ibis_models}

    # Only look at ref. source references are not relevant for this sorting
    # as they already exist -> Don't need to compile them. Also no need to consider
    # refs which are not Ibis models
    graph = {
        ibis_model_name: [
            d.model_name
            for d in ibis_model.depends_on
            if isinstance(d, ref) and d.model_name in ibis_model_lookup
        ]
        for ibis_model_name, ibis_model in ibis_model_lookup.items()
    }
    sorter = graphlib.TopologicalSorter(graph)
    ibis_model_order = list(sorter.static_order())

    return [ibis_model_lookup[m] for m in ibis_model_order]


def _get_model_and_source_infos() -> tuple[_ModelsLookup, _SourcesLookup]:
    dbt_manifest = _get_dbt_manifest()
    nodes = list(dbt_manifest.nodes.values())
    models = [n for n in nodes if n.resource_type.name == "Model"]
    models_lookup = {m.name: m for m in models}

    sources = dbt_manifest.sources.values()
    sources_lookup = defaultdict(dict)
    for s in sources:
        sources_lookup[s.source_name][s.name] = s
    return models_lookup, dict(sources_lookup)


def _get_schema_for_source(
    source: source,
    source_infos: _SourcesLookup,
):
    columns = source_infos[source.source_name][source.table_name].columns
    return _columns_to_ibis_schema(columns)


def _get_schema_for_ref(
    ref: ref, model_infos: _SourcesLookup, ibis_model_schemas: dict[str, ibis.Schema]
) -> ibis.Schema:
    schema: ibis.Schema | None = None
    # Take column data types and hence schema from parsed model infos if available
    # as this is the best source for the schema as it will appear in
    # the database if the user enforces the data type contracts.
    # However, this means that also all columns need to be defined in the yml files
    # which are used in the Ibis expression
    if ref.model_name in model_infos:
        columns = model_infos[ref.model_name].columns
        has_columns_with_data_types = len(columns) > 0 and all(
            c.data_type is not None for c in columns.values()
        )
        if has_columns_with_data_types:
            schema = _columns_to_ibis_schema(columns)

    # Else, see if it is an Ibis model in which case we would have the schema
    # in ibis_model_schemas
    if schema is None and ref.model_name in ibis_model_schemas:
        schema = ibis_model_schemas[ref.model_name]

    if schema is None:
        raise ValueError(
            f"Could not determine schema for model '{ref.model_name}'."
            + " You either need to define it as a model contract or the model needs to"
            + " be an Ibis model as well"
        )
    return schema


def _columns_to_ibis_schema(columns: dict[str, ColumnInfo]) -> ibis.Schema:
    schema = ibis.schema(
        {c.name: _parse_db_dtype_to_ibis_dtype(c.data_type) for c in columns.values()}
    )
    return schema


def _parse_db_dtype_to_ibis_dtype(db_dtype: str) -> dt.DataType:
    # Needs to be made dialect specific
    from ibis.backends.duckdb.datatypes import parse

    return parse(db_dtype)


def _to_dbt_sql(ibis_expr: ibis.expr.types.Table) -> str:
    sql = ibis.to_sql(ibis_expr)
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
