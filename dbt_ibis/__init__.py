__all__ = ["ref", "source", "depends_on", "compile_ibis_to_sql_models"]
__version__ = "0.2.0dev"

import graphlib
import re
import subprocess
import sys
from abc import ABC, abstractproperty
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Any, Callable, Final, Literal, Optional, Union

import click
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types
from dbt.cli.main import cli, p, requires
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ColumnInfo, ModelNode, SourceDefinition
from dbt.parser import manifest

_REF_IDENTIFIER_PREFIX: Final = "__ibd_ref__"
_REF_IDENTIFIER_SUFFIX: Final = "__rid__"
_SOURCE_IDENTIFIER_PREFIX: Final = "__ibd_source__"
_SOURCE_IDENTIFIER_SUFFIX: Final = "__sid__"
_SOURCE_IDENTIFIER_SEPARATOR: Final = "__ibd_sep__"
_IBIS_MODEL_FILE_EXTENSION: Final = "ibis"
_IBIS_SQL_FOLDER_NAME: Final = "__ibis_sql"

_ModelsLookup = dict[str, ModelNode]
_SourcesLookup = dict[str, dict[str, SourceDefinition]]


class _Reference(ABC):
    @abstractproperty
    def _ibis_table_name(self) -> str:
        pass

    def to_ibis(
        self, schema: Union[ibis.Schema, dict[str, dt.DataType]]
    ) -> ibis.expr.types.Table:
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


# Type hints could be improved here. Could use a typing.Protocol with a typed __call__
# method to indicate that the function that is wrapped by depends_on needs to be
# callable, accept a variadic number of _Reference arguments and needs to
# return an ibis Table
def depends_on(*references: _Reference) -> Callable:
    if not all(isinstance(r, _Reference) for r in references):
        raise ValueError(
            "All arguments to depends_on need to be either an instance of"
            + " dbt_ibis.ref or dbt_ibis.source"
        )

    def decorator(
        func: Callable[..., ibis.expr.types.Table]
    ) -> Callable[..., ibis.expr.types.Table]:
        @wraps(func)
        def wrapper(*args: _Reference, **kwargs: _Reference) -> ibis.expr.types.Table:
            return func(*args, **kwargs)

        wrapper.depends_on = references  # type: ignore[attr-defined]
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
        return self.ibis_path.parent / _IBIS_SQL_FOLDER_NAME / f"{self.name}.sql"


@cli.command(
    "parse_customized",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.pass_context
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.postflight
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest(write_perf_info=True)
def _parse_customized(
    ctx: click.Context, **kwargs: Any  # noqa: ARG001
) -> tuple[tuple[Manifest, RuntimeConfig], Literal[True]]:
    # This is a slightly modified version of the dbt parse command
    # which, in addition to the manifest, also returns the runtime_config
    # Would be nice if we can instead directly use the dbt parse command
    # as it might be difficult to keep this command in sync with the dbt parse command
    # We also ignore unknown options and allow extra arguments so that we can just
    # pass all arguments which are for the actual dbt command to dbt parse
    # without having to filter out the relevant ones

    return (ctx.obj["manifest"], ctx.obj["runtime_config"]), True


@contextmanager
def _disable_node_not_found_error() -> Iterator[None]:
    original_func = manifest.invalid_target_fail_unless_test

    def _do_nothing(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        pass

    try:
        manifest.invalid_target_fail_unless_test = _do_nothing
        yield
    finally:
        manifest.invalid_target_fail_unless_test = original_func


def compile_ibis_to_sql_models() -> None:
    # dbt cannot yet see the Ibis models as they are not yet compiled
    # Hence, we need to disable the error which is raised if dbt cannot find
    # all referenced nodes while building the manifest.
    # The manifest that you get back below will therefore also not contain
    # the Ibis models which is fine for our purposes as we need information on the
    # other models and the project in general.
    with _disable_node_not_found_error():
        manifest, runtime_config = _invoke_parse_customized()
    all_ibis_models = _get_ibis_models(
        project_root=runtime_config.project_root, model_paths=runtime_config.model_paths
    )

    # Order Ibis models by their dependencies so that the once which depend on other
    # Ibis model are compiled after the ones they depend on. For example, if
    # model_a depends on model_b, then model_b will be compiled first and will appear
    # in the list before model_a.
    all_ibis_models = _sort_ibis_models_by_dependencies(all_ibis_models)

    model_infos, source_infos = _extract_model_and_source_infos(manifest)

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
        ibis_model.sql_path.parent.mkdir(parents=False, exist_ok=True)
        ibis_model.sql_path.write_text(dbt_sql)


def _invoke_parse_customized() -> tuple[Manifest, RuntimeConfig]:
    args = _get_parse_arguments()
    dbt_ctx = cli.make_context(cli.name, args)
    result, success = cli.invoke(dbt_ctx)
    if not success:
        raise ValueError("Could not parse dbt project")
    return result


def _get_parse_arguments() -> list[str]:
    # First argument of sys.argv is path to this file. We then look for
    # the name of the actual dbt subcommand that the user wants to run and ignore
    # any global flags that come before it. All subsequent arguments are passed to
    # _parse_customized so that a user can e.g. set --project-dir etc.
    # For example, "dbt-ibis --warn-error run --select stg_orders --project-dir folder"
    # becomes "parse_customized run --select stg_orders --project-dir folder"
    # in variable args. parse_customized will then ignore "--select stg_orders"
    all_args = sys.argv[1:]
    subcommand_idx = next(i for i, arg in enumerate(all_args) if arg in cli.commands)
    parse_command = _parse_customized.name
    # For the benefit of mypy
    assert isinstance(parse_command, str)  # noqa: S101
    args = [parse_command] + all_args[subcommand_idx + 1 :]
    return args


def _get_ibis_models(
    project_root: Union[str, Path], model_paths: list[str]
) -> list[_IbisModel]:
    if isinstance(project_root, str):
        project_root = Path(project_root)

    ibis_model_files: list[Path] = []
    for m_path in model_paths:
        ibis_model_files.extend(
            list((project_root / m_path).glob(f"**/*.{_IBIS_MODEL_FILE_EXTENSION}"))
        )
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


def _get_model_func(model_file: Path) -> Callable[..., ibis.expr.types.Table]:
    # Name arguments to spec_from_loader and SourceFileLoader probably don't matter
    # but maybe a good idea to keep them unique across the models
    spec = spec_from_loader(
        model_file.stem, SourceFileLoader(model_file.stem, str(model_file))
    )
    if spec is None:
        raise ValueError(f"Could not load model file: {model_file}")
    model_module = module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"Could not load model file: {model_file}")
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


def _extract_model_and_source_infos(
    dbt_manifest: Manifest,
) -> tuple[_ModelsLookup, _SourcesLookup]:
    nodes = list(dbt_manifest.nodes.values())
    models = [n for n in nodes if isinstance(n, ModelNode)]
    models_lookup = {m.name: m for m in models}

    sources = dbt_manifest.sources.values()
    sources_lookup: defaultdict[str, dict[str, SourceDefinition]] = defaultdict(dict)
    for s in sources:
        sources_lookup[s.source_name][s.name] = s
    return models_lookup, dict(sources_lookup)


def _get_schema_for_source(
    source: source,
    source_infos: _SourcesLookup,
) -> ibis.Schema:
    columns = source_infos[source.source_name][source.table_name].columns
    return _columns_to_ibis_schema(columns)


def _get_schema_for_ref(
    ref: ref, model_infos: _ModelsLookup, ibis_model_schemas: dict[str, ibis.Schema]
) -> ibis.Schema:
    schema: Optional[ibis.Schema] = None
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
    schema_dict: dict[str, dt.DataType] = {}
    for c in columns.values():
        if c.data_type is None:
            raise ValueError(f"Could not determine data type for column '{c.name}'")
        schema_dict[c.name] = _parse_db_dtype_to_ibis_dtype(c.data_type)
    return ibis.schema(schema_dict)


def _parse_db_dtype_to_ibis_dtype(db_dtype: str) -> dt.DataType:
    # Needs to be made dialect specific
    from ibis.backends.duckdb.datatypes import parse

    return parse(db_dtype)


def _to_dbt_sql(ibis_expr: ibis.expr.types.Table) -> str:
    # Return type of .to_sql is SqlString which is a normal string with some
    # custom repr methods -> Convert it to a normal string here to make it easier
    # for mypy.
    sql = str(ibis.to_sql(ibis_expr))
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


def main() -> None:
    compile_ibis_to_sql_models()
    # Execute the actual dbt command
    process = subprocess.run(
        ["dbt"] + sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr  # noqa: S603
    )
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
