__all__ = ["ref", "source", "depends_on", "compile_ibis_to_sql_models"]
__version__ = "0.5.0"

import graphlib
import logging
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
from dbt.contracts.graph.nodes import (
    ColumnInfo,
    ModelNode,
    SeedNode,
    SnapshotNode,
    SourceDefinition,
)
from dbt.parser import manifest

from dbt_ibis import _dialects

_REF_IDENTIFIER_PREFIX: Final = "__ibd_ref__"
_REF_IDENTIFIER_SUFFIX: Final = "__rid__"
_SOURCE_IDENTIFIER_PREFIX: Final = "__ibd_source__"
_SOURCE_IDENTIFIER_SUFFIX: Final = "__sid__"
_SOURCE_IDENTIFIER_SEPARATOR: Final = "__ibd_sep__"
_IBIS_MODEL_FILE_EXTENSION: Final = "ibis"
_IBIS_SQL_FOLDER_NAME: Final = "__ibis_sql"

_RefLookup = dict[str, Union[ModelNode, SeedNode, SnapshotNode]]
_SourcesLookup = dict[str, dict[str, SourceDefinition]]
_LetterCase = Literal["lower", "upper"]


def _configure_logging(logger: logging.Logger) -> None:
    log_level = logging.INFO
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Imitate dbt's log format but add dbt-ibis before the log message
    formatter = logging.Formatter(
        "%(asctime)s  dbt-ibis: %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


logger = logging.getLogger(__name__)
_configure_logging(logger)


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
    name: str

    @property
    def _ibis_table_name(self) -> str:
        return _REF_IDENTIFIER_PREFIX + self.name + _REF_IDENTIFIER_SUFFIX


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


def _back_up_and_restore_target_files(func: Callable) -> Callable:
    """Backs up and then restores again all files which are in the target
    folder. Ignores the folders "compiled" and "run".

    Reason is that we want to prevent dbt from reusing the generated artifacts from
    an incomplete dbt parse command for subsequent commands as those
    artifacts are based on an incomplete dbt project (i.e. the missing compiled
    Ibis models). If, for example, an Ibis model is referenced in a .yml file but
    does not yet exist, dbt will disable the model in the manifest. Any subsequent
    dbt command might not recreate the manifest due to partial parsing.

    This can't be replaced by the global --no-write-json flag as dbt will then
    still write partial_parse.msgpack.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        ctx = args[0]
        if not isinstance(ctx, click.Context):
            raise ValueError(
                "First argument needs to be a click context. Please report"
                + " this error in the GitHub repository."
            )
        runtime_config: RuntimeConfig = ctx.obj["runtime_config"]
        target_path = Path(runtime_config.project_root) / runtime_config.target_path

        def get_files_to_backup(target_path: Path) -> list[Path]:
            all_files = target_path.rglob("*")
            # Don't backup the compiled and run folders as they are not modified
            # by the dbt parse command and this speeds up the whole process.
            # Some other files in the top-level target folder are also not modified
            # by dbt parse but it's more future-proof if we just back up all other files
            # in case this is changed in the future.
            folders_to_exclude = [target_path / "compiled", target_path / "run"]
            files_to_backup = [
                f
                for f in all_files
                if f.is_file()
                and not any(folder in f.parents for folder in folders_to_exclude)
            ]
            return files_to_backup

        files_to_backup = get_files_to_backup(target_path)
        backups = {f: f.read_bytes() for f in files_to_backup}
        try:
            return func(*args, **kwargs)
        finally:
            for f, content in backups.items():
                f.write_bytes(content)

            # Remove any files which would have been backed up but didn't exist before
            for f in get_files_to_backup(target_path):
                if f not in backups:
                    f.unlink()

    return wrapper


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
@_back_up_and_restore_target_files
@requires.manifest(write_perf_info=False)
def _parse_customized(
    ctx: click.Context, **kwargs: Any  # noqa: ARG001
) -> tuple[tuple[Manifest, RuntimeConfig], Literal[True]]:
    # This is a slightly modified version of the dbt parse command
    # which:
    # * in addition to the manifest, also returns the runtime_config
    #     Would be nice if we can instead directly use the dbt parse command
    #     as it might be difficult to keep this command in sync with the dbt parse
    #     command.
    # * ignores unknown options and allow extra arguments so that we can just
    #     pass all arguments which are for the actual dbt command to dbt parse
    #     without having to filter out the relevant ones.
    # * Backs up and then restores again all files which are in the target folder
    #     so that this command doesnot have any side effects. See the docstring
    #     of _back_up_and_restore_target_files for more details.

    return (ctx.obj["manifest"], ctx.obj["runtime_config"]), True


@contextmanager
def _disable_node_not_found_error() -> Iterator[None]:
    """A dbt parse command will raise an error if it cannot find all referenced nodes.
    This context manager disables the error as dbt cannot yet see the Ibis models
    as they are not yet compiled at this point and hence will raise an error.
    """
    original_func = manifest.invalid_target_fail_unless_test

    def _do_nothing(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        pass

    try:
        manifest.invalid_target_fail_unless_test = _do_nothing
        yield
    finally:
        manifest.invalid_target_fail_unless_test = original_func


def compile_ibis_to_sql_models(dbt_parse_arguments: Optional[list[str]] = None) -> None:
    logger.info("Parse dbt project")
    with _disable_node_not_found_error():
        manifest, runtime_config = _invoke_parse_customized(dbt_parse_arguments)

    ibis_dialect = _dialects.get_ibis_dialect(manifest)

    all_ibis_models = _get_ibis_models(
        project_root=runtime_config.project_root,
        model_paths=runtime_config.model_paths,
    )
    if len(all_ibis_models) == 0:
        logger.info("No Ibis models found.")
        return
    else:
        logger.info(f"Compiling {len(all_ibis_models)} Ibis models to SQL")

        # Order Ibis models by their dependencies so that the once which depend on other
        # Ibis model are compiled after the ones they depend on. For example, if
        # model_a depends on model_b, then model_b will be compiled first and will
        # appear in the list before model_a.
        all_ibis_models = _sort_ibis_models_by_dependencies(all_ibis_models)

        ref_infos, source_infos = _extract_ref_and_source_infos(manifest)

        letter_case_in_db, letter_case_in_model = _get_letter_case_conversion_rules(
            runtime_config
        )

        # Schemas of the Ibis models themselves in case they are referenced
        # by other downstream Ibis models
        ibis_model_schemas: dict[str, ibis.Schema] = {}
        # Convert Ibis models to SQL and write to file
        for ibis_model in all_ibis_models:
            references: list[ibis.expr.types.Table] = []
            for r in ibis_model.depends_on:
                if isinstance(r, source):
                    schema = _get_schema_for_source(
                        r,
                        source_infos,
                        ibis_dialect=ibis_dialect,
                        letter_case_in_db=letter_case_in_db,
                    )
                elif isinstance(r, ref):
                    schema = _get_schema_for_ref(
                        r,
                        ref_infos,
                        ibis_model_schemas=ibis_model_schemas,
                        ibis_dialect=ibis_dialect,
                        letter_case_in_db=letter_case_in_db,
                    )
                else:
                    raise ValueError(f"Unknown reference type: {type(r)}")
                ibis_table = r.to_ibis(schema)
                ibis_table = _set_letter_case_on_ibis_expression(
                    ibis_table, letter_case_in_model
                )

                references.append(ibis_table)
            ibis_expr = ibis_model.model_func(*references)
            ibis_expr = _set_letter_case_on_ibis_expression(
                ibis_expr, letter_case_in_db
            )

            ibis_model_schemas[ibis_model.name] = ibis_expr.schema()

            # Convert to SQL and write to file
            dbt_sql = _to_dbt_sql(ibis_expr, ibis_dialect=ibis_dialect)
            ibis_model.sql_path.parent.mkdir(parents=False, exist_ok=True)
            ibis_model.sql_path.write_text(dbt_sql)
        logger.info("Finished compiling Ibis models to SQL")


def _set_letter_case_on_ibis_expression(
    ibis_expr: ibis.expr.types.Table, letter_case: Optional[_LetterCase]
) -> ibis.expr.types.Table:
    if letter_case == "upper":
        ibis_expr = ibis_expr.rename("ALL_CAPS")
    elif letter_case == "lower":
        ibis_expr = ibis_expr.rename("snake_case")
    return ibis_expr


def _invoke_parse_customized(
    dbt_parse_arguments: Optional[list[str]],
) -> tuple[Manifest, RuntimeConfig]:
    dbt_parse_arguments = dbt_parse_arguments or []
    parse_command = _parse_customized.name
    # For the benefit of mypy
    assert isinstance(parse_command, str)  # noqa: S101
    # Use --quiet to suppress non-error logs in stdout. These logs would be
    # confusing to a user as they don't expect two dbt commands to be executed.
    # Furthermore, the logs might contain warnings which the user can ignore
    # as they come from the fact that Ibis models might not yet be present as .sql
    # files when running the parse command.
    args = ["--quiet", parse_command, *dbt_parse_arguments]

    dbt_ctx = cli.make_context(cli.name, args)
    result, success = cli.invoke(dbt_ctx)
    if not success:
        raise ValueError("Could not parse dbt project")
    return result


def _parse_cli_arguments() -> tuple[str, list[str]]:
    # First argument of sys.argv is path to this file. We then look for
    # the name of the actual dbt subcommand that the user wants to run and ignore
    # any global flags that come before it.
    # We return the subcommand as well as separately in a list, all subsequent
    # arguments which can then be passed to
    # _parse_customized so that a user can e.g. set --project-dir etc.
    # For example, "dbt-ibis --warn-error run --select stg_orders --project-dir folder"
    # becomes "--select stg_orders --project-dir folder"
    # in variable args. parse_customized will then ignore "--select stg_orders"
    all_args = sys.argv[1:]
    subcommand_idx = next(
        i
        for i, arg in enumerate(all_args)
        if arg in [*list(cli.commands.keys()), "precompile"]
    )
    args = all_args[subcommand_idx + 1 :]
    subcommand = all_args[subcommand_idx]
    return subcommand, args


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
    model_func = getattr(model_module, "model", None)
    if model_func is None:
        raise ValueError(
            f"Could not find function called 'model' in {str(model_file)}."
        )
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
            d.name
            for d in ibis_model.depends_on
            if isinstance(d, ref) and d.name in ibis_model_lookup
        ]
        for ibis_model_name, ibis_model in ibis_model_lookup.items()
    }
    sorter = graphlib.TopologicalSorter(graph)
    ibis_model_order = list(sorter.static_order())

    return [ibis_model_lookup[m] for m in ibis_model_order]


def _extract_ref_and_source_infos(
    dbt_manifest: Manifest,
) -> tuple[_RefLookup, _SourcesLookup]:
    nodes = list(dbt_manifest.nodes.values())
    models_and_seeds = [
        n for n in nodes if isinstance(n, (ModelNode, SeedNode, SnapshotNode))
    ]
    ref_lookup = {m.name: m for m in models_and_seeds}

    sources = dbt_manifest.sources.values()
    sources_lookup: defaultdict[str, dict[str, SourceDefinition]] = defaultdict(dict)
    for s in sources:
        sources_lookup[s.source_name][s.name] = s
    return ref_lookup, dict(sources_lookup)


def _get_letter_case_conversion_rules(
    runtime_config: RuntimeConfig,
) -> tuple[Optional[_LetterCase], Optional[_LetterCase]]:
    # Variables as defined in e.g. dbt_project.yml
    dbt_project_vars = runtime_config.vars.vars
    project_name = runtime_config.project_name
    target_name = runtime_config.target_name

    in_db_var_name = f"dbt_ibis_letter_case_in_db_{project_name}_{target_name}"
    in_model_var_name = "dbt_ibis_letter_case_in_model"

    in_db_raw = dbt_project_vars.get(in_db_var_name, None)
    in_model_raw = dbt_project_vars.get(in_model_var_name, None)
    in_db = _validate_letter_case_var(in_db_var_name, in_db_raw)
    in_model = _validate_letter_case_var(in_model_var_name, in_model_raw)
    return in_db, in_model


def _validate_letter_case_var(variable_name: str, value: Any) -> Optional[_LetterCase]:
    if value is not None and value not in ["lower", "upper"]:
        raise ValueError(
            f"The {variable_name} variable needs to be set to"
            + f" either 'lower' or 'upper' but currently has a value of '{value}'."
            + " If you want the default behaviour of Ibis, you can omit this variable."
        )
    return value


def _get_schema_for_source(
    source: source,
    source_infos: _SourcesLookup,
    ibis_dialect: _dialects.IbisDialect,
    letter_case_in_db: Optional[_LetterCase] = None,
) -> ibis.Schema:
    columns = source_infos[source.source_name][source.table_name].columns
    return _columns_to_ibis_schema(
        columns,
        ibis_dialect=ibis_dialect,
        letter_case_in_db=letter_case_in_db,
    )


def _get_schema_for_ref(
    ref: ref,
    ref_infos: _RefLookup,
    ibis_model_schemas: dict[str, ibis.Schema],
    ibis_dialect: _dialects.IbisDialect,
    letter_case_in_db: Optional[_LetterCase] = None,
) -> ibis.Schema:
    schema: Optional[ibis.Schema] = None
    columns_with_missing_data_types: list[ColumnInfo] = []
    # Take column data types and hence schema from parsed model infos if available
    # as this is the best source for the schema as it will appear in
    # the database if the user enforces the data type contracts.
    # However, this means that also all columns need to be defined in the yml files
    # which are used in the Ibis expression.
    if ref.name in ref_infos:
        columns: dict[str, ColumnInfo]
        info = ref_infos[ref.name]
        if isinstance(info, SeedNode):
            # For seeds, the information is not stored in the
            # columns attribute but in the config.column_types attribute
            columns = {
                name: ColumnInfo(name=name, data_type=data_type)
                for name, data_type in info.config.column_types.items()
            }
        else:
            columns = info.columns

        if len(columns) > 0:
            columns_with_missing_data_types = [
                c for c in columns.values() if c.data_type is None
            ]
            if not columns_with_missing_data_types:
                schema = _columns_to_ibis_schema(
                    columns,
                    ibis_dialect=ibis_dialect,
                    letter_case_in_db=letter_case_in_db,
                )
            # Do not yet raise an error if there are missing data types as we might
            # be able to get the schema from the Ibis model itself

    # Else, see if it is an Ibis model in which case we would have the schema
    # in ibis_model_schemas
    if schema is None and ref.name in ibis_model_schemas:
        schema = ibis_model_schemas[ref.name]

    if schema is None:
        if columns_with_missing_data_types:
            raise ValueError(
                f"The following columns of '{ref.name}' do not have"
                + " a data type configured: "
                + ", ".join("'" + c.name + "'" for c in columns_with_missing_data_types)
            )
        else:
            raise ValueError(
                f"Could not determine schema for model '{ref.name}'."
                + " You either need to define it as a model contract or"
                + " the model needs to be an Ibis model as well."
            )
    return schema


def _columns_to_ibis_schema(
    columns: dict[str, ColumnInfo],
    ibis_dialect: _dialects.IbisDialect,
    letter_case_in_db: Optional[_LetterCase] = None,
) -> ibis.Schema:
    schema_dict: dict[str, dt.DataType] = {}
    for c in columns.values():
        if c.data_type is None:
            raise ValueError(f"Could not determine data type for column '{c.name}'")
        column_name = c.name
        if letter_case_in_db is not None:
            column_name = getattr(column_name, letter_case_in_db)()
        schema_dict[column_name] = _dialects.parse_db_dtype_to_ibis_dtype(
            c.data_type, ibis_dialect=ibis_dialect
        )
    return ibis.schema(schema_dict)


def _to_dbt_sql(
    ibis_expr: ibis.expr.types.Table, ibis_dialect: _dialects.IbisDialect
) -> str:
    sql = _dialects.ibis_expr_to_sql(ibis_expr, ibis_dialect=ibis_dialect)
    capture_pattern = "(.+?)"

    # Remove quotation marks around the source name and table name as
    # quoting identifiers should be handled by DBT in case it is needed.
    quotation_marks_pattern = r'"?'

    # Insert ref jinja function
    sql = re.sub(
        quotation_marks_pattern
        + _REF_IDENTIFIER_PREFIX
        + capture_pattern
        + _REF_IDENTIFIER_SUFFIX
        + quotation_marks_pattern,
        r"{{ ref('\1') }}",
        sql,
    )

    # Insert source jinja function
    sql = re.sub(
        quotation_marks_pattern
        + _SOURCE_IDENTIFIER_PREFIX
        + capture_pattern
        + _SOURCE_IDENTIFIER_SEPARATOR
        + capture_pattern
        + _SOURCE_IDENTIFIER_SUFFIX
        + quotation_marks_pattern,
        r"{{ source('\1', '\2') }}",
        sql,
    )
    return sql


def main() -> None:
    dbt_subcommand, dbt_parse_arguments = _parse_cli_arguments()
    if dbt_subcommand != "deps":
        # If it's deps, we cannot yet parse the dbt project as it will raise
        # an error due to missing dependencies. We also don't need to so that's fine
        compile_ibis_to_sql_models(dbt_parse_arguments)
    if dbt_subcommand != "precompile":
        # Execute the actual dbt command
        process = subprocess.run(
            ["dbt"] + sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr  # noqa: S603
        )
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()
