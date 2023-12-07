__all__ = ["ref", "source", "depends_on", "compile_ibis_to_sql"]
__version__ = "0.8.0dev"

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
_IBIS_FILE_EXTENSION: Final = "ibis"
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
    """A reference to a dbt model."""

    name: str

    @property
    def _ibis_table_name(self) -> str:
        return _REF_IDENTIFIER_PREFIX + self.name + _REF_IDENTIFIER_SUFFIX


@dataclass
class source(_Reference):
    """A reference to a dbt source."""

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
    """Decorator to specify the dependencies of an Ibis model. You can pass
    either dbt_ibis.ref or dbt_ibis.source objects as arguments.
    """
    if not all(isinstance(r, _Reference) for r in references):
        raise ValueError(
            "All arguments to depends_on need to be either an instance of"
            + " dbt_ibis.ref or dbt_ibis.source"
        )

    def decorator(
        func: Callable[..., ibis.expr.types.Table],
    ) -> Callable[..., ibis.expr.types.Table]:
        @wraps(func)
        def wrapper(*args: _Reference, **kwargs: _Reference) -> ibis.expr.types.Table:
            return func(*args, **kwargs)

        wrapper.depends_on = references  # type: ignore[attr-defined]
        return wrapper

    return decorator


@dataclass
class _IbisExprInfo:
    ibis_path: Path
    depends_on: list[_Reference]
    func: Callable[..., ibis.expr.types.Table]

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
    ctx: click.Context,
    **kwargs: Any,  # noqa: ARG001
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


def compile_ibis_to_sql(dbt_parse_arguments: Optional[list[str]] = None) -> None:
    """Compiles all Ibis code to SQL and writes them to the .sql files
    in the dbt project. There is no need to call this function directly as
    you'd usually use the dbt-ibis command line interface instead. This function
    is equivalent to `dbt-ibis precompile`. However, it is
    provided for convenience in case you want to call it from Python.
    """
    logger.info("Parse dbt project")
    with _disable_node_not_found_error():
        manifest, runtime_config = _invoke_parse_customized(dbt_parse_arguments)

    ibis_dialect = _dialects.get_ibis_dialect(manifest)

    project_root = runtime_config.project_root
    # We can treat models and singular tests as equivalent for the purpose
    # of compiling Ibis expressions to SQL.
    paths = runtime_config.model_paths + runtime_config.test_paths

    all_ibis_expr_infos = _get_ibis_expr_infos(
        project_root=project_root,
        paths=paths,
    )
    if len(all_ibis_expr_infos) == 0:
        logger.info("No Ibis expressions found.")
        return
    else:
        logger.info(f"Compiling {len(all_ibis_expr_infos)} Ibis expressions to SQL")

        # Order Ibis expressions by their dependencies so that the once which
        # depend on other Ibis expressions are compiled after the ones they depend on.
        # For example, if model_a depends on model_b, then model_b will be compiled
        # first and will appear in the list before model_a.
        all_ibis_expr_infos = _sort_ibis_exprs_by_dependencies(all_ibis_expr_infos)

        ref_infos, source_infos = _extract_ref_and_source_infos(manifest)

        letter_case_in_db, letter_case_in_expr = _get_letter_case_conversion_rules(
            runtime_config
        )

        # Schemas of the Ibis expressions themselves in case they are referenced
        # by other downstream Ibis expressions
        ibis_expr_schemas: dict[str, ibis.Schema] = {}
        # Convert Ibis expressions to SQL and write to file
        for ibis_expr_info in all_ibis_expr_infos:
            references: list[ibis.expr.types.Table] = []
            for r in ibis_expr_info.depends_on:
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
                        ibis_expr_schemas=ibis_expr_schemas,
                        ibis_dialect=ibis_dialect,
                        letter_case_in_db=letter_case_in_db,
                    )
                else:
                    raise ValueError(f"Unknown reference type: {type(r)}")
                ibis_table = r.to_ibis(schema)
                ibis_table = _set_letter_case_on_ibis_expression(
                    ibis_table, letter_case_in_expr
                )

                references.append(ibis_table)
            ibis_expr = ibis_expr_info.func(*references)
            ibis_expr = _set_letter_case_on_ibis_expression(
                ibis_expr, letter_case_in_db
            )

            ibis_expr_schemas[ibis_expr_info.name] = ibis_expr.schema()

            # Convert to SQL and write to file
            dbt_sql = _to_dbt_sql(ibis_expr, ibis_dialect=ibis_dialect)
            ibis_expr_info.sql_path.parent.mkdir(parents=False, exist_ok=True)
            ibis_expr_info.sql_path.write_text(dbt_sql)

        _clean_up_unused_sql_files(
            [ibis_expr_info.sql_path for ibis_expr_info in all_ibis_expr_infos],
            project_root=project_root,
            paths=paths,
        )
        logger.info("Finished compiling Ibis expressions to SQL")


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
    # as they come from the fact that Ibis expressions might not yet be present as .sql
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


def _get_ibis_expr_infos(
    project_root: Union[str, Path], paths: list[str]
) -> list[_IbisExprInfo]:
    ibis_files = _glob_in_paths(
        project_root=project_root,
        paths=paths,
        pattern=f"**/*.{_IBIS_FILE_EXTENSION}",
    )
    ibis_expr_infos: list[_IbisExprInfo] = []
    for file in ibis_files:
        func = _get_expr_func(file)
        depends_on = getattr(func, "depends_on", [])
        ibis_expr_infos.append(
            _IbisExprInfo(ibis_path=file, depends_on=depends_on, func=func)
        )
    return ibis_expr_infos


def _glob_in_paths(
    project_root: Union[str, Path], paths: list[str], pattern: str
) -> list[Path]:
    if isinstance(project_root, str):
        project_root = Path(project_root)

    matches: list[Path] = []
    for m_path in paths:
        matches.extend(list((project_root / m_path).glob(pattern)))
    return matches


def _get_expr_func(file: Path) -> Callable[..., ibis.expr.types.Table]:
    # Name arguments to spec_from_loader and SourceFileLoader probably don't matter
    # but maybe a good idea to keep them unique across the expressions
    spec = spec_from_loader(file.stem, SourceFileLoader(file.stem, str(file)))
    if spec is None:
        raise ValueError(f"Could not load file: {file}")
    expr_module = module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"Could not load file: {file}")
    spec.loader.exec_module(expr_module)
    func = getattr(expr_module, "model", None) or getattr(expr_module, "test", None)
    if func is None:
        raise ValueError(
            f"Could not find function called 'model' or 'test' in {str(file)}."
        )
    return func


def _sort_ibis_exprs_by_dependencies(
    ibis_exprs: list[_IbisExprInfo],
) -> list[_IbisExprInfo]:
    ibis_expr_lookup = {m.name: m for m in ibis_exprs}

    # Only look at ref. source references are not relevant for this sorting
    # as they already exist -> Don't need to compile them. Also no need to consider
    # refs which are not Ibis expressions
    graph = {
        ibis_expr_name: [
            d.name
            for d in ibis_expr.depends_on
            if isinstance(d, ref) and d.name in ibis_expr_lookup
        ]
        for ibis_expr_name, ibis_expr in ibis_expr_lookup.items()
    }
    sorter = graphlib.TopologicalSorter(graph)
    ibis_expr_order = list(sorter.static_order())

    return [ibis_expr_lookup[m] for m in ibis_expr_order]


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
    in_expr_var_name = "dbt_ibis_letter_case_in_expr"

    in_db_raw = dbt_project_vars.get(in_db_var_name, None)
    in_expr_raw = dbt_project_vars.get(in_expr_var_name, None)
    in_db = _validate_letter_case_var(in_db_var_name, in_db_raw)
    in_expr = _validate_letter_case_var(in_expr_var_name, in_expr_raw)
    return in_db, in_expr


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
    ibis_expr_schemas: dict[str, ibis.Schema],
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
    # in ibis_expr_schemas
    if schema is None and ref.name in ibis_expr_schemas:
        schema = ibis_expr_schemas[ref.name]

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


def _clean_up_unused_sql_files(
    used_sql_files: list[Path],
    project_root: Union[str, Path],
    paths: list[str],
) -> None:
    """Deletes all .sql files in any of the _IBIS_SQL_FOLDER_NAME folders which
    are not referenced by any Ibis expression. This takes care of the case where
    a user deletes an Ibis expression but the .sql file remains.
    """
    all_sql_files = _glob_in_paths(
        project_root=project_root,
        paths=paths,
        pattern=f"**/{_IBIS_SQL_FOLDER_NAME}/*.sql",
    )
    # Resolve to absolute paths so we can compare them
    all_sql_files = [f.resolve() for f in all_sql_files]
    used_sql_files = [f.resolve() for f in used_sql_files]
    unused_sql_files = [f for f in all_sql_files if f not in used_sql_files]
    if unused_sql_files:
        for f in unused_sql_files:
            f.unlink()
        logger.info(
            f"Cleaned up {len(unused_sql_files)} unused .sql files"
            + f" in your {_IBIS_SQL_FOLDER_NAME} folders"
        )


def main() -> None:
    dbt_subcommand, dbt_parse_arguments = _parse_cli_arguments()
    if dbt_subcommand != "deps":
        # If it's deps, we cannot yet parse the dbt project as it will raise
        # an error due to missing dependencies. We also don't need to so that's fine
        compile_ibis_to_sql(dbt_parse_arguments)
    if dbt_subcommand != "precompile":
        # Execute the actual dbt command
        process = subprocess.run(
            ["dbt"] + sys.argv[1:],  # noqa: S603
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()
