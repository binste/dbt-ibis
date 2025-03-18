__all__ = ["compile_ibis_to_sql", "depends_on", "ref", "source"]
__version__ = "0.11.0"

import logging
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Callable, Optional, Union

import ibis.expr.types as ir
from dbt.cli.main import cli

from dbt_ibis import _dialects
from dbt_ibis._compile import IBIS_FILE_EXTENSION as _IBIS_FILE_EXTENSION
from dbt_ibis._compile import IBIS_SQL_FOLDER_NAME as _IBIS_SQL_FOLDER_NAME
from dbt_ibis._compile import IbisExprInfo as _IbisExprInfo
from dbt_ibis._compile import (
    compile_ibis_expressions_to_sql as _compile_ibis_expressions_to_sql,
)
from dbt_ibis._logging import configure_logging as _configure_logging
from dbt_ibis._parse_dbt_project import (
    disable_node_not_found_error as _disable_node_not_found_error,
)
from dbt_ibis._parse_dbt_project import (
    invoke_parse_customized as _invoke_parse_customized,
)
from dbt_ibis._references import (
    depends_on,
    ref,
    source,
)

logger = logging.getLogger(__name__)
_configure_logging(logger)


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

        _compile_ibis_expressions_to_sql(
            all_ibis_expr_infos=all_ibis_expr_infos,
            manifest=manifest,
            ibis_dialect=ibis_dialect,
            runtime_config=runtime_config,
        )

        _clean_up_unused_sql_files(
            [ibis_expr_info.sql_path for ibis_expr_info in all_ibis_expr_infos],
            project_root=project_root,
            paths=paths,
        )
        logger.info("Finished compiling Ibis expressions to SQL")


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


def _get_expr_func(file: Path) -> Callable[..., ir.Table]:
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
        process = subprocess.run(  # noqa: S603
            ["dbt"] + sys.argv[1:],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()
