from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import click
from dbt.cli.main import cli, p, requires
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.parser import manifest


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


def invoke_parse_customized(
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


@contextmanager
def disable_node_not_found_error() -> Iterator[None]:
    """A dbt parse command will raise an error if it cannot find all referenced nodes.
    This context manager disables the error as dbt cannot yet see the Ibis models
    as they are not yet compiled at this point and hence will raise an error.
    """
    original_func = manifest.invalid_target_fail_unless_test

    def _do_nothing(*args: Any, **kwargs: Any) -> None:
        pass

    try:
        manifest.invalid_target_fail_unless_test = _do_nothing
        yield
    finally:
        manifest.invalid_target_fail_unless_test = original_func
