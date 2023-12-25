import inspect
import logging
import textwrap
from pathlib import Path

from dbt_ibis._extract import get_expr_func

try:
    import nbformat as nbf
except ImportError as err:
    raise ImportError(
        "This functionality requires additional dependencies."
        + " Run 'pip install dbt-ibis[jupyter]' to install them."
    ) from err


logger = logging.getLogger(__name__)


def convert_ibis_file_to_notebook(file_path: str | Path) -> None:
    logger.info(f"Converting {file_path} to notebook")
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Split source code into multiple parts:
    # - Code before the Ibis expression
    # - Code after the Ibis expression
    # - Ibis expression itself (i.e. 'model' or 'test' function)
    # - References to other Ibis expressions based on depends_on
    ibis_expr_func = get_expr_func(file_path)

    ibis_expr_function_source_code, start_line = inspect.getsourcelines(ibis_expr_func)
    end_line = start_line + len(ibis_expr_function_source_code) - 1

    source_code = file_path.read_text().splitlines()

    code_before_ibis_expr = source_code[: start_line - 1]
    code_after_ibis_expr = source_code[end_line:]

    expr_references = []
    for reference_name, reference in zip(
        inspect.signature(ibis_expr_func).parameters,
        ibis_expr_func.depends_on,  # type: ignore[attr-defined]
    ):
        reference_repr = repr(reference).replace("'", '"')
        expr_references.append(
            f"{reference_name} = {reference_repr}.to_ibis_table(con=con)"
        )

    cleaned_function_code = _remove_decorator(ibis_expr_function_source_code)
    function_content = _unpack_function_conent(cleaned_function_code)

    # Create and save notebook
    nb = _create_notebook(
        ibis_expr_code=function_content,
        code_before_ibis_expr=code_before_ibis_expr,
        code_after_ibis_expr=code_after_ibis_expr,
        depends_on_code=expr_references,
    )
    notebook_path = file_path.with_suffix(".ipynb")
    nbf.write(nb, notebook_path)
    logger.info(f"Notebook saved as {notebook_path}")


def _remove_decorator(ibis_expr_function_source_code: list[str]) -> list[str]:
    cleaned_function_code = []
    def_started = False
    for line in ibis_expr_function_source_code:
        if not def_started and not line.strip().startswith("def "):
            continue
        else:
            def_started = True
            cleaned_function_code.append(line)

    return cleaned_function_code


def _unpack_function_conent(cleaned_function_code: list[str]) -> str:
    function_content = []
    indent_found = False
    for line in cleaned_function_code:
        if not indent_found and line.startswith((" ", "\t")):
            indent_found = True
        if indent_found:
            function_content.append(line)

    function_content = [line.replace("return ", "") for line in function_content]
    # New lines are already part of the source code here so no need to add them
    function_content = textwrap.dedent("".join(function_content))
    return function_content


def _create_notebook(
    *,
    ibis_expr_code: str,
    code_before_ibis_expr: list[str],
    code_after_ibis_expr: list[str],
    depends_on_code: list[str],
) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    # Define all code before the Ibis expression as the expression
    # might reference functionality defined there. In the ibis file,
    # this was not an issue as that was all in a function. Now, we unpack
    # the function -> references need to be valid.
    nb["cells"] = [
        nbf.v4.new_code_cell(
            source="\n".join(
                [*code_before_ibis_expr, "", "", *code_after_ibis_expr]
            ).strip()
        ),
        nbf.v4.new_markdown_cell("# Depends on"),
        nbf.v4.new_code_cell("\n".join(depends_on_code)),
        nbf.v4.new_markdown_cell("# Model"),
        nbf.v4.new_code_cell(ibis_expr_code),
    ]

    return nb
