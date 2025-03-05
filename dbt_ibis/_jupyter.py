import inspect
import logging
import re
import textwrap
from pathlib import Path
from typing import Literal

from dbt_ibis._compile import IBIS_FILE_EXTENSION
from dbt_ibis._extract import get_expr_func

try:
    import nbformat as nbf
except ImportError as err:
    raise ImportError(
        "This functionality requires additional dependencies."
        + " Run 'pip install dbt-ibis[jupyter]' to install them."
    ) from err


logger = logging.getLogger(__name__)

_NB_HEADER_MODEL = "# Model"
_NB_HEADER_DEPENDS_ON = "# Depends on"


def convert_ibis_file_to_notebook(file_path: str | Path) -> None:
    logger.info(f"Converting {file_path} to notebook")

    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Split source code into multiple parts:
    # - Ibis expression itself (i.e. 'model' or 'test' function)
    # - Other code, i.e. code before and after the Ibis expression
    # - References to other Ibis expressions based on depends_on
    ibis_expr_func = get_expr_func(file_path)

    ibis_expr_func_code, func_start_line = inspect.getsourcelines(ibis_expr_func)
    other_code = _extract_other_code(
        file_lines=file_path.read_text().splitlines(),
        ibis_expr_func_code=ibis_expr_func_code,
        func_start_line=func_start_line,
    )

    expr_references: list[str] = [
        "# TODO: Create a connected Ibis backend",
        "con = ...",
        "",
    ]
    for reference_name, reference in zip(
        inspect.signature(ibis_expr_func).parameters,
        ibis_expr_func.depends_on,  # type: ignore[attr-defined]
    ):
        reference_repr = repr(reference).replace("'", '"')
        expr_references.append(
            f"{reference_name} = {reference_repr}.to_ibis_table(con=con)"
        )

    cleaned_function_code = _remove_decorator(ibis_expr_func_code)
    function_content = _unpack_function_conent(cleaned_function_code)

    # Create and save notebook
    nb = _create_notebook(
        ibis_expr_code=function_content,
        other_code=other_code,
        depends_on_code=expr_references,
    )
    notebook_path = file_path.with_suffix(".ipynb")
    nbf.write(nb, notebook_path)
    logger.info(f"Notebook saved as {notebook_path}")


def _extract_other_code(
    file_lines: list[str], ibis_expr_func_code: list[str], func_start_line: int
) -> list[str]:
    end_line = func_start_line + len(ibis_expr_func_code) - 1

    other_code = file_lines[: func_start_line - 1]
    code_after = file_lines[end_line:]
    if code_after:
        # Add two empty lines in between so that its better formatted in the notebook
        other_code = [*other_code, "", "", *code_after]
    return other_code


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
    other_code: list[str],
    depends_on_code: list[str],
) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    # Define all code before the Ibis expression as the expression
    # might reference functionality defined there. In the ibis file,
    # this was not an issue as that was all in a function. Now, we unpack
    # the function -> references need to be valid.
    nb["cells"] = [
        nbf.v4.new_code_cell(source="\n".join(other_code).strip()),
        nbf.v4.new_markdown_cell(_NB_HEADER_DEPENDS_ON),
        nbf.v4.new_code_cell("\n".join(depends_on_code)),
        nbf.v4.new_markdown_cell(_NB_HEADER_MODEL),
        nbf.v4.new_code_cell(ibis_expr_code),
    ]

    return nb


def convert_notebook_to_ibis_file(file_path: str | Path) -> None:
    logger.info(f"Converting {file_path} to an Ibis file")
    if isinstance(file_path, str):
        file_path = Path(file_path)

    nb = nbf.read(file_path, as_version=4)
    other_code, depends_on, ibis_expr_code = _split_notebook_content(nb)

    references = _extract_references(depends_on)

    depends_on_decorator = f"@depends_on({', '.join([ref for _, ref in references])})"
    function_definition = f"def model({', '.join([name for name, _ in references])}):"

    # Return value of last line of ibis expression. Don't check for now
    # if this is valid Python code -> Easy for users to fix themselves if its
    # wrong.
    def remove_empty_strings_from_end(lst: list[str]) -> list[str]:
        while lst and lst[-1] == "":
            lst.pop()
        return lst

    ibis_expr_code = remove_empty_strings_from_end(ibis_expr_code)
    ibis_expr_code[-1] = f"return {ibis_expr_code[-1]}"
    ibis_expr_function_code = "\n".join(
        [
            depends_on_decorator,
            function_definition,
            textwrap.indent("\n".join(ibis_expr_code), prefix=" " * 4),
        ]
    )

    ibis_file_path = file_path.with_suffix(f".{IBIS_FILE_EXTENSION}")
    ibis_file_content_lines = [*other_code, ibis_expr_function_code]
    if ibis_file_content_lines[-1] != "":
        # Add an empty line at the end as that conforms to most style guides
        ibis_file_content_lines.append("")
    ibis_file_content = "\n".join(ibis_file_content_lines)
    ibis_file_path.write_text(ibis_file_content)
    logger.info(f"Ibis file saved as {ibis_file_path}")


def _split_notebook_content(  # noqa: C901
    nb: nbf.NotebookNode,
) -> tuple[list[str], list[str], list[str]]:
    other_code: list[str] = []
    depends_on: list[str] = []
    ibis_expr_code: list[str] = []

    current_section: Literal["depends_on", "model"] | None = None

    def process_cell_source(source: str) -> list[str]:
        return source.split("\n")

    for cell in nb.cells:
        if cell.cell_type == "code":
            if current_section is None:
                other_code.extend(process_cell_source(cell.source))
            elif current_section == "depends_on":
                depends_on.extend(process_cell_source(cell.source))
            elif current_section == "model":
                ibis_expr_code.extend(process_cell_source(cell.source))
        elif cell.cell_type == "markdown":
            if not cell.source.startswith("#"):
                continue
            elif cell.source.startswith("# Depends on"):
                current_section = "depends_on"
            elif cell.source.startswith("# Model"):
                current_section = "model"
            elif cell.source.startswith("##"):
                # Count this as a subheading of the current section -> Don't change it
                continue
            else:
                # In this case, it's a top-level heading -> Reset the current section
                # and just count the following code cells as other code
                current_section = None
        else:
            # Ignore other cell types such as raw cells
            continue
    return other_code, depends_on, ibis_expr_code


def _extract_references(depends_on: list[str]) -> list[tuple[str, str]]:
    references = []
    for line in depends_on:
        variable_name = re.search(r"^(\w+)\s*=", line)
        reference = re.search(r"=\s*((ref|source).*?)\.to_ibis_table", line)
        if variable_name is not None and reference is not None:
            references.append((variable_name.group(1), reference.group(1)))
    return references
