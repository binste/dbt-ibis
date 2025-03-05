from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Callable, Union

from ibis import ir


def glob_in_paths(
    project_root: Union[str, Path], paths: list[str], pattern: str
) -> list[Path]:
    if isinstance(project_root, str):
        project_root = Path(project_root)

    matches: list[Path] = []
    for m_path in paths:
        matches.extend(list((project_root / m_path).glob(pattern)))
    return matches


def get_expr_func(file: Path) -> Callable[..., ir.Table]:
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
