from abc import ABC, abstractproperty
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Final, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

_REF_IDENTIFIER_PREFIX: Final = "__ibd_ref__"
_REF_IDENTIFIER_SUFFIX: Final = "__rid__"
_SOURCE_IDENTIFIER_PREFIX: Final = "__ibd_source__"
_SOURCE_IDENTIFIER_SUFFIX: Final = "__sid__"
_SOURCE_IDENTIFIER_SEPARATOR: Final = "__ibd_sep__"


class _Reference(ABC):
    @abstractproperty
    def _ibis_table_name(self) -> str:
        pass

    @abstractproperty
    def table_name(self) -> str:
        pass

    def to_ibis_table(
        self,
        schema_definition: Union[ibis.Schema, dict[str, dt.DataType]] | None = None,
        con: ibis.BaseBackend | None = None,
        schema_name: str | None = None,
        database_name: str | None = None,
    ) -> ir.Table:
        if schema_definition is None and con is None:
            raise ValueError(
                "Either schema_definition or an Ibis backend need to be provided"
            )
        if schema_definition is not None and con is not None:
            raise ValueError(
                "Either schema_definition or an Ibis backend need to be provided"
                + " but not both"
            )

        if schema_definition is not None:
            return ibis.table(
                schema_definition,
                name=self._ibis_table_name,
            )
        else:
            return con.table(
                self.table_name,
                schema=schema_name,
                database=database_name,
            )


@dataclass
class ref(_Reference):
    """A reference to a dbt model."""

    name: str

    @property
    def _ibis_table_name(self) -> str:
        return _REF_IDENTIFIER_PREFIX + self.name + _REF_IDENTIFIER_SUFFIX

    @property
    def table_name(self) -> str:
        return self.name


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
        func: Callable[..., ir.Table],
    ) -> Callable[..., ir.Table]:
        @wraps(func)
        def wrapper(*args: _Reference, **kwargs: _Reference) -> ir.Table:
            return func(*args, **kwargs)

        wrapper.depends_on = references  # type: ignore[attr-defined]
        return wrapper

    return decorator
