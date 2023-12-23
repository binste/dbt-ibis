import graphlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Literal, Optional, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import (
    ColumnInfo,
    ModelNode,
    SeedNode,
    SnapshotNode,
    SourceDefinition,
)

from dbt_ibis import _dialects
from dbt_ibis._references import (
    _REF_IDENTIFIER_PREFIX,
    _REF_IDENTIFIER_SUFFIX,
    _SOURCE_IDENTIFIER_PREFIX,
    _SOURCE_IDENTIFIER_SEPARATOR,
    _SOURCE_IDENTIFIER_SUFFIX,
    _Reference,
    ref,
    source,
)

IBIS_FILE_EXTENSION: Final = "ibis"
IBIS_SQL_FOLDER_NAME: Final = "__ibis_sql"


_RefLookup = dict[str, Union[ModelNode, SeedNode, SnapshotNode]]
_SourcesLookup = dict[str, dict[str, SourceDefinition]]
_LetterCase = Literal["lower", "upper"]


@dataclass
class IbisExprInfo:
    ibis_path: Path
    depends_on: list[_Reference]
    func: Callable[..., ir.Table]

    @property
    def name(self) -> str:
        return self.ibis_path.stem

    @property
    def sql_path(self) -> Path:
        return self.ibis_path.parent / IBIS_SQL_FOLDER_NAME / f"{self.name}.sql"


def compile_ibis_expressions_to_sql(
    all_ibis_expr_infos: list[IbisExprInfo],
    manifest: Manifest,
    runtime_config: RuntimeConfig,
    ibis_dialect: _dialects.IbisDialect,
) -> None:
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
        references: list[ir.Table] = []
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
        ibis_expr = _set_letter_case_on_ibis_expression(ibis_expr, letter_case_in_db)

        ibis_expr_schemas[ibis_expr_info.name] = ibis_expr.schema()

        # Convert to SQL and write to file
        dbt_sql = _to_dbt_sql(ibis_expr, ibis_dialect=ibis_dialect)
        ibis_expr_info.sql_path.parent.mkdir(parents=False, exist_ok=True)
        ibis_expr_info.sql_path.write_text(dbt_sql)


def _set_letter_case_on_ibis_expression(
    ibis_expr: ir.Table, letter_case: Optional[_LetterCase]
) -> ir.Table:
    if letter_case == "upper":
        ibis_expr = ibis_expr.rename("ALL_CAPS")
    elif letter_case == "lower":
        ibis_expr = ibis_expr.rename("snake_case")
    return ibis_expr


def _sort_ibis_exprs_by_dependencies(
    ibis_exprs: list[IbisExprInfo],
) -> list[IbisExprInfo]:
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


def _to_dbt_sql(ibis_expr: ir.Table, ibis_dialect: _dialects.IbisDialect) -> str:
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


def _validate_letter_case_var(variable_name: str, value: Any) -> Optional[_LetterCase]:
    if value is not None and value not in ["lower", "upper"]:
        raise ValueError(
            f"The {variable_name} variable needs to be set to"
            + f" either 'lower' or 'upper' but currently has a value of '{value}'."
            + " If you want the default behaviour of Ibis, you can omit this variable."
        )
    return value
