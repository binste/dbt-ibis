from typing import NewType

import ibis.backends.base.sqlglot.datatypes as sqlglot_dt
import ibis.expr.datatypes as dt
import ibis.expr.types
from dbt.contracts.graph.manifest import Manifest
from ibis.formats import TypeMapper

IbisDialect = NewType("IbisDialect", str)
DBTAdapterType = NewType("DBTAdapterType", str)

DBTAdapterTypeToIbisDialect: dict[DBTAdapterType, IbisDialect] = {
    "postgres": "postgres",
    "redshift": "postgres",
    "snowflake": "snowflake",
    "trino": "trino",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "oracle": "oracle",
    "duckdb": "duckdb",
}

IbisDialectToTypeMapper: dict[IbisDialect, TypeMapper] = {
    "postgres": sqlglot_dt.PostgresType,
    "snowflake": sqlglot_dt.SnowflakeType,
    "trino": sqlglot_dt.TrinoType,
    "mysql": sqlglot_dt.MySQLType,
    "sqlite": sqlglot_dt.SQLiteType,
    "oracle": sqlglot_dt.OracleType,
    "duckdb": sqlglot_dt.DuckDBType,
}


def get_ibis_dialect(manifest: Manifest) -> IbisDialect:
    dbt_adapter_type: DBTAdapterType = manifest.metadata.adapter_type
    if dbt_adapter_type is None:
        raise ValueError("Could not determine dbt adapter type")
    elif dbt_adapter_type not in DBTAdapterTypeToIbisDialect:
        raise ValueError(
            f"DBT adapter type {dbt_adapter_type} is not supported by dbt-ibis."
        )
    return DBTAdapterTypeToIbisDialect[dbt_adapter_type]


def parse_db_dtype_to_ibis_dtype(
    db_dtype: str, ibis_dialect: IbisDialect
) -> dt.DataType:
    type_mapper = IbisDialectToTypeMapper[ibis_dialect]
    return type_mapper.from_string(db_dtype)


def ibis_expr_to_sql(
    ibis_expr: ibis.expr.types.Table, ibis_dialect: IbisDialect
) -> str:
    # Return type of .to_sql is SqlString which is a normal string with some
    # custom repr methods -> Convert it to a normal string here to make it easier
    # for mypy.
    return str(ibis.to_sql(ibis_expr, dialect=ibis_dialect))
