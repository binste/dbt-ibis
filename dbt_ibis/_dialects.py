from typing import NewType

import ibis.backends.base.sqlglot.datatypes as sqlglot_dt
import ibis.expr.datatypes as dt
import ibis.expr.types
from dbt.contracts.graph.manifest import Manifest
from ibis.formats import TypeMapper

# Use NewType to make sure that we don't accidentally mix these up, i.e.
# pass a DBTAdapterType to a function that expects an IbisDialect or vice versa.
IbisDialect = NewType("IbisDialect", str)
DBTAdapterType = NewType("DBTAdapterType", str)

DBTAdapterTypeToIbisDialect: dict[DBTAdapterType, IbisDialect] = {
    DBTAdapterType("postgres"): IbisDialect("postgres"),
    DBTAdapterType("redshift"): IbisDialect("postgres"),
    DBTAdapterType("snowflake"): IbisDialect("snowflake"),
    DBTAdapterType("trino"): IbisDialect("trino"),
    DBTAdapterType("mysql"): IbisDialect("mysql"),
    DBTAdapterType("sqlite"): IbisDialect("sqlite"),
    DBTAdapterType("oracle"): IbisDialect("oracle"),
    DBTAdapterType("duckdb"): IbisDialect("duckdb"),
}

IbisDialectToTypeMapper: dict[IbisDialect, type[TypeMapper]] = {
    IbisDialect("postgres"): sqlglot_dt.PostgresType,
    IbisDialect("snowflake"): sqlglot_dt.SnowflakeType,
    IbisDialect("trino"): sqlglot_dt.TrinoType,
    IbisDialect("mysql"): sqlglot_dt.MySQLType,
    IbisDialect("sqlite"): sqlglot_dt.SQLiteType,
    IbisDialect("oracle"): sqlglot_dt.OracleType,
    IbisDialect("duckdb"): sqlglot_dt.DuckDBType,
}


def get_ibis_dialect(manifest: Manifest) -> IbisDialect:
    dbt_adapter_type = manifest.metadata.adapter_type
    if dbt_adapter_type is None:
        raise ValueError("Could not determine dbt adapter type")
    elif dbt_adapter_type not in DBTAdapterTypeToIbisDialect:
        raise ValueError(
            f"DBT adapter type {dbt_adapter_type} is not supported by dbt-ibis."
        )
    return DBTAdapterTypeToIbisDialect[DBTAdapterType(dbt_adapter_type)]


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
