from typing import NewType

import ibis
import ibis.backends.sql.datatypes as sql_dt
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
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
    DBTAdapterType("bigquery"): IbisDialect("bigquery"),
    DBTAdapterType("risingwave"): IbisDialect("risingwave"),
    DBTAdapterType("databricks"): IbisDialect("databricks"),
}

IbisDialectToTypeMapper: dict[IbisDialect, type[TypeMapper]] = {
    IbisDialect("postgres"): sql_dt.PostgresType,
    IbisDialect("snowflake"): sql_dt.SnowflakeType,
    IbisDialect("trino"): sql_dt.TrinoType,
    IbisDialect("mysql"): sql_dt.MySQLType,
    IbisDialect("sqlite"): sql_dt.SQLiteType,
    IbisDialect("oracle"): sql_dt.OracleType,
    IbisDialect("duckdb"): sql_dt.DuckDBType,
    IbisDialect("bigquery"): sql_dt.BigQueryType,
    IbisDialect("risingwave"): sql_dt.RisingWaveType,
    IbisDialect("databricks"): sql_dt.DatabricksType,
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


def ibis_expr_to_sql(ibis_expr: ir.Table, ibis_dialect: IbisDialect) -> str:
    # Return type of .to_sql is SqlString which is a normal string with some
    # custom repr methods -> Convert it to a normal string here to make it easier
    # for mypy.
    return str(ibis.to_sql(ibis_expr, dialect=ibis_dialect))
