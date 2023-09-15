import ibis
import ibis.expr.datatypes as dt
import pytest

from dbt_ibis._dialects import (
    DBTAdapterTypeToIbisDialect,
    IbisDialect,
    ibis_expr_to_sql,
    parse_db_dtype_to_ibis_dtype,
)

SUPPORTED_IBIS_DIALECTS = list(DBTAdapterTypeToIbisDialect.values())


@pytest.mark.parametrize(
    ("db_dtype", "ibis_dialect", "ibis_dtype"),
    # Check that we have a type mapper for all Ibis dialects that should be supported
    [("varchar", ibis_dialect, dt.String()) for ibis_dialect in SUPPORTED_IBIS_DIALECTS]
    +
    # Some special cases to check
    [
        ("bigint", "duckdb", dt.Int64()),
        ("BIGINT", "duckdb", dt.Int64()),
        ("numeric", "duckdb", dt.Decimal(18, 3)),
        ("numeric", "snowflake", dt.Int64()),
        ("numeric(30, 10)", "snowflake", dt.Decimal(30, 10)),
    ],
)
def test_parse_db_dtype_to_ibis_dtype(
    db_dtype: str, ibis_dialect: IbisDialect, ibis_dtype: dt.DataType
):
    # We don't need to test all possible datatypes here. This already
    # happens in the Ibis test suite. Just want to make sure that the parse function
    # works as expected.
    assert (
        parse_db_dtype_to_ibis_dtype(db_dtype, ibis_dialect=ibis_dialect) == ibis_dtype
    )


def test_ibis_expr_to_sql():
    schema = ibis.schema({"col1": "int"})
    table = ibis.table(schema=schema, name="test_table")
    expression = table["col1"].sum()

    # We do not check the actual SQL string here. This is already tested in the Ibis
    # and/or SQLGlot codebase. We just want to make sure that the function works as
    # expected.
    # We can only test it for duckdb as the other backends are not installed
    # in development mode.
    sql = ibis_expr_to_sql(expression, ibis_dialect="duckdb")
    assert isinstance(sql, str)
