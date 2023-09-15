import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types.relations as ir
import pandas as pd
import pytest
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import (
    ColumnInfo,
    GenericTestNode,
    ModelNode,
    SeedNode,
    SourceDefinition,
)
from dbt.node_types import NodeType
from dbt.parser import manifest

from dbt_ibis import (
    _IBIS_SQL_FOLDER_NAME,
    _columns_to_ibis_schema,
    _disable_node_not_found_error,
    _extract_ref_and_source_infos,
    _get_ibis_models,
    _get_model_func,
    _get_parse_arguments,
    _get_schema_for_ref,
    _get_schema_for_source,
    _IbisModel,
    _parse_db_dtype_to_ibis_dtype,
    _sort_ibis_models_by_dependencies,
    _to_dbt_sql,
    depends_on,
    ref,
    source,
)


@pytest.fixture()
def stg_orders_model_node():
    return ModelNode(
        name="stg_orders",
        # These values do not matter for our purposes
        database="",
        schema="",
        resource_type=NodeType.Model,
        package_name="",
        path="",
        original_file_path="",
        unique_id="",
        fqn=[""],
        alias="",
        checksum="",  # type: ignore[arg-type]
    )


@pytest.fixture()
def raw_payments_seed_node():
    return SeedNode(
        name="raw_payments",
        # These values do not matter for our purposes
        resource_type=NodeType.Seed,
        database="",
        schema="",
        package_name="",
        path="",
        original_file_path="",
        unique_id="",
        fqn=[""],
        alias="",
        checksum="",  # type: ignore[arg-type]
    )


@pytest.fixture()
def orders_source_definition():
    return SourceDefinition(
        name="orders",
        source_name="source1",
        columns={
            "col1": ColumnInfo(name="col1", data_type="bigint"),
            "col2": ColumnInfo(name="col2", data_type="varchar"),
        },
        # These attributes are required but do not matter for our purposes
        database="",
        schema="",
        resource_type=NodeType.Source,
        package_name="",
        path="",
        original_file_path="",
        unique_id="",
        fqn=[""],
        source_description="",
        loader="",
        identifier="",
    )


def get_compiled_sql_files(project_dir: Path) -> list[Path]:
    return list(project_dir.glob(f"models/**/{_IBIS_SQL_FOLDER_NAME}/*.sql"))


@pytest.fixture()
def project_dir_and_database_file(monkeypatch) -> tuple[Path, Path]:
    # Remove files which might exist from
    # a previous test run which can happen if the tests are run locally, i.e.
    # not in the GitHub pipeline.
    project_dir = Path.cwd() / "demo_project" / "jaffle_shop"
    # Change working directory so that all dbt-ibis commands below are executed
    # in the project directory
    monkeypatch.chdir(project_dir)

    for file in get_compiled_sql_files(project_dir):
        file.unlink()

    target_folder = project_dir / "target"
    if target_folder.exists():
        shutil.rmtree(target_folder)

    database_file = project_dir / "db.duckdb"
    if database_file.exists():
        database_file.unlink()
    return project_dir, database_file


def test_ref():
    model_name = "stg_orders"
    stg_orders = ref(model_name)

    assert stg_orders.name == model_name

    schema = ibis.schema({"col1": "int"})
    ibis_table = stg_orders.to_ibis(schema=schema)

    assert isinstance(ibis_table, ir.Table)
    assert ibis_table.schema() == schema
    assert ibis_table.get_name() == f"__ibd_ref__{model_name}__rid__"


def test_source():
    source_name = "source1"
    table_name = "orders"
    orders = source(source_name, table_name)

    assert orders.source_name == source_name
    assert orders.table_name == table_name

    schema = ibis.schema({"col1": "int"})
    ibis_table = orders.to_ibis(schema=schema)

    assert isinstance(ibis_table, ir.Table)
    assert ibis_table.schema() == schema
    assert (
        ibis_table.get_name()
        == f"__ibd_source__{source_name}__ibd_sep__{table_name}__sid__"
    )


def test_depends_on():
    references = [source("source1", "orders"), ref("stg_customers")]

    @depends_on(*references)
    def func(orders, stg_customers, something_else):
        return orders, stg_customers, something_else

    assert func.depends_on == tuple(references)
    # Make sure that function can be executed and arguments are passed through
    assert func(*references, something_else=1) == (*references, 1)

    with pytest.raises(
        ValueError,
        match="All arguments to depends_on need to be either an instance of"
        + " dbt_ibis.ref or dbt_ibis.source",
    ):

        @depends_on("stg_orders", ref("stg_customers"))  # type: ignore[arg-type]
        def another_func():
            pass


def test_ibis_model():
    references = [source("source1", "orders"), ref("stg_customers")]
    model = _IbisModel(
        ibis_path=Path("path/to/some_model.ibis"),
        depends_on=references,
        model_func=lambda: None,  # type: ignore  # noqa: PGH003
    )

    assert model.name == "some_model"
    assert model.sql_path == Path(f"path/to/{_IBIS_SQL_FOLDER_NAME}/some_model.sql")


def test_get_ibis_models():
    ibis_models = _get_ibis_models(
        Path.cwd() / "tests", model_paths=["mock_model_folder_1", "mock_model_folder_2"]
    )

    assert len(ibis_models) == 3
    assert all(isinstance(model, _IbisModel) for model in ibis_models)
    assert all(callable(model.model_func) for model in ibis_models)
    assert all(
        isinstance(model.depends_on, tuple) and len(model.depends_on) == 1
        for model in ibis_models
    )
    ibis_models = sorted(ibis_models, key=lambda model: model.name)
    assert [model.name for model in ibis_models] == ["model_1", "model_2", "model_3"]
    assert [model.ibis_path for model in ibis_models] == [
        Path("tests/mock_model_folder_1/model_1.ibis").absolute(),
        Path("tests/mock_model_folder_1/subfolder/model_2.ibis").absolute(),
        Path("tests/mock_model_folder_2/model_3.ibis").absolute(),
    ]


def test_get_model_func():
    model_func = _get_model_func(Path("tests/mock_model_folder_1/model_1.ibis"))

    assert callable(model_func)
    assert model_func.__name__ == "model"
    assert model_func.depends_on == (ref("stg_orders"),)  # type: ignore[attr-defined]


def test_sort_ibis_models_by_dependencies():
    ibis_models = [
        _IbisModel(
            ibis_path=Path("path/to/another_model.ibis"),
            depends_on=[ref("some_model")],
            model_func=lambda: None,  # type: ignore  # noqa: PGH003
        ),
        _IbisModel(
            ibis_path=Path("path/to/some_model.ibis"),
            # Using same name for source table name as for the
            # other model above to make sure that sources are ignored
            # when building a dependency graph and this function
            # does not suddenly treat source and ref the same.
            depends_on=[source("source1", "another_model")],
            model_func=lambda: None,  # type: ignore  # noqa: PGH003
        ),
    ]

    sorted_ibis_models = _sort_ibis_models_by_dependencies(ibis_models)

    assert sorted_ibis_models == ibis_models[::-1]


def test_extract_model_and_source_infos(
    orders_source_definition, stg_orders_model_node, raw_payments_seed_node
):
    dbt_manifest = Manifest(
        nodes={
            "stg_orders": stg_orders_model_node,
            "some_test": GenericTestNode(
                test_metadata="",  # type: ignore[arg-type]
                database="",
                schema="",
                name="",
                resource_type=NodeType.Test,
                package_name="",
                path="",
                original_file_path="",
                unique_id="",
                fqn=[""],
                alias="",
                checksum="",  # type: ignore[arg-type]
            ),
            "raw_payments": raw_payments_seed_node,
        },
        sources={"orders": orders_source_definition},
    )

    ref_lookup, sources_lookup = _extract_ref_and_source_infos(dbt_manifest)

    assert ref_lookup == {
        "stg_orders": stg_orders_model_node,
        "raw_payments": raw_payments_seed_node,
    }
    assert sources_lookup == {"source1": {"orders": orders_source_definition}}


def test_get_schema_for_source(orders_source_definition):
    orders = source("source1", "orders")
    sources_lookup = {"source1": {"orders": orders_source_definition}}
    schema = _get_schema_for_source(orders, sources_lookup)
    assert schema == ibis.schema({"col1": dt.Int64(), "col2": dt.String()})


def test_get_schema_for_ref(stg_orders_model_node, raw_payments_seed_node):
    stg_orders = ref("stg_orders")
    models_lookup = {"stg_orders": stg_orders_model_node}
    # As the ModelNode does not have columns defined, it should raise an error
    # as we also did not pass in a schema in ibis_model_schemas
    with pytest.raises(
        ValueError, match="Could not determine schema for model 'stg_orders'"
    ):
        _get_schema_for_ref(stg_orders, models_lookup, ibis_model_schemas={})

    schema_from_ibis_model = ibis.schema({"col1": dt.String()})
    ibis_model_schemas = {"stg_orders": schema_from_ibis_model}
    assert (
        _get_schema_for_ref(
            stg_orders, models_lookup, ibis_model_schemas=ibis_model_schemas
        )
        == schema_from_ibis_model
    )

    # Now make sure that columns from model_lookup have priority over
    # passed in Ibis model schemas
    models_lookup["stg_orders"].columns = {
        "col1": ColumnInfo(name="col1", data_type="bigint"),
        "col2": ColumnInfo(name="col2", data_type="varchar"),
    }

    schema = _get_schema_for_ref(
        stg_orders, models_lookup, ibis_model_schemas=ibis_model_schemas
    )
    assert schema == ibis.schema({"col1": dt.Int64(), "col2": dt.String()})

    # Test if it works for a seed
    with pytest.raises(
        ValueError, match="Could not determine schema for model 'raw_payments'"
    ):
        _get_schema_for_ref(
            ref("raw_payments"),
            ref_infos={"raw_payments": raw_payments_seed_node},
            ibis_model_schemas={},
        )

    raw_payments_seed_node.config.column_types = {
        "id": "integer",
        "order_id": "integer",
        "payment_method": "varchar",
        "amount": "decimal(30, 10)",
    }
    schema = _get_schema_for_ref(
        ref("raw_payments"),
        ref_infos={"raw_payments": raw_payments_seed_node},
        ibis_model_schemas={},
    )
    assert schema == ibis.schema(
        {
            "id": dt.Int32(),
            "order_id": dt.Int32(),
            "payment_method": dt.String(),
            "amount": dt.Decimal(30, 10),
        }
    )


def test_columns_to_ibis_schema():
    columns = {
        "col1": ColumnInfo(name="col1", data_type="bigint"),
        "col2": ColumnInfo(name="col2", data_type="varchar"),
    }

    schema = _columns_to_ibis_schema(columns)
    assert schema == ibis.schema({"col1": dt.Int64(), "col2": dt.String()})

    columns["col3"] = ColumnInfo(name="col3")
    with pytest.raises(
        ValueError, match="Could not determine data type for column 'col3'"
    ):
        _columns_to_ibis_schema(columns)


@pytest.mark.parametrize(
    ("db_dtype", "db_dialect", "ibis_dtype"),
    [
        ("bigint", "duckdb", dt.Int64()),
        ("BIGINT", "duckdb", dt.Int64()),
        ("varchar", "duckdb", dt.String()),
    ],
)
def test_parse_db_dtype_to_ibis_dtype(
    db_dtype: str, db_dialect: str, ibis_dtype: dt.DataType  # noqa: ARG001
):
    # We don't need to test all possible dialects and datatypes here. This already
    # happens in the Ibis test suite.

    # Function does not yet support different dialects. Waiting for Ibis 7
    assert _parse_db_dtype_to_ibis_dtype(db_dtype) == ibis_dtype


def test_to_dbt_sql():
    orders = source("source1", "orders")
    orders_table = orders.to_ibis(
        schema=ibis.schema({"order_id": "int", "customer_id": "int"})
    )

    stg_customers = ref("stg_customers")
    stg_customers_table = stg_customers.to_ibis(
        schema=ibis.schema({"customer_id": "int"})
    )

    model_expr = orders_table.join(
        stg_customers_table,
        orders_table["customer_id"] == stg_customers_table["customer_id"],
        how="left",
    )
    dbt_sql = _to_dbt_sql(model_expr)

    assert (
        dbt_sql
        == """\
SELECT
  t0.order_id,
  t0.customer_id,
  t1.customer_id AS customer_id_right
FROM {{ source('source1', 'orders') }} AS t0
LEFT OUTER JOIN {{ ref('stg_customers') }} AS t1
  ON t0.customer_id = t1.customer_id"""
    )


def test_disable_node_not_found_error():
    assert (
        manifest.invalid_target_fail_unless_test.__name__
        == "invalid_target_fail_unless_test"
    )

    with _disable_node_not_found_error():
        assert manifest.invalid_target_fail_unless_test.__name__ == "_do_nothing"
        manifest.invalid_target_fail_unless_test()  # type: ignore[call-arg]

    assert (
        manifest.invalid_target_fail_unless_test.__name__
        == "invalid_target_fail_unless_test"
    )


def test_get_parse_arguments(mocker):
    mocker.patch.object(
        sys,
        "argv",
        [
            "dbt",
            "--global-flag",
            "run",
            "--project-dir",
            "some_folder",
            "--select",
            "stg_orders+",
        ],
    )

    args = _get_parse_arguments()

    assert args == [
        "--quiet",
        "parse_customized",
        "--project-dir",
        "some_folder",
        "--select",
        "stg_orders+",
    ]

    mocker.patch.object(
        sys,
        "argv",
        [
            "dbt",
            "ls",
        ],
    )

    args = _get_parse_arguments()

    assert args == [
        "--quiet",
        "parse_customized",
    ]


def execute_command(cmd: list[str]) -> None:
    process = subprocess.run(
        cmd,  # noqa: S603
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.check_returncode()


def validate_compiled_sql_files(project_dir: Path) -> list[Path]:
    compiled_sql_files = get_compiled_sql_files(project_dir)
    assert len(compiled_sql_files) == 5

    # Test content of some of the compiled SQL files
    stg_stores = next(p for p in compiled_sql_files if p.stem == "stg_stores")
    assert (
        stg_stores.read_text()
        == """\
SELECT
  CAST(t0.store_id AS BIGINT) AS store_id,
  t0.store_name,
  t0.country
FROM {{ source('sources_db', 'stores') }} AS t0"""
    )

    usa_stores = stg_stores = next(
        p for p in compiled_sql_files if p.stem == "usa_stores"
    )
    assert (
        usa_stores.read_text()
        == """\
SELECT
  t0.store_id,
  t0.store_name,
  t0.country
FROM {{ ref('stg_stores') }} AS t0
WHERE
  t0.country = 'USA'"""
    )
    return compiled_sql_files


def test_precompile_command(project_dir_and_database_file: tuple[Path, Path]):
    project_dir, database_file = project_dir_and_database_file
    execute_command(["dbt-ibis", "precompile"])

    assert (
        not database_file.exists()
    ), "Database was created although precompile command should not create it."
    validate_compiled_sql_files(project_dir)


def test_end_to_end(project_dir_and_database_file: tuple[Path, Path]):
    project_dir, database_file = project_dir_and_database_file

    execute_command(["dbt-ibis", "seed"])
    # Check that all Ibis models were compiled and seed
    # was executed
    compiled_sql_files = validate_compiled_sql_files(project_dir)

    # Check that seeds command was executed and the tables were created
    # but not already any models
    def get_db_con():
        return duckdb.connect(str(database_file), read_only=True)

    def get_tables() -> list[str]:
        db_con = duckdb.connect(str(database_file), read_only=True)
        table_names = [
            r[0]
            for r in db_con.execute(
                """select table_name
                from information_schema.tables
                order by table_name"""
            ).fetchall()
        ]
        db_con.close()
        return table_names

    seed_tables = ["raw_orders", "raw_customers", "raw_payments"]
    assert get_tables() == sorted(seed_tables)

    execute_command(["dbt-ibis", "snapshot"])

    snapshot_tables = ["orders_snapshot"]
    assert get_tables() == sorted([*seed_tables, *snapshot_tables])

    # Only run for a few models at first to make sure that --select
    # is passed through to dbt run
    execute_command(["dbt-ibis", "run", "--select", "stg_orders"])
    assert get_tables() == sorted([*seed_tables, *snapshot_tables, "stg_orders"])

    execute_command(
        [
            "dbt-ibis",
            "run",
        ]
    )
    assert get_tables() == sorted(
        [
            *seed_tables,
            *snapshot_tables,
            "stg_orders",
            "stg_customers",
            "stg_payments",
            "stg_stores",
            "customers_with_multiple_orders",
            "customers",
            "orders",
            "usa_stores",
        ]
    )

    execute_command(
        [
            "dbt-ibis",
            "test",
        ]
    )

    # Check for one Ibis model that the data is what we expect
    assert any(
        p.stem == "usa_stores" for p in compiled_sql_files
    ), "usa_stores is no longer an Ibis model. Adapt test below to use one."
    db_con = get_db_con()
    usa_stores_df = db_con.execute(
        "select * from usa_stores order by store_id"
    ).fetchdf()

    assert usa_stores_df.equals(
        pd.DataFrame(
            [[1, "San Francisco", "USA"], [2, "New York", "USA"]],
            columns=["store_id", "store_name", "country"],
        )
    )
