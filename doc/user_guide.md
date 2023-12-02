# User guide
## Basic example
You can write your Ibis model in files with the extension `.ibis`. Each `.ibis` file needs to correspond to one model which is defined as a `model` function returning an Ibis table expression:

`stg_stores.ibis`:
```python
from dbt_ibis import depends_on, source


@depends_on(source("sources_db", "stores"))
def model(stores):
    return stores.mutate(store_id=stores["store_id"].cast("int"))
```

You can now reference the `stg_stores` model in either a normal SQL model using `{{ ref('stg_stores') }}` or in another Ibis model:

`usa_stores.ibis`:
```python
from dbt_ibis import depends_on, ref


@depends_on(ref("stg_stores"))
def model(stores):
    return stores.filter(stores["country"] == "USA")
```

See [this version of the dbt jaffle shop project](https://github.com/binste/dbt-ibis/tree/main/demo_project/jaffle_shop) for a full example.

## Column data types
Whenever your Ibis model references either a source, a seed, a snapshot, or a SQL model, you'll need to specify the column data types of the referred dbt object in the corresponding `yaml` config. If you reference another Ibis model, this is not necessary. This is simpler than it sounds, don't worry! In the `stg_stores` examples above, you are referencing the `stores` source and hence you'll need to specify the data types for the `stores` source table:

```yaml
sources:
  - name: sources_db
    database: sources_db
    schema: main
    tables:
      - name: stores
        columns:
          - name: store_id
            data_type: varchar
          - name: store_name
            data_type: varchar
          - name: country
            data_type: varchar

```
The `usa_stores` model references `stg_stores` which is already an Ibis model and so you don't need to do anything.

Instead of crafting this `yaml` code by hand, you can use the [dbt-codegen](https://hub.getdbt.com/dbt-labs/codegen/latest/) package to automatically generate it for you. After installing it (see the previous link), you can run the following in your terminal (adjust the database, schema, and table names accordingly):

```bash
dbt run-operation generate_source --args '{"database_name": "sources_db", "schema_name": "main", "generate_columns": true, "table_names": ["stores"]}'
```

For a model, you can generate the `yml` code with the following command:

```bash
dbt run-operation generate_model_yaml --args '{"model_names": ["stg_customers"]}'
```

For further optional reading on specifying data types, see [Model Contracts - getdbt.com](https://docs.getdbt.com/docs/collaborate/govern/model-contracts) (for sources, snapshots, and SQL models) and [Seed configurations - getdbt.com](https://docs.getdbt.com/reference/seed-configs) (for seeds).

## CLI
You can use all the dbt commands you're used to, you simply need to replace `dbt` with `dbt-ibis`. For example:
```bash
dbt-ibis run --select stg_stores+
```
If you want to continue to use the `dbt` command, see the instructions on the [Advanced](./advanced.md) page on how to configure your shell accordingly.

You'll notice that for every `.ibis` file, `dbt-ibis` will generate a corresponding `.sql` file in a `__ibis_sql` subfolder. This is because `dbt-ibis` simply compiles all Ibis expressions to SQL and then let's DBT do its thing. You should not edit those files as they are overwritten every time you execute a `dbt-ibis` command. However, it might be interesting to look at them if you want to debug an expression. `dbt-ibis` will take care of cleaning up unused `.sql` files in these subfolders which might happen if you rename or delete an `.ibis` file.

You can also execute `dbt-ibis precompile` if you only want to compile the `.ibis` to `.sql` files:

```bash
# This
dbt-ibis run

# Is the samee as
dbt-ibis precompile
dbt run
```

## Editor configuration
You might want to configure your editor to treat `.ibis` files as normal Python files. In VS Code, you can do this by putting the following into your `settings.json` file:
```json
    "files.associations": {
        "*.ibis": "python"
    },
```


## Column name casing
`dbt-ibis` relies on the default Ibis behavior when it comes to quoting column names and writing them as upper or lowercase which can depend on your database. However, for databases such as Snowflake which store case-insensitive identifiers in all uppercase letters, this might not always be the most convenient for you to write dbt models. This section shows with an example what different case conventions can mean and how you can configure `dbt-ibis` to deal with it. For databases where identifiers are always case-insensitive, e.g. DuckDB, you can skip this part.

For the columns, for which `dbt-ibis` loads the data types from the `.yml` files (see above), it assumes that the column name appears exactly in the database as they are specified in the `.yml` file. Taking the following example:

```yaml
models:
  - name: customers
    columns:
      - name: customer_id
        data_type: integer
      - name: customer_name
        data_type: varchar
```
and a dbt model which references this table:

```python
from dbt_ibis import ref, depends_on

@depends_on(ref("customers"))
def model(customers):
    return customers.select("customer_id")
```
This will be rendered as the following query if you're using Snowflake:

```
SELECT
  t0."customer_id"
FROM {{ ref('customers') }} as t0
```

If the column identifier is stored as case-insensitive, this query will fail as the lowercase column `"customer_id"` does not exist. To fix this, you'll have to write the column names in the `.yml` file in uppercase:

```yaml
models:
  - name: customers
    columns:
      - name: CUSTOMER_ID
        data_type: integer
      - name: CUSTOMER_NAME
        data_type: varchar
```

and also change it in the model

```python
@depends_on(ref("customers"))
def model(customers):
    return customers.select("CUSTOMER_ID")
```

If you want to keep using lowercase column names in your model but case-insensitive (i.e. uppercase) identifiers in the database, it would look something like this:

```python
@depends_on(ref("customers"))
def model(customers):
    customers = customers.rename("snake_case")
    customers = customers.select("customer_id")
    customers = customers.rename("ALL_CAPS")
    return customers
```

This is rather cumbersome to do for every model and many of us are used to work with lowercase column names as a convention. To simplify the process, you can tell `dbt-ibis` to do these conversions for you. Going back to our original example of using all lowercase names in the `.yml` file as well as in the model, you can make that work by setting the following variables in your `dbt_project.yml` file:

```yaml
vars:
  dbt_ibis_letter_case_in_db_jaffle_shop_prod: upper
  dbt_ibis_letter_case_in_model: lower
```
This tells `dbt-ibis` that in the database, uppercase letters should be used and can be expected, and that in your dbt model you want to use lowercase letters. Both variables accept `upper` and `lower` as values. In addition, the first variable is specific to a profile (`jaffle_shop`) and target (`prod`) following the format `dbt_ibis_letter_case_in_db_{profile}_{target}`. This allows you to set different conventions for different databases. If in the above example, you would have a `dev` target which points to a local duckdb file, this `dev` target would still use the default letter case behavior of Ibis.

If all of this sounds confusing, I'd recommend to play around with the different configurations and run `dbt-ibis precompile` to inspect the generated SQL. If you have any questions, feel free to open an Issue in this repository.

See [this GitHub issue](https://github.com/ibis-project/ibis/issues/6772) for some further explanations and examples on case handling in Ibis and Snowflake.

## Limitations
* There is no database connection available in the Ibis `model` functions. Hence, you cannot use Ibis functionality which would require this.
* For non-Ibis models, seeds, snapshots, and for sources, you need to specify the data types of the columns. See "Basic example" above.