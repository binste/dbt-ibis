# dbt-ibis
With dbt-ibis you can write your [dbt](https://www.getdbt.com/) models using [Ibis](https://ibis-project.org/).

This package is in very early development. Things might go wrong. [Feedback](https://github.com/binste/dbt-ibis/issues) and contributions are welcome!

Supported adapters:
* DuckDB
* Soon to come:
  * Snowflake
  * ... (hopefully all which are supported by both dbt and Ibis)

## Basic example
```bash
pip install dbt-ibis
```

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

Whenever your Ibis model references either a source, a seed, a snapshot, or a SQL model, you'll need to define the column data types as described in [Model Contracts - getdbt.com](https://docs.getdbt.com/docs/collaborate/govern/model-contracts) (`data_type` refers to the data types as they are called by your database system) (for sources, snapshots, and SQL models) or in [Seed configurations - getdbt.com](https://docs.getdbt.com/reference/seed-configs) (for seeds). If you reference another Ibis model, this is not necessary. In the examples above, you would need to provide it for the `stores` source table:

```yml
sources:
  - name: sources_db
    ...
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
For more examples, including column data type definitions, see the [demo project](./demo_project/jaffle_shop/).

You can use all the dbt commands you're used to, you simply need to replace `dbt` with `dbt-ibis`. For example:
```bash
dbt-ibis run --select stg_stores+
```

You'll notice that for every `.ibis` file, `dbt-ibis` will generate a corresponding `.sql` file in a `__ibis_sql` subfolder. This is because `dbt-ibis` simply compiles all Ibis expressions to SQL and then let's DBT do its thing. You should not edit those files as they are overwritten every time you execute a `dbt-ibis` command. However, it might be interesting to look at them if you want to debug an expression.

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

## Limitations
* There is no database connection available in the Ibis `model` functions. Hence, you cannot use Ibis functions which would require this.
* For non-Ibis models, seeds, snapshots, and for sources, you need to specify the data types of the columns. See "Basic example" above.

## Integration with DBT
There are [discussions](https://github.com/dbt-labs/dbt-core/pull/5274#issuecomment-1132772028) on [adding a plugin system to dbt](https://github.com/dbt-labs/dbt-core/issues/6184) which could be used to provide first-class support for other modeling languages such as Ibis (see [this PoC](https://github.com/dbt-labs/dbt-core/pull/6296) by dbt and the [discussion on Ibis as a dataframe API](https://github.com/dbt-labs/dbt-core/discussions/5738)) or PRQL (see [dbt-prql](https://github.com/PRQL/dbt-prql)).

As this feature didn't make it [onto the roadmap of dbt for 2023](https://github.com/dbt-labs/dbt-core/blob/main/docs/roadmap/2023-02-back-to-basics.md), I've decided to create `dbt-ibis` to bridge the time until then. Apart from the limitations mentioned above, I think this approach can scale reasonably well. However, the goal is to migrate to the official plugin system as soon as it's available.


## Development
```bash
pip install -e '.[dev]'
```

You can run linters and tests with
```bash
hatch run linters
hatch run tests
```
