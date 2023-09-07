# dbt-ibis
With dbt-ibis you can write your [dbt](https://www.getdbt.com/) models using [Ibis](https://ibis-project.org/).

This package is in very early development. Things might go wrong. [Feedback](https://github.com/binste/dbt-ibis/issues) and contributions are welcome!

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

Whenever your Ibis model references either a source or a SQL model, you'll need to define the column data types as described in [Model Contracts - getdbt.com](https://docs.getdbt.com/docs/collaborate/govern/model-contracts). If you reference another Ibis model, this is not necessary. For more examples, including column data type definitions, see the [demo project](./demo_project/jaffle_shop/).

You can use all the dbt commands you're used to, you simply need to replace `dbt` with `dbt-ibis`. For example:
```bash
dbt-ibis run --select stg_stores+
```


You'll notice that for every `.ibis` file, `dbt-ibis` will generate a corresponding `.sql` file in a `__ibis_sql` subfolder. This is because `dbt-ibis` simply compiles all Ibis expressions to SQL and then let's DBT do its thing. You should not edit those files as they are overwritten every time you execute a `dbt-ibis` command. However, it might be interesting to look at them if you want to debug an expression.

## Editor configuration
You might want to configure your editor to treat `.ibis` files as normal Python files. In VS Code, you can do this by putting the following into your `settings.json` file:
```json
    "files.associations": {
        "*.ibis": "python"
    },
```

## Limitations
* There is no database connection available in the Ibis `model` functions. Hence, you cannot use Ibis functions which would require this.
* It currently only works with duckdb. I'll soon add support for Snowflake and, if possible, all databases which are supported by both dbt and Ibis.

## Integration with DBT
There are [discussions](https://github.com/dbt-labs/dbt-core/pull/5274#issuecomment-1132772028) on [adding a plugin system to dbt](https://github.com/dbt-labs/dbt-core/issues/6184) which could be used to provide first-class support for other modeling languages such as Ibis (also see [this PoC](https://github.com/dbt-labs/dbt-core/pull/6296) by dbt) or PRQL (see [dbt-prql](https://github.com/PRQL/dbt-prql)).

As this feature didn't make it [onto the roadmap of dbt for 2023](https://github.com/dbt-labs/dbt-core/blob/main/docs/roadmap/2023-02-back-to-basics.md), I've decided to create `dbt-ibis` to bridge the time until then. Apart from the limitations mentioned above, I think this approach can scale reasonably well. However, the goal is to migrate to the official plugin system as soon as it's available.


## Development
```bash
pip install -e '.[dev]'
```
Tests, linting, etc. will follow.