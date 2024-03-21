# dbt-ibis
With dbt-ibis you can write your [dbt](https://www.getdbt.com/) models using [Ibis](https://ibis-project.org/). You can find the full documentation [here](https://binste.github.io/dbt-ibis/intro.html).

A simple dbt-ibis model looks like this:
```python
from dbt_ibis import depends_on, ref


@depends_on(ref("stg_stores"))
def model(stores):
    return stores.filter(stores["country"] == "USA")
```

You can install `dbt-ibis` via pip or conda:
```bash
pip install dbt-ibis
# or
conda install -c conda-forge dbt-ibis
```

In addition, you'll need to install the relevant [`Ibis` backend](https://ibis-project.org/install) for your database.

You can read about the advantages of combining dbt and Ibis in [this blog post](https://ibis-project.org/posts/dbt-ibis/).


## Development
```bash
pip install -e '.[dev]'
```

You can run linters and tests with
```bash
hatch run linters
hatch run tests
```
