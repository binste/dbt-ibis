# Introduction
With dbt-ibis you can write your [dbt](https://www.getdbt.com/) models using [Ibis](https://ibis-project.org/).

Supported dbt adapters:
* DuckDB
* Snowflake
* BigQuery
* Postgres
* Redshift
* Trino
* MySQL
* SQLite
* Oracle

## Installation
You can install `dbt-ibis` via pip or conda:
```bash
pip install dbt-ibis
# or
conda install -c conda-forge dbt-ibis
```

In addition, you'll need to install the relevant [`ibis` backend](https://ibis-project.org/install) for your database.

## Why dbt and Ibis go great together
You can read about the advantages of combining dbt and Ibis in [this blog post](https://ibis-project.org/posts/dbt-ibis/).
