from dbt_ibis import depends_on, ref


@depends_on(ref("stg_stores"))
def model(stores):
    return stores.filter(stores["country"] == "USA")
