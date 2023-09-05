from pathlib import Path

import duckdb
import pandas as pd


def main():
    db_path = Path("sources_db.duckdb")
    if db_path.exists():
        db_path.unlink()
    # store_id is intentionally a varchar so we can cast it in the staging model
    stores = pd.DataFrame(
        [["1", "San Francisco"], ["2", "New York"], ["3", "Berlin"]],
        columns=["store_id", "store_name"],
    )
    con = duckdb.connect(str(db_path))
    con.register("stores", stores)
    con.execute("CREATE TABLE stores AS SELECT * FROM stores")


if __name__ == "__main__":
    main()
