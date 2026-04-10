"""
This script converts the data splits from Parquet format to a single SQLite database.
Each split (train, dev, test) is stored in a separate table within the database.
With a SQLite-based dataset, we can efficiently query and access arbitrary individual samples where the parquet format does not support efficient random access.
The script handles complex data types by encoding them as JSON strings for storage in SQLite.

FIXME: SQLite is not efficient in data footprint and can result in a much larger file size (~250 GB) compared to the original Parquet files (~35 GB).
FIXME: This script should not been used as corresponding dataset class is not implemented yet.
"""

import json
import re
import sqlite3
from itertools import islice

import polars as pl
import tqdm

TARGET_COLUMN = "interpreted"
LOW_MEMORY = True
OUT_DATABASE_PATH = "data/dataset.sqlite"

INPUT_TABLE_MAPPING = {
    "data/train.parquet": "train",
    "data/dev.parquet": "dev",
    "data/test.parquet": "test",
}


def sqlite_safe_batch(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []

    def to_json_text(x):
        if x is None:
            return None
        if isinstance(x, pl.Series):
            x = x.to_list()
        return json.dumps(x, ensure_ascii=False, default=str)

    for name, dtype in df.schema.items():
        base = dtype.base_type()

        if base is pl.Struct:
            exprs.append(pl.col(name).struct.json_encode().alias(name))
        elif base is pl.List or base is pl.Array:
            exprs.append(
                pl.col(name)
                .map_elements(
                    to_json_text,
                    return_dtype=pl.String,
                )
                .alias(name)
            )
        else:
            exprs.append(pl.col(name))

    return df.select(exprs)


def sqlite_type_from_polars(dtype: pl.DataType) -> str:
    base = dtype.base_type()
    if base in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Boolean,
    }:
        return "INTEGER"
    if base in {pl.Float32, pl.Float64}:
        return "REAL"
    return "TEXT"


def create_table(
    conn: sqlite3.Connection, table_name: str, schema: dict[str, pl.DataType]
) -> None:
    cols = ['"id" INTEGER PRIMARY KEY AUTOINCREMENT']
    cols.extend(
        f'"{name}" {sqlite_type_from_polars(dtype)}' for name, dtype in schema.items()
    )
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(cols)})')


def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def insert_batch(
    conn: sqlite3.Connection,
    table_name: str,
    df: pl.DataFrame,
    insert_chunk_size: int = 1000,
) -> None:
    cols = df.columns
    placeholders = ",".join(["?"] * len(cols))
    quoted_cols = ",".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

    for rows in batched(df.iter_rows(), insert_chunk_size):
        conn.executemany(sql, rows)


def estimate_rows_from_explain(lf: pl.LazyFrame) -> int | None:
    plan = lf.explain()
    m = re.search(r"ESTIMATED ROWS:\s*([\d,]+)", plan)
    return int(m.group(1).replace(",", "")) if m else None


def open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path,
        cached_statements=0,
    )
    return conn


def main():

    # Set a smaller chunk size for streaming to reduce memory usage
    pl.Config.set_streaming_chunk_size(128)

    for input_path, output_table in tqdm.tqdm(
        INPUT_TABLE_MAPPING.items(),
        total=len(INPUT_TABLE_MAPPING),
        desc="Files",
        unit="file",
    ):
        lf = (
            pl.scan_parquet(
                input_path,
                low_memory=LOW_MEMORY,
                cache=False,
            )
            .select(pl.col(TARGET_COLUMN))
            .unnest(TARGET_COLUMN)
        )

        schema = lf.collect_schema()

        estimated = estimate_rows_from_explain(lf)
        pbar = (
            tqdm.tqdm(
                total=estimated,
                desc=f"Writing {output_table}",
                unit="rows",
            )
            if estimated is not None
            else tqdm.tqdm(
                desc=f"Writing {output_table}",
                unit="rows",
            )
        )

        with open_sqlite(OUT_DATABASE_PATH) as conn:
            create_table(conn, output_table, schema)

            try:
                for df in lf.collect_batches(
                    chunk_size=10000,
                    maintain_order=False,
                    lazy=True,
                    engine="streaming",
                ):
                    df = sqlite_safe_batch(df)

                    insert_batch(conn, output_table, df, insert_chunk_size=1000)
                    conn.commit()

                    pbar.update(df.height)

            finally:
                pbar.close()


if __name__ == "__main__":
    main()
