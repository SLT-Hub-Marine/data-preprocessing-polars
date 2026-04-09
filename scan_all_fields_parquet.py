#!/usr/bin/env python3

import time
import polars as pl


# The target column we want to analyze for unique fields
TARGET_COLUMN = "interpreted"
TARGET_FIELD = "samplingProtocol"

# Directory containing the parquet files to process
DATA_DIR = "data/occurrence"

def main():

    time_start = time.time()

    # count total samples
    total_count = pl.scan_parquet(DATA_DIR + "/*.parquet").select(pl.len()).collect().item()

    # count valid samples
    valid_count = (
        pl.scan_parquet(DATA_DIR + "/*.parquet")
        .select(
            pl.col(TARGET_COLUMN).struct.field(TARGET_FIELD).count().alias("cnt")
        )
        .collect(engine="streaming")
        .item()
    )

    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")

    print(f"Total samples: {total_count}")
    print(f"Valid samples: {valid_count}")

if __name__ == "__main__":
    main()
