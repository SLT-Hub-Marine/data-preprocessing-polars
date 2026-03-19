from pathlib import Path
import sys
import time

import polars as pl

from const import INTERESTED_FIELDS

import logging

# Fixed random seed for reproducibility
SEED = 42

# The target column and fields we want to keep non-null
TARGET_COLUMN = "interpreted"
TARGET_FIELDS = INTERESTED_FIELDS

# Directory containing the parquet files to process
DATA_DIR = Path("data/occurrence")

# Output directory for the generated train/dev/test splits
OUTPUT_DIR = Path("data")

def main():
    
    # Set up logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)  # stdout
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Start timing

    time_start = time.time()

    # Create an expression to filter rows where all target fields in the 'interpreted' struct are not null

    valid_expr = pl.all_horizontal(
        [
            pl.col(TARGET_COLUMN).struct.field(f).is_not_null()
            for f in TARGET_FIELDS
        ]
    )

    # Base dataset containing all valid rows with an additional '_bucket' column for splitting
    # When loading the parquet files, we specify 'row_index_name' to get a unique row ID for hashing
    # The hash of the row ID is used to assign each row to a bucket (0-99) for splitting into train/dev/test
    # Ideally, the hash function should uniformly distribute rows into buckets, ensuring a random split

    base = (
        pl.scan_parquet(
            DATA_DIR / "*.parquet",
            row_index_name="_row_id",
        )
        .filter(valid_expr)
        .with_columns(
            (
                pl.col("_row_id")
                .hash(seed=SEED)
                % 100
            ).alias("_bucket")
        )
    )

    # Create train/dev/test splits based on the '_bucket' column
    # - Train: buckets 0-79 (80% of data)
    # - Dev: buckets 80-89 (10% of data)
    # - Test: buckets 90-99 (10% of data)
    # We drop the '_row_id' and '_bucket' columns from the final splits as they are no longer needed

    train = (
        base
        .filter(pl.col("_bucket") < 80)
        .drop(["_row_id", "_bucket"])
    )
    dev = (
        base
        .filter((pl.col("_bucket") >= 80) & (pl.col("_bucket") < 90))
        .drop(["_row_id", "_bucket"])
    )
    test = (
        base
        .filter(pl.col("_bucket") >= 90)
        .drop(["_row_id", "_bucket"])
    )

    # At this point, 'train', 'dev', and 'test' are lazy frames that represent the respective splits,
    # which means that the actual data has not been loaded or processed yet, 
    # only the operations to create these splits have been defined.
    # The data will be processed and loaded when we call 'sink_parquet' to write the splits to disk.

    logger.info("Generating train split...")
    train.sink_parquet(OUTPUT_DIR / "train.parquet")
    
    logger.info("Generating dev split...")
    dev.sink_parquet(OUTPUT_DIR / "dev.parquet")

    logger.info("Generating test split...")
    test.sink_parquet(OUTPUT_DIR / "test.parquet")

    # End timing
    time_end = time.time()

    # Log the time taken for the entire process
    logger.info(f"Time taken: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    main()
