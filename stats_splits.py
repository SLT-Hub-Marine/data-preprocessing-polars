import polars as pl

BATCH_SIZE = 32

TARGET_COLUMN = "interpreted"

TARGET_FIELD = "samplingProtocol"

LOW_MEMORY = True

print(f"Scanning train/dev/test splits for unique {TARGET_FIELD}...")

df_train = (
    pl.scan_parquet(
        "data/train.parquet",
        low_memory=LOW_MEMORY,
        cache=False,
    )
)
df_dev = (
    pl.scan_parquet(
        "data/dev.parquet",
        low_memory=LOW_MEMORY,
        cache=False,
    )
)
df_test = (
    pl.scan_parquet(
        "data/test.parquet",
        low_memory=LOW_MEMORY,
        cache=False,
    )
)

unique_count_train = (
    df_train.select(pl.col(TARGET_COLUMN))
            .unnest(TARGET_COLUMN)
            .select(pl.col(TARGET_FIELD).value_counts())
)

unique_count_dev = (
    df_dev.select(pl.col(TARGET_COLUMN))
          .unnest(TARGET_COLUMN)
          .select(pl.col(TARGET_FIELD).value_counts())
)

unique_count_test = (
    df_test.select(pl.col(TARGET_COLUMN))
           .unnest(TARGET_COLUMN)
           .select(pl.col(TARGET_FIELD).value_counts())
)

print(f"Unique {TARGET_FIELD} in train split:")
res_train = unique_count_train.collect(engine="streaming").to_dicts()
counter_train = {row[TARGET_FIELD][TARGET_FIELD]: row[TARGET_FIELD]["count"] for row in res_train}
print(f"Number of unique {TARGET_FIELD}: {len(counter_train)}")

print(f"\nUnique {TARGET_FIELD} in dev split:")
res_dev = unique_count_dev.collect(engine="streaming").to_dicts()
counter_dev = {row[TARGET_FIELD][TARGET_FIELD]: row[TARGET_FIELD]["count"] for row in res_dev}
print(f"Number of unique {TARGET_FIELD}: {len(counter_dev)}")

print(f"\nUnique {TARGET_FIELD} in test split:")
res_test = unique_count_test.collect(engine="streaming").to_dicts()
counter_test = {row[TARGET_FIELD][TARGET_FIELD]: row[TARGET_FIELD]["count"] for row in res_test}
print(f"Number of unique {TARGET_FIELD}: {len(counter_test)}")

set_train = set(counter_train.keys())
set_dev   = set(counter_dev.keys())
set_test  = set(counter_test.keys())

print("Train intersection Dev:", len(set_train & set_dev))
print("Train intersection Test:", len(set_train & set_test))
print("Dev intersection Test:", len(set_dev & set_test))

print("Train only:", len(set_train - set_dev - set_test))
print("Dev only:", len(set_dev - set_train - set_test))
print("Test only:", len(set_test - set_train - set_dev))

def jaccard(a, b):
    return len(a & b) / len(a | b)

print("Jaccard Train–Dev:", jaccard(set_train, set_dev))
print("Jaccard Train–Test:", jaccard(set_train, set_test))
print("Jaccard Dev–Test:", jaccard(set_dev, set_test))

print("Dev protocols not in Train:", len(set_dev - set_train))
print("Test protocols not in Train:", len(set_test - set_train))

