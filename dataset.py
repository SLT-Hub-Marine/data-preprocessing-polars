"""
This module defines the MarineDataset class, which is a PyTorch IterableDataset for loading and processing marine occurrence data from Parquet files using Polars.
The dataset supports streaming data in chunks to reduce memory usage, and includes functionality for sharding files across multiple workers when used with a DataLoader.
Each sample is represented as a DataSample instance, and batches of samples are collated into DataBatch instances for efficient processing in training loops.
"""

import glob
import random
from typing import Optional

import polars as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path

import tqdm

TARGET_COLUMN = "interpreted"


class DataSample:
    def __init__(self, sample: dict):
        """
        Initialize a data sample.

        Args:
            sample: A dictionary containing the sample data.
        """
        self.samplingProtocol = sample.get("samplingProtocol")
        self.bathymetry = sample.get("bathymetry")
        self.shoredistance = sample.get("shoredistance")
        self.decimalLatitude = sample.get("decimalLatitude")
        self.decimalLongitude = sample.get("decimalLongitude")
        self.geodeticDatum = sample.get("geodeticDatum")
        self.kingdom = sample.get("kingdom")
        self.phylum = sample.get("phylum")
        self.class_ = sample.get("class")
        self.order = sample.get("order")
        self.family = sample.get("family")
        self.genus = sample.get("genus")
        self.species = sample.get("species")
        self.scientificName = sample.get("scientificName")
        self.occurrenceID = sample.get("occurrenceID")
        self.raw = sample


class DataBatch:
    def __init__(self, samples: list[DataSample]):
        """
        Initialize a batch from a list of samples.

        Args:
            samples: A list of samples to include in the batch.
        """
        self.samplingProtocol = [sample.samplingProtocol for sample in samples]
        self.bathymetry, self.bathymetryMask = self.list_to_tensor(
            [sample.bathymetry for sample in samples],
            default_value=0,
            dtype=torch.float32,
        )
        self.shoredistance, self.shoredistanceMask = self.list_to_tensor(
            [sample.shoredistance for sample in samples],
            default_value=0,
            dtype=torch.float32,
        )
        self.decimalLatitude, self.decimalLatitudeMask = self.list_to_tensor(
            [sample.decimalLatitude for sample in samples],
            default_value=0,
            dtype=torch.float32,
        )
        self.decimalLongitude, self.decimalLongitudeMask = self.list_to_tensor(
            [sample.decimalLongitude for sample in samples],
            default_value=0,
            dtype=torch.float32,
        )
        self.geodeticDatum = [sample.geodeticDatum for sample in samples]
        self.kingdom = [sample.kingdom for sample in samples]
        self.phylum = [sample.phylum for sample in samples]
        self.class_ = [sample.class_ for sample in samples]
        self.order = [sample.order for sample in samples]
        self.family = [sample.family for sample in samples]
        self.genus = [sample.genus for sample in samples]
        self.species = [sample.species for sample in samples]
        self.scientificName = [sample.scientificName for sample in samples]
        self.occurrenceID = [sample.occurrenceID for sample in samples]
        self.raw = [sample.raw for sample in samples]

    @staticmethod
    def list_to_tensor(
        lst, default_value=0, dtype=torch.float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        res = torch.tensor(
            [default_value if x is None else x for x in lst], dtype=dtype
        )
        mask = torch.tensor([x is not None for x in lst], dtype=torch.bool)
        return res, mask


class MarineDataset(IterableDataset):
    def __init__(
        self,
        source_glob: str,
        filter_expr: Optional[pl.Expr] = None,
        polars_low_memory: bool = True,
        polars_chunk_size: int = 2048,
        shuffle_files: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the dataset by loading the Parquet file.

        Args:
            source_glob: The path to the Parquet file(s) containing the dataset. Can include glob patterns.
            filter_expr: An optional Polars expression to filter the dataset.
            polars_chunk_size: The chunk size to use when streaming data with Polars.
            shuffle_files: Whether to shuffle the order of files when using multiple files.
            seed: The random seed to use for shuffling files (if enabled).
        """
        # Set a smaller chunk size for streaming to reduce memory usage
        pl.Config.set_streaming_chunk_size(128)

        self.files = list(map(lambda p: Path(p), sorted(glob.glob(source_glob))))
        self.polars_low_memory = polars_low_memory
        self.filter_expr = filter_expr
        self.polars_chunk_size = polars_chunk_size
        self.shuffle_files = shuffle_files
        self.seed = seed

        self.total_samples = (
            pl.scan_parquet(source_glob, low_memory=self.polars_low_memory, cache=False)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )

    def _shard_files_for_worker(self) -> Optional[list[Path]]:
        """
        Shard the list of files for the current worker when using a DataLoader with multiple workers.

        Returns:
            A list of file paths assigned to the current worker, or None if the worker ID is out of range.
        """
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
            worker_seed = self.seed
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
            worker_seed = worker.seed

        if (worker_id >= num_workers) or (worker_id < 0):
            return None
        files = self.files[worker_id::num_workers]

        if self.shuffle_files:
            rng = random.Random(worker_seed)
            rng.shuffle(files)

        return files

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            The total number of samples.
        """
        return self.total_samples

    def __iter__(self):
        """
        Iterate over the dataset, yielding one sample at a time.

        Yields:
            A DataSample instance for each sample in the dataset.
        """
        files = self._shard_files_for_worker()
        if not files:
            return

        lf = (
            pl.scan_parquet(files, low_memory=self.polars_low_memory, cache=False)
            .select(pl.col(TARGET_COLUMN))
            .unnest(TARGET_COLUMN)
        )

        if self.filter_expr is not None:
            lf = lf.filter(self.filter_expr)

        for df in lf.collect_batches(
            chunk_size=self.polars_chunk_size,
            maintain_order=not self.shuffle_files,
        ):
            if df.height == 0:
                continue

            dicts = df.to_dicts()

            for sample_dict in dicts:
                yield DataSample(sample_dict)

    def __getitem__(self, idx):
        """
        Indexing is not supported for this dataset. Use an iterator or DataLoader to access samples.
        """
        raise NotImplementedError(
            "Indexing is not implemented for MarineDataset. Use an iterator or DataLoader to access samples."
        )

    def collate_fn(self, batch: list[DataSample]) -> DataBatch:
        """
        Collate a list of samples into a batch.

        Args:
            batch: A list of DataSample instances to collate into a batch.
        Returns:
            A DataBatch instance containing the collated samples.
        """
        data_batch = DataBatch(batch)
        return data_batch


if __name__ == "__main__":
    # Example usage
    dataset = MarineDataset("data/train.parquet")
    print(f"Total samples in train split: {len(dataset)}")

    dataset_dev = MarineDataset("data/dev.parquet")
    print(f"Total samples in dev split: {len(dataset_dev)}")

    dataset_test = MarineDataset("data/test.parquet")
    print(f"Total samples in test split: {len(dataset_test)}")

    # Get the first sample and print its contents
    sample = next(iter(dataset))
    print(f"First sample: {sample.__dict__}")

    # Create a DataLoader to iterate over the dataset in batches
    BATCH_SIZE = 32
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn
    )
    for batch in dataloader:
        print(f"First batch: {batch.__dict__}")
        break

    NUM_BATCHES_TO_TEST = 10000
    print(f"Test iterating through batches... ({NUM_BATCHES_TO_TEST} batches)")
    for batch_idx, batch in enumerate(
        tqdm.tqdm(
            dataloader,
            total=len(dataset) // BATCH_SIZE,
            desc="Iterating batches",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    ):
        # Test iterating through the batches
        if batch_idx == NUM_BATCHES_TO_TEST:
            tqdm.tqdm.write("Reached 10,000 batches, stopping iteration.")
            break

    print("Done.")
