# data-preprocessing-polars

This directory contains the code for data preprocessing using `Polars`, which is a fast DataFrame library in Python.

The code is designed to process the `interpreted` data stored in `Parquet` files, extract relevant fields, filter out invalid rows, and generate training, validation, and test splits for model training and evaluation.

## Getting Started

You can clone the repository with
```sh
git clone git@github.com:SLT-Hub-Marine/data-preprocessing-polars.git
```

*Optionally, create/activate a conda/virtual environment or enter a dev container to isolate the dependencies for this project.*

Then, you can navigate to the directory and install the required dependencies using pip:
```sh
cd data-preprocessing-polars
pip install -r requirements.txt
```

Once you have the environment set up, you also need to link the data directory containing the Parquet files to the current directory. You can do this by creating a symbolic link:
```sh
ln -s /path/to/your/data ./
```

The directory structure should look like this:
```
data-preprocessing-polars/
├── data/
|   └── occurrence/  # This is the symbolic link to the original data directory
├── gen_data_splits.py
├── requirements.txt
└── README.md
```

After setting up the environment, you can run the main data processing script:
```sh
python gen_data_splits.py
```

`gen_data_splits.py` is the main script that performs the data processing and splitting. It reads the Parquet files, applies the necessary transformations and filters, and writes the resulting splits to disk in Parquet format.

You may see the following output in the console as the script runs:
```
YYYY-MM-DD HH:MM:SS,mmm INFO __main__ Generating train split...
YYYY-MM-DD HH:MM:SS,mmm INFO __main__ Generating dev split...
YYYY-MM-DD HH:MM:SS,mmm INFO __main__ Generating test split...
YYYY-MM-DD HH:MM:SS,mmm INFO __main__ Time taken: 1314.02 seconds
```

The generated splits will be saved in the `data` directory as `train.parquet`, `dev.parquet`, and `test.parquet`. You should have the following directory structure after running the script:
```
data-preprocessing-polars/
├── data/
|   ├── occurrence/  # This is the symbolic link to the original data directory
|   ├── train.parquet
|   ├── dev.parquet
|   └── test.parquet
├── gen_data_splits.py
├── requirements.txt
└── README.md
```

To generate some statistics about the generated splits:
```sh
python stats_splits.py
```

To verify that the generated splits, you can use the `dataset.py` script to load the splits and inspect their contents. `dataset.py` contains a demo to run a simple test to load the dataset and print out some samples. You can run it with:
```sh
python dataset.py
```
