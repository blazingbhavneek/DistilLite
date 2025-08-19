from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for CSV processing."""

    csv_path: str  # Path for the input CSV file
    input_col: Optional[str]  # Column name for input data
    output_col: Optional[
        str
    ]  # Column name for output data, if it doesn't exist, will be created
    chunk_size: int  # Number of rows per chunk to be loaded from disk
    batch_size: int  # Number of rows per batch to be processed
    intermediate_dir: (
        str  # Directory for storing intermediate tensor/ Output from hidden layers
    )
    final_output_dir: str  # Directory for final output after processing
    save_output: bool  # Whether to save output to disk or not
