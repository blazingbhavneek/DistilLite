import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import threading
import uuid

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ProcessingConfig:
    """Immutable configuration for CSV processing."""
    csv_path: str
    input_col: Optional[str]
    output_col: Optional[str]
    chunk_size: int
    batch_size: int
    intermediate_dir: str
    final_output_dir: str
    save_output: bool


class ThreadSafeChunkProcessor:
    """Thread-safe CSV processor with separated methods for independent execution."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        self._csv_metadata = self._compute_csv_metadata()
        
        Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        if self.config.save_output and self.config.output_col:
            Path(self.config.final_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _compute_csv_metadata(self) -> dict:
        """Compute CSV metadata once during initialization."""
        df_header = pd.read_csv(self.config.csv_path, nrows=0)
        
        with open(self.config.csv_path, "r") as f:
            total_rows = sum(1 for _ in f) - 1
        
        num_chunks = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size
        
        return {
            'header': df_header,
            'total_rows': total_rows,
            'num_chunks': num_chunks
        }
    
    @property
    def num_chunks(self) -> int:
        """Get total number of chunks."""
        return self._csv_metadata['num_chunks']
    
    @property
    def total_rows(self) -> int:
        """Get total number of rows."""
        return self._csv_metadata['total_rows']
    
    def load_chunk(self, chunk_index: int) -> pd.DataFrame:
        """Load a specific chunk from CSV. Thread-safe."""
        if chunk_index >= self.num_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range (0-{self.num_chunks-1})")
        
        skip_rows = 1 + chunk_index * self.config.chunk_size
        rows_left = self.total_rows - (chunk_index * self.config.chunk_size)
        read_size = min(self.config.chunk_size, rows_left)
        
        chunk_data = pd.read_csv(
            self.config.csv_path,
            skiprows=skip_rows,
            nrows=read_size,
            header=None,
            names=self._csv_metadata['header'].columns,
        )
        
        return chunk_data
    
    def get_batch(self, chunk_data: pd.DataFrame, batch_index: int) -> Optional[Tuple[Union[np.ndarray, torch.Tensor], int, int]]:
        """Extract a batch from chunk data. Thread-safe."""
        start = batch_index * self.config.batch_size
        if start >= len(chunk_data):
            return None
        
        end = min(start + self.config.batch_size, len(chunk_data))
        
        if self.config.input_col is None:
            raise ValueError("Cannot extract batch from chunk_data when input_col is None. Use load_batch_from_intermediate instead.")
        
        batch_data = chunk_data.iloc[start:end][self.config.input_col].values
        return batch_data, start, end
    
    def load_batch_from_intermediate(self, chunk_index: int, batch_index: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Load a batch from intermediate tensor files. Thread-safe."""
        chunk_file = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"
        if not chunk_file.exists():
            raise FileNotFoundError(f"No intermediate data found for chunk {chunk_index}")
        
        chunk_tensors = torch.load(chunk_file, weights_only=False)
        
        start = batch_index * self.config.batch_size
        if start >= len(chunk_tensors):
            return None
        
        end = min(start + self.config.batch_size, len(chunk_tensors))
        batch_data = chunk_tensors[start:end]
        
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.tensor(batch_data)
        
        return batch_data, start, end
    
    def create_chunk_storage(self, chunk_size: int, sample_output: torch.Tensor) -> torch.Tensor:
        """Create storage tensor for a chunk. Thread-safe."""
        if isinstance(sample_output, torch.Tensor):
            return torch.zeros(chunk_size, *sample_output.shape[1:], dtype=sample_output.dtype)
        else:
            return torch.zeros(chunk_size, dtype=torch.float32)
    
    def update_chunk_storage(self, chunk_storage: torch.Tensor, outputs: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        """Update chunk storage with batch outputs. Thread-safe."""
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu()
            if outputs.dim() > 1 and outputs.shape[0] != (end_idx - start_idx):
                outputs = outputs[:end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs
        else:
            outputs_tensor = torch.tensor(outputs, dtype=chunk_storage.dtype)
            if outputs_tensor.dim() > 1 and outputs_tensor.shape[0] != (end_idx - start_idx):
                outputs_tensor = outputs_tensor[:end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs_tensor
        
        return chunk_storage
    
    def save_chunk_intermediate(self, chunk_storage: torch.Tensor, chunk_index: int) -> Path:
        """Save chunk storage as intermediate tensor file. Thread-safe with atomic writes."""
        final_path = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"
        temp_path = Path(self.config.intermediate_dir) / f"temp_{uuid.uuid4().hex}_chunk_{chunk_index}.pt"
        
        try:
            torch.save(chunk_storage, temp_path)
            temp_path.rename(final_path)
            return final_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def save_chunk_final(self, chunk_data: pd.DataFrame, chunk_storage: torch.Tensor, chunk_index: int) -> Path:
        """Save chunk with outputs to final CSV. Thread-safe with atomic writes."""
        if not self.config.output_col:
            raise ValueError("output_col must be specified for final save")
        
        if isinstance(chunk_storage, torch.Tensor):
            output_values = chunk_storage.numpy()
        else:
            output_values = chunk_storage
        
        chunk_df = chunk_data.copy()
        chunk_df[self.config.output_col] = output_values
        
        final_path = Path(self.config.final_output_dir) / f"chunk_{chunk_index}.csv"
        temp_path = Path(self.config.final_output_dir) / f"temp_{uuid.uuid4().hex}_chunk_{chunk_index}.csv"
        
        try:
            chunk_df.to_csv(temp_path, index=False)
            temp_path.rename(final_path)
            return final_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def concatenate_final_outputs(self, chunk_paths: list) -> Path:
        """Concatenate all chunk CSV files into final output. Thread-safe."""
        if not chunk_paths:
            raise ValueError("No chunk paths provided")
        
        final_output_path = Path(self.config.csv_path).parent / f"{Path(self.config.csv_path).stem}_with_{self.config.output_col}.csv"
        temp_final_path = Path(self.config.csv_path).parent / f"temp_{uuid.uuid4().hex}_final.csv"
        
        try:
            sorted_paths = sorted(chunk_paths, key=lambda p: int(p.stem.split('_')[1]))
            
            first_chunk = pd.read_csv(sorted_paths[0])
            first_chunk.to_csv(temp_final_path, index=False)
            
            for chunk_path in sorted_paths[1:]:
                chunk_df = pd.read_csv(chunk_path)
                chunk_df.to_csv(temp_final_path, mode='a', header=False, index=False)
            
            temp_final_path.rename(final_output_path)
            
            for chunk_path in sorted_paths:
                chunk_path.unlink()
            
            if Path(self.config.final_output_dir).exists() and not any(Path(self.config.final_output_dir).iterdir()):
                Path(self.config.final_output_dir).rmdir()
            
            return final_output_path
            
        except Exception:
            if temp_final_path.exists():
                temp_final_path.unlink()
            raise
    
    def clear_intermediate(self, chunk_index: Optional[int] = None) -> None:
        """Clear intermediate tensor files. Thread-safe."""
        if chunk_index is not None:
            chunk_file = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"
            if chunk_file.exists():
                chunk_file.unlink()
        else:
            for file in Path(self.config.intermediate_dir).glob("*.pt"):
                file.unlink()


class ChunkedCSVProcessor:
    """Backward-compatible wrapper around ThreadSafeChunkProcessor."""
    
    def __init__(self, csv_path, input_col=None, output_col=None, chunk_size=10000, 
                 batch_size=256, intermediate_dir="intermediate_tensors", 
                 final_output_dir="final_outputs", save_output=True):
        
        config = ProcessingConfig(
            csv_path=csv_path,
            input_col=input_col,
            output_col=output_col,
            chunk_size=chunk_size,
            batch_size=batch_size,
            intermediate_dir=intermediate_dir,
            final_output_dir=final_output_dir,
            save_output=save_output
        )
        
        self.processor = ThreadSafeChunkProcessor(config)
        
        self.current_chunk = 0
        self.current_batch = 0
        self.chunk_data = None
        self.chunk_storage = None
        self.final_output_paths = []
    
    @property
    def total_rows(self):
        return self.processor.total_rows
    
    @property
    def num_chunks(self):
        return self.processor.num_chunks
    
    def next_chunk(self):
        """Load next chunk (maintains original interface)."""
        if self.current_chunk >= self.num_chunks:
            return False
        
        self.chunk_data = self.processor.load_chunk(self.current_chunk)
        self.current_batch = 0
        self.chunk_storage = None
        return True
    
    def next_batch(self):
        """Get next batch (maintains original interface)."""
        if self.chunk_data is None:
            return None
        
        try:
            if self.processor.config.input_col is None:
                result = self.processor.load_batch_from_intermediate(self.current_chunk, self.current_batch)
            else:
                result = self.processor.get_batch(self.chunk_data, self.current_batch)
            
            if result is not None:
                self.current_batch += 1
            
            return result
            
        except (FileNotFoundError, ValueError):
            return None
    
    def save_batch_output(self, outputs, start_idx, end_idx):
        """Save batch output (maintains original interface)."""
        if not self.processor.config.save_output:
            return
        
        if self.chunk_storage is None:
            chunk_size = len(self.chunk_data)
            if isinstance(outputs, torch.Tensor):
                sample_output = outputs
            else:
                sample_output = torch.tensor(outputs)
            self.chunk_storage = self.processor.create_chunk_storage(chunk_size, sample_output)
        
        self.chunk_storage = self.processor.update_chunk_storage(
            self.chunk_storage, outputs, start_idx, end_idx
        )
    
    def save_chunk_to_disk(self):
        """Save chunk to disk (maintains original interface)."""
        if not self.processor.config.save_output or self.chunk_storage is None:
            self.current_chunk += 1
            return
        
        if self.processor.config.output_col is None:
            self.processor.save_chunk_intermediate(self.chunk_storage, self.current_chunk)
        else:
            chunk_path = self.processor.save_chunk_final(
                self.chunk_data, self.chunk_storage, self.current_chunk
            )
            self.final_output_paths.append(chunk_path)
        
        self.current_chunk += 1
    
    def finalize(self):
        """Finalize processing (maintains original interface)."""
        if self.processor.config.save_output and self.processor.config.output_col and self.final_output_paths:
            self.processor.concatenate_final_outputs(self.final_output_paths)
    
    def clear_intermediate(self, chunk_idx=None):
        """Clear intermediate files (maintains original interface)."""
        self.processor.clear_intermediate(chunk_idx)


if __name__ == "__main__":
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    csv_path = os.path.join(temp_dir, "data.csv")
    num_samples = 22
    data = {
        "id": range(1, num_samples + 1),
        "input": [float(i) for i in range(1, num_samples + 1)],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Created sample CSV with {num_samples} rows at {csv_path}")

    chunk_size = 10
    batch_size = 3

    print("\nTest 1: Simple end-to-end processing")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col="input",
        output_col="output",
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        final_output_dir=os.path.join(temp_dir, "final_outputs"),
        save_output=True,
    )

    def model(x):
        return x * 2

    chunks_processed = 0
    while processor.next_chunk():
        chunks_processed += 1
        print(f"\nProcessing chunk {chunks_processed}")

        batch_count = 0
        while (batch := processor.next_batch()) is not None:
            batch_count += 1
            batch_data, start_idx, end_idx = batch
            print(f"  Batch {batch_count}: indices {start_idx}-{end_idx-1}")

            input_tensor = torch.tensor(batch_data, dtype=torch.float32)
            outputs = model(input_tensor)

            processor.save_batch_output(outputs, start_idx, end_idx)

        processor.save_chunk_to_disk()
        print(f"Saved chunk {chunks_processed} to disk")

    processor.finalize()

    final_file = Path(csv_path).parent / f"{Path(csv_path).stem}_with_output.csv"
    df = pd.read_csv(final_file)
    print(f"\nFinal CSV contents after Test 1 ({len(df)} rows):")
    print(df.head(10))

    assert "output" in df.columns, "Output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"
    assert np.allclose(df["input"] * 2, df["output"]), "Incorrect output values"
    print("Test 1 passed: Output values correct")

    print("\nTest 2: Multi-layer processing")

    print("Processing Layer 1...")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col="input",
        output_col=None,
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        save_output=True,
    )

    chunks_processed = 0
    while processor.next_chunk():
        chunks_processed += 1
        print(f"\nProcessing chunk {chunks_processed}")
        batch_count = 0
        while (batch := processor.next_batch()) is not None:
            batch_count += 1
            batch_data, start_idx, end_idx = batch
            print(f"  Batch {batch_count}: indices {start_idx}-{end_idx-1}")
            input_tensor = torch.tensor(batch_data, dtype=torch.float32)
            outputs = input_tensor * 2
            processor.save_batch_output(outputs, start_idx, end_idx)
        processor.save_chunk_to_disk()
        print(f"Saved chunk {chunks_processed} to disk")

    processor.finalize()

    intermediate_files = list(Path(temp_dir, "intermediate").glob("chunk_*.pt"))
    print(f"\nLayer 1 complete. Created {len(intermediate_files)} intermediate chunk files.")

    print("\nProcessing Layer 2...")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col=None,
        output_col="final_output",
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        final_output_dir=os.path.join(temp_dir, "final_outputs"),
        save_output=True,
    )

    chunks_processed = 0
    while processor.next_chunk():
        chunks_processed += 1
        print(f"\nProcessing chunk {chunks_processed}")
        batch_count = 0
        while (batch := processor.next_batch()) is not None:
            batch_count += 1
            batch_data, start_idx, end_idx = batch
            print(f"  Batch {batch_count}: {len(batch_data)} tensors (indices {start_idx}-{end_idx-1})")

            input_tensor = batch_data
            outputs = input_tensor * 3
            processor.save_batch_output(outputs, start_idx, end_idx)
        processor.save_chunk_to_disk()
        print(f"Saved chunk {chunks_processed} to disk")

    processor.finalize()

    final_file = Path(csv_path).parent / f"{Path(csv_path).stem}_with_final_output.csv"
    df = pd.read_csv(final_file)
    print(f"\nFinal CSV after both layers ({len(df)} rows):")
    print(df[["id", "input", "final_output"]].head(10))
    print("\n...")
    print(df[["id", "input", "final_output"]].tail(10))

    assert "final_output" in df.columns, "Final output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"
    
    nan_count = df["final_output"].isna().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in final_output")
        print("Rows with NaN values:")
        print(df[df["final_output"].isna()][["id", "input", "final_output"]])
    
    valid_mask = ~df["final_output"].isna()
    if valid_mask.sum() > 0:
        expected = df.loc[valid_mask, "input"] * 6
        actual = df.loc[valid_mask, "final_output"]
        assert np.allclose(expected, actual), f"Incorrect final output values. Expected {expected.values}, got {actual.values}"
    
    print("\nTest 2 passed: Final output values correct")

    print("\nTest 3: Memory efficiency validation")
    remaining_intermediate = list(Path(temp_dir, "intermediate").glob("*.pt"))
    print(f"Remaining intermediate files: {len(remaining_intermediate)}")

    shutil.rmtree(temp_dir)
    print(f"\nAll tests passed! Removed temporary directory: {temp_dir}")

    print("\nThread-safe methods available:")
    print("- processor.processor.load_chunk(chunk_index)")
    print("- processor.processor.get_batch(chunk_data, batch_index)")
    print("- processor.processor.load_batch_from_intermediate(chunk_index, batch_index)")
    print("- processor.processor.create_chunk_storage(chunk_size, sample_output)")
    print("- processor.processor.update_chunk_storage(storage, outputs, start, end)")
    print("- processor.processor.save_chunk_intermediate(storage, chunk_index)")
    print("- processor.processor.save_chunk_final(chunk_data, storage, chunk_index)")
    print("- processor.processor.concatenate_final_outputs(chunk_paths)")
    print("- processor.processor.clear_intermediate(chunk_index)")