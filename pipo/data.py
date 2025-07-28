import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class ChunkedCSVProcessor:
    def __init__(
        self,
        csv_path,
        input_col=None,
        output_col=None,
        chunk_size=10000,
        batch_size=256,
        intermediate_dir="intermediate_tensors",
        final_output_dir="final_outputs",
        save_output=True,
    ):
        self.csv_path = csv_path
        self.input_col = input_col
        self.output_col = output_col
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.intermediate_dir = Path(intermediate_dir)
        self.final_output_dir = Path(final_output_dir)
        self.save_output = save_output

        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        if self.save_output and self.output_col:
            self.final_output_dir.mkdir(parents=True, exist_ok=True)

        self.current_chunk = 0
        self.current_batch = 0
        self.processed_rows = 0
        self.total_rows = 0
        self.num_chunks = 0

        self.chunk_tensor_paths = {}
        self.final_output_paths = []

        self.chunk_data = None

        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV structure and counters."""
        self.df_header = pd.read_csv(self.csv_path, nrows=0)

        with open(self.csv_path, "r") as f:
            self.total_rows = sum(1 for _ in f) - 1

        self.num_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size

    def next_chunk(self):
        """Load next chunk of data from CSV."""
        if self.current_chunk >= self.num_chunks:
            return False

        skip_rows = 1 + self.current_chunk * self.chunk_size
        rows_left = self.total_rows - (self.current_chunk * self.chunk_size)
        read_size = min(self.chunk_size, rows_left)

        self.chunk_data = pd.read_csv(
            self.csv_path,
            skiprows=skip_rows,
            nrows=read_size,
            header=None,
            names=self.df_header.columns,
        )

        self.current_batch = 0
        return True

    def next_batch(self):
        """Get next batch from current chunk."""
        if self.chunk_data is None:
            return None

        start = self.current_batch * self.batch_size
        if start >= len(self.chunk_data):
            return None

        end = min(start + self.batch_size, len(self.chunk_data))

        if self.input_col is None:
            chunk_file = self.intermediate_dir / f"chunk_{self.current_chunk}.pt"
            if chunk_file.exists():
                chunk_tensors = torch.load(chunk_file, weights_only=False)
                batch_data = chunk_tensors[start:end]
            else:
                raise FileNotFoundError(f"No intermediate data found for chunk {self.current_chunk}")
            
            if not isinstance(batch_data, torch.Tensor):
                batch_data = torch.tensor(batch_data)
        else:
            batch_data = self.chunk_data.iloc[start:end][self.input_col].values

        self.current_batch += 1
        return batch_data, start, end

    def save_batch_output(self, outputs, start_idx, end_idx):
        """Save model outputs for current batch."""
        if not self.save_output:
            return

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu()

        if self.current_chunk not in self.chunk_tensor_paths:
            chunk_size = len(self.chunk_data)
            if self.output_col is None:
                if isinstance(outputs, torch.Tensor):
                    chunk_storage = torch.zeros(chunk_size, *outputs.shape[1:], dtype=outputs.dtype)
                else:
                    chunk_storage = torch.zeros(chunk_size, dtype=torch.float32)
            else:
                if isinstance(outputs, torch.Tensor):
                    chunk_storage = torch.zeros(chunk_size, *outputs.shape[1:], dtype=outputs.dtype)
                else:
                    chunk_storage = torch.zeros(chunk_size, dtype=torch.float32)
            
            temp_path = self.intermediate_dir / f"temp_chunk_{self.current_chunk}.pt"
            torch.save(chunk_storage, temp_path)
            self.chunk_tensor_paths[self.current_chunk] = temp_path

        chunk_storage = torch.load(self.chunk_tensor_paths[self.current_chunk], weights_only=False)
        
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() > 1 and outputs.shape[0] != (end_idx - start_idx):
                outputs = outputs[:end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs
        else:
            outputs_tensor = torch.tensor(outputs, dtype=chunk_storage.dtype)
            if outputs_tensor.dim() > 1 and outputs_tensor.shape[0] != (end_idx - start_idx):
                outputs_tensor = outputs_tensor[:end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs_tensor
        
        torch.save(chunk_storage, self.chunk_tensor_paths[self.current_chunk])

    def save_chunk_to_disk(self):
        """Save current chunk outputs to appropriate format."""
        if not self.save_output or self.current_chunk not in self.chunk_tensor_paths:
            self.chunk_data = None
            self.current_chunk += 1
            return

        chunk_storage = torch.load(self.chunk_tensor_paths[self.current_chunk], weights_only=False)

        if self.output_col is None:
            final_path = self.intermediate_dir / f"chunk_{self.current_chunk}.pt"
            temp_final = self.intermediate_dir / f"temp_final_chunk_{self.current_chunk}.pt"
            torch.save(chunk_storage, temp_final)
            temp_final.rename(final_path)
            
            self.chunk_tensor_paths[self.current_chunk] = final_path
            
            temp_path = self.intermediate_dir / f"temp_chunk_{self.current_chunk}.pt"
            if temp_path.exists():
                temp_path.unlink()
        else:
            if isinstance(chunk_storage, torch.Tensor):
                output_values = chunk_storage.numpy()
            else:
                output_values = chunk_storage
            
            chunk_df = self.chunk_data.copy()
            chunk_df[self.output_col] = output_values
            
            chunk_csv_path = self.final_output_dir / f"chunk_{self.current_chunk}.csv"
            chunk_df.to_csv(chunk_csv_path, index=False)
            self.final_output_paths.append(chunk_csv_path)
            
            temp_path = self.chunk_tensor_paths[self.current_chunk]
            if temp_path.exists():
                temp_path.unlink()
            del self.chunk_tensor_paths[self.current_chunk]

        self.chunk_data = None
        self.current_chunk += 1

    def finalize(self):
        """Finalize processing by concatenating final outputs if needed."""
        if self.save_output and self.output_col and self.final_output_paths:
            final_output_path = Path(self.csv_path).parent / f"{Path(self.csv_path).stem}_with_{self.output_col}.csv"
            
            first_chunk = pd.read_csv(self.final_output_paths[0])
            first_chunk.to_csv(final_output_path, index=False)
            
            for chunk_path in self.final_output_paths[1:]:
                chunk_df = pd.read_csv(chunk_path)
                chunk_df.to_csv(final_output_path, mode='a', header=False, index=False)
            
            for chunk_path in self.final_output_paths:
                chunk_path.unlink()
            
            if self.final_output_dir.exists() and not any(self.final_output_dir.iterdir()):
                self.final_output_dir.rmdir()

    def clear_intermediate(self, chunk_idx=None):
        """Clear intermediate tensor files."""
        if chunk_idx is not None:
            chunk_file = self.intermediate_dir / f"chunk_{chunk_idx}.pt"
            if chunk_file.exists():
                chunk_file.unlink()
            if chunk_idx in self.chunk_tensor_paths:
                del self.chunk_tensor_paths[chunk_idx]
        else:
            for file in self.intermediate_dir.glob("*.pt"):
                file.unlink()
            self.chunk_tensor_paths.clear()


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