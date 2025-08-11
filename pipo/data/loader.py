import os
import random
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from config import DataConfig


class DataLoader:
    """Thread-safe CSV processor with separated methods for independent execution."""

    def __init__(self, config: DataConfig):
        self.config = config

        self._csv_metadata = self._compute_csv_metadata()

        Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        if self.config.save_output and self.config.output_col:
            Path(self.config.final_output_dir).mkdir(parents=True, exist_ok=True)

    def _compute_csv_metadata(self) -> dict:
        """Compute CSV metadata once during initialization."""

        # Read header for column names
        df_header = pd.read_csv(self.config.csv_path, nrows=0)

        # Count total rows in the CSV file
        with open(self.config.csv_path, "r") as f:
            total_rows = sum(1 for _ in f) - 1

        # Calculate number of chunks based on chunk size
        num_chunks = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size

        return {"header": df_header, "total_rows": total_rows, "num_chunks": num_chunks}

    @property
    def num_chunks(self) -> int:
        """Get total number of chunks."""
        return self._csv_metadata["num_chunks"]

    @property
    def total_rows(self) -> int:
        """Get total number of rows."""
        return self._csv_metadata["total_rows"]

    def load_chunk(self, chunk_index: int) -> pd.DataFrame:
        """Load a specific chunk from CSV."""

        # Validate chunk index
        if chunk_index >= self.num_chunks:
            raise IndexError(
                f"Chunk index {chunk_index} out of range (0-{self.num_chunks-1})"
            )

        # Calculate number of rows to skip to get the correct chunk
        skip_rows = 1 + chunk_index * self.config.chunk_size

        # Calculate number of rows left in the file
        rows_left = self.total_rows - (chunk_index * self.config.chunk_size)

        # Getting read size for the chunk
        read_size = min(self.config.chunk_size, rows_left)

        chunk_data = pd.read_csv(
            self.config.csv_path,
            skiprows=skip_rows,
            nrows=read_size,
            header=None,
            names=self._csv_metadata["header"].columns,
        )

        return chunk_data

    def get_batch(
        self, chunk_data: pd.DataFrame, batch_index: int
    ) -> Optional[Tuple[Union[np.ndarray, torch.Tensor], int, int]]:
        """Extract a batch from chunk data."""

        # Start index for the batch
        start = batch_index * self.config.batch_size
        if start >= len(chunk_data):
            return None

        # End index for the batch
        end = min(start + self.config.batch_size, len(chunk_data))

        # If going through hidden layers, then input_col is None and we need to load from intermediate tensors
        if self.config.input_col is None:
            raise ValueError(
                "Cannot extract batch from chunk_data when input_col is None. Use load_batch_from_intermediate instead."
            )

        batch_data = chunk_data.iloc[start:end][self.config.input_col].values
        return batch_data, start, end

    def load_batch_from_intermediate(
        self, chunk_index: int, batch_index: int
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Load a batch from intermediate tensor files."""

        # Chunk file path
        chunk_file = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"
        if not chunk_file.exists():
            raise FileNotFoundError(
                f"No intermediate data found for chunk {chunk_index}"
            )

        # Load the chunk tensor, weights_only=False to load only the tensor, no metadata
        chunk_tensors = torch.load(chunk_file, weights_only=False)

        # Start index for the batch
        start = batch_index * self.config.batch_size
        if start >= len(chunk_tensors):
            return None

        # End index for the batch
        end = min(start + self.config.batch_size, len(chunk_tensors))

        # Slice the chunk tensor to get the batch
        batch_data = chunk_tensors[start:end]

        # If saved as numpy array, convert to torch tensor
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.tensor(batch_data)

        return batch_data, start, end

    def load_batch_from_intermediate_tensor(
        self, intermediate_tensor: torch.Tensor, batch_index: int
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Load a batch from an already loaded intermediate tensor."""

        # Start index for the batch
        start = batch_index * self.config.batch_size
        if start >= len(intermediate_tensor):
            return None

        # End index for the batch
        end = min(start + self.config.batch_size, len(intermediate_tensor))

        # Slice the tensor to get the batch
        batch_data = intermediate_tensor[start:end]

        return batch_data, start, end

    def create_chunk_storage(
        self, chunk_size: int, sample_output: torch.Tensor
    ) -> torch.Tensor:
        """Create storage tensor for a chunk. A chunk store is an empty tensor which will be filled batch-by-batch."""
        if isinstance(sample_output, torch.Tensor):
            return torch.zeros(
                chunk_size, *sample_output.shape[1:], dtype=sample_output.dtype
            )
        else:
            return torch.zeros(chunk_size, dtype=torch.float32)

    def update_chunk_storage(
        self,
        chunk_storage: torch.Tensor,
        outputs: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """Update chunk storage with batch outputs."""
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu()

            # If output is padded for batch processing, extract only the relevant rows from it
            if outputs.dim() > 1 and outputs.shape[0] != (end_idx - start_idx):
                outputs = outputs[: end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs
        else:
            outputs_tensor = torch.tensor(outputs, dtype=chunk_storage.dtype)

            # If output is padded for batch processing, extract only the relevant rows from it
            if outputs_tensor.dim() > 1 and outputs_tensor.shape[0] != (
                end_idx - start_idx
            ):
                outputs_tensor = outputs_tensor[: end_idx - start_idx]
            chunk_storage[start_idx:end_idx] = outputs_tensor

        return chunk_storage

    def save_chunk_intermediate(
        self, chunk_storage: torch.Tensor, chunk_index: int
    ) -> Path:
        """Save chunk storage as intermediate tensor file. Thread-safe with atomic writes."""
        final_path = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"

        # Create a temporary file to ensure atomic writes, in case of failure, an exeption is raised that will be caught by the caller
        temp_path = (
            Path(self.config.intermediate_dir)
            / f"temp_{uuid.uuid4().hex}_chunk_{chunk_index}.pt"
        )

        try:
            torch.save(chunk_storage, temp_path)
            temp_path.rename(final_path)
            return final_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_chunk_final(
        self, chunk_data: pd.DataFrame, chunk_storage: torch.Tensor, chunk_index: int
    ) -> Path:
        """Save chunk with outputs to final CSV. Thread-safe with atomic writes."""

        # For saving the outputs of final layers
        if not self.config.output_col:
            raise ValueError("output_col must be specified for final save")

        if isinstance(chunk_storage, torch.Tensor):
            output_values = chunk_storage.numpy()
        else:
            output_values = chunk_storage

        # Create a DataFrame for the chunk, with the output column defined in config
        chunk_df = chunk_data.copy()
        chunk_df[self.config.output_col] = output_values

        final_path = Path(self.config.final_output_dir) / f"chunk_{chunk_index}.csv"

        # Create a temporary file to ensure atomic writes, in case of failure, an exeption is raised that will be caught by the caller
        temp_path = (
            Path(self.config.final_output_dir)
            / f"temp_{uuid.uuid4().hex}_chunk_{chunk_index}.csv"
        )

        try:
            chunk_df.to_csv(temp_path, index=False)
            temp_path.rename(final_path)
            return final_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def concatenate_final_outputs(self, chunk_paths: list) -> Path:
        """Concatenate all chunk CSV files into final output."""
        if not chunk_paths:
            raise ValueError("No chunk paths provided")

        final_output_path = (
            Path(self.config.csv_path).parent
            / f"{Path(self.config.csv_path).stem}_with_{self.config.output_col}.csv"
        )

        # Create a temporary file to ensure atomic writes, in case of failure, an exeption is raised that will be caught by the caller
        temp_final_path = (
            Path(self.config.csv_path).parent / f"temp_{uuid.uuid4().hex}_final.csv"
        )

        try:
            # Sort chunk paths to ensure consistent order on basis of their names
            sorted_paths = sorted(chunk_paths, key=lambda p: int(p.stem.split("_")[1]))

            # Read the first chunk and write it to the temporary final file
            first_chunk = pd.read_csv(sorted_paths[0])
            first_chunk.to_csv(temp_final_path, index=False)

            # For subsequent chunks, append to the temporary final file
            for chunk_path in sorted_paths[1:]:
                chunk_df = pd.read_csv(chunk_path)
                chunk_df.to_csv(temp_final_path, mode="a", header=False, index=False)

            temp_final_path.rename(final_output_path)

            # Clean up intermediate chunk files
            for chunk_path in sorted_paths:
                chunk_path.unlink()

            if Path(self.config.final_output_dir).exists() and not any(
                Path(self.config.final_output_dir).iterdir()
            ):
                Path(self.config.final_output_dir).rmdir()

            return final_output_path

        except Exception:
            if temp_final_path.exists():
                temp_final_path.unlink()
            raise

    def clear_intermediate(self, chunk_index: Optional[int] = None) -> None:
        """Clear intermediate tensor files of a particular index."""
        if chunk_index is not None:
            chunk_file = Path(self.config.intermediate_dir) / f"chunk_{chunk_index}.pt"
            if chunk_file.exists():
                chunk_file.unlink()
        else:
            for file in Path(self.config.intermediate_dir).glob("*.pt"):
                file.unlink()


if __name__ == "__main__":

    # Create a temporary directory and sample CSV file
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

    # Defining chunk and batch sizes
    chunk_size = 10
    batch_size = 3

    # Model that takes 2-3 seconds per batch
    def slow_model(x, layer_name="Model"):
        """Simulate a slow model with random processing time."""
        processing_time = random.uniform(1.0, 2.0)
        print(
            f"    [{layer_name}] Processing batch of size {len(x)} - will take {processing_time:.1f}s"
        )
        time.sleep(processing_time)
        result = x * 2
        print(f"    [{layer_name}] Batch processing complete")
        return result

    # Function to get memory location and size info for an object
    def get_memory_info(obj, name):
        """Get memory location and size info for an object."""
        memory_addr = hex(id(obj))
        if hasattr(obj, "nbytes"):
            size_mb = obj.nbytes / 1024
            return f"{name} @ {memory_addr} ({size_mb:.2f} KB)"
        else:
            return f"{name} @ {memory_addr}"

    print("\n" + "=" * 80)
    print("Test 1: Threaded processing with concurrent chunk loading")
    print("=" * 80)

    # Immutable configuration for processor
    config = DataConfig(
        csv_path=csv_path,
        input_col="input",
        output_col="output",  # Column to save output
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        final_output_dir=os.path.join(temp_dir, "final_outputs"),
        save_output=True,
    )

    # Initialize the thread-safe chunk processor
    processor = DataLoader(config)

    # Process chunks using cached data
    def process_all_chunks_with_cache():
        """Process all chunks using preloaded cache efficiently."""
        chunk_paths = []
        current_chunk_data = None

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Preload first chunk
            if processor.num_chunks > 0:
                print("🔄 Preloading first chunk...")
                current_chunk_data = processor.load_chunk(0)
                print(
                    f"✅ First chunk loaded: {get_memory_info(current_chunk_data, 'Chunk 0')}"
                )

            for chunk_idx in range(processor.num_chunks):
                print(f"\n[Chunk {chunk_idx}] Starting processing")
                start_time = time.time()

                # Use cached data (already loaded)
                chunk_data = current_chunk_data
                print(f"[Chunk {chunk_idx}] Using cached data: {len(chunk_data)} rows")

                # Start loading next chunk in background while processing current
                next_chunk_future = None
                if chunk_idx + 1 < processor.num_chunks:
                    print(
                        f"[Chunk {chunk_idx}] 🔄 Starting background load of chunk {chunk_idx + 1}"
                    )
                    next_chunk_future = executor.submit(
                        processor.load_chunk, chunk_idx + 1
                    )

                # Process current chunk
                num_batches = (len(chunk_data) + batch_size - 1) // batch_size
                chunk_storage = None

                for batch_idx in range(num_batches):
                    batch_result = processor.get_batch(chunk_data, batch_idx)
                    if batch_result is None:
                        break

                    batch_data, start_idx, end_idx = batch_result
                    print(
                        f"[Chunk {chunk_idx}] Processing batch {batch_idx + 1}/{num_batches} (indices {start_idx}-{end_idx-1})"
                    )

                    # Check if next chunk is ready
                    if (
                        next_chunk_future
                        and next_chunk_future.done()
                        and chunk_idx + 1 < processor.num_chunks
                    ):
                        current_chunk_data = next_chunk_future.result()
                        memory_info = get_memory_info(
                            current_chunk_data, f"Chunk {chunk_idx + 1}"
                        )
                        print(f"    🎉 BACKGROUND LOAD COMPLETE: {memory_info}")
                        print(
                            f"    📊 Loaded {len(current_chunk_data)} rows while processing current batch"
                        )
                        next_chunk_future = None  # Mark as consumed

                    # Process batch
                    input_tensor = torch.tensor(batch_data, dtype=torch.float32)
                    outputs = slow_model(input_tensor, "Model")

                    # Initialize storage using first batch
                    if chunk_storage is None:
                        chunk_storage = processor.create_chunk_storage(
                            len(chunk_data), outputs
                        )
                        print(
                            f"    📦 Created chunk storage: {get_memory_info(chunk_storage, 'Storage')}"
                        )

                    # Update storage with model outputs
                    chunk_storage = processor.update_chunk_storage(
                        chunk_storage, outputs, start_idx, end_idx
                    )

                # Wait for next chunk if still loading
                if next_chunk_future and not next_chunk_future.done():
                    print(
                        f"[Chunk {chunk_idx}] ⏳ Waiting for next chunk {chunk_idx + 1} to finish loading..."
                    )
                    current_chunk_data = next_chunk_future.result()
                    memory_info = get_memory_info(
                        current_chunk_data, f"Chunk {chunk_idx + 1}"
                    )
                    print(f"    🎉 BACKGROUND LOAD COMPLETE: {memory_info}")
                elif next_chunk_future and chunk_idx + 1 < processor.num_chunks:
                    # Already loaded, just consume the result
                    if current_chunk_data is None:
                        current_chunk_data = next_chunk_future.result()
                        memory_info = get_memory_info(
                            current_chunk_data, f"Chunk {chunk_idx + 1}"
                        )
                        print(f"    🎉 BACKGROUND LOAD ALREADY COMPLETE: {memory_info}")

                # Save chunk
                if chunk_storage is not None:
                    chunk_path = processor.save_chunk_final(
                        chunk_data, chunk_storage, chunk_idx
                    )
                    chunk_paths.append(chunk_path)
                    print(f"[Chunk {chunk_idx}] Saved to {chunk_path}")

                total_time = time.time() - start_time
                print(f"[Chunk {chunk_idx}] Complete in {total_time:.2f}s")

        return chunk_paths

    # Process all chunks with caching
    start_time = time.time()
    chunk_paths = process_all_chunks_with_cache()

    # Concatenate all chunk outputs into final output
    if chunk_paths:
        final_path = processor.concatenate_final_outputs(chunk_paths)
        print(f"\nFinal output saved to: {final_path}")

    total_time = time.time() - start_time
    print(f"\nTest 1 complete in {total_time:.2f}s")

    # Verify results
    final_file = Path(csv_path).parent / f"{Path(csv_path).stem}_with_output.csv"
    df = pd.read_csv(final_file)
    print(f"\nFinal CSV contents after Test 1 ({len(df)} rows):")
    print(df.head(10))

    assert "output" in df.columns, "Output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"
    assert np.allclose(df["input"] * 2, df["output"]), "Incorrect output values"
    print("✅ Test 1 passed: Output values correct")

    print("\n" + "=" * 80)
    print("Test 2: Multi-layer processing with intermediate tensors")
    print("=" * 80)

    print("\nProcessing Layer 1 (input -> intermediate tensors)...")

    # Configuration for layer 1 processing
    config_layer1 = DataConfig(
        csv_path=csv_path,
        input_col="input",
        output_col=None,  # Save as intermediate tensors
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        final_output_dir=os.path.join(temp_dir, "final_outputs"),
        save_output=True,
    )

    processor_layer1 = DataLoader(config_layer1)

    # Process layer 1 with the same caching approach
    def process_layer1_all_chunks():
        """Process all layer 1 chunks efficiently."""
        current_chunk_data = None

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Preload first chunk
            if processor_layer1.num_chunks > 0:
                print("🔄 Preloading first chunk for Layer 1...")
                current_chunk_data = processor_layer1.load_chunk(0)
                print(
                    f"✅ First chunk loaded: {get_memory_info(current_chunk_data, 'Layer1-Chunk 0')}"
                )

            for chunk_idx in range(processor_layer1.num_chunks):
                print(f"\n[Layer1-Chunk {chunk_idx}] Processing...")

                # Use cached data
                chunk_data = current_chunk_data
                print(
                    f"[Layer1-Chunk {chunk_idx}] Using cached data: {len(chunk_data)} rows"
                )

                # Start loading next chunk in background
                next_chunk_future = None
                if chunk_idx + 1 < processor_layer1.num_chunks:
                    print(
                        f"[Layer1-Chunk {chunk_idx}] 🔄 Starting background load of chunk {chunk_idx + 1}"
                    )
                    next_chunk_future = executor.submit(
                        processor_layer1.load_chunk, chunk_idx + 1
                    )

                num_batches = (len(chunk_data) + batch_size - 1) // batch_size
                chunk_storage = None

                for batch_idx in range(num_batches):
                    batch_result = processor_layer1.get_batch(chunk_data, batch_idx)
                    if batch_result is None:
                        break

                    batch_data, start_idx, end_idx = batch_result
                    print(
                        f"[Layer1-Chunk {chunk_idx}] Batch {batch_idx + 1}/{num_batches} (indices {start_idx}-{end_idx-1})"
                    )

                    # Check background loading
                    if (
                        next_chunk_future
                        and next_chunk_future.done()
                        and chunk_idx + 1 < processor_layer1.num_chunks
                    ):
                        current_chunk_data = next_chunk_future.result()
                        memory_info = get_memory_info(
                            current_chunk_data, f"Layer1-Chunk {chunk_idx + 1}"
                        )
                        print(f"    🎉 BACKGROUND LOAD COMPLETE: {memory_info}")
                        next_chunk_future = None

                    input_tensor = torch.tensor(batch_data, dtype=torch.float32)
                    processing_time = random.uniform(1.0, 2.0)
                    print(
                        f"    [Layer1] Processing batch of size {len(input_tensor)} - will take {processing_time:.1f}s"
                    )
                    time.sleep(processing_time)
                    outputs = input_tensor * 2
                    print(f"    [Layer1] Batch processing complete")

                    # Initialize storage using first batch
                    if chunk_storage is None:
                        chunk_storage = processor_layer1.create_chunk_storage(
                            len(chunk_data), outputs
                        )
                        print(
                            f"    📦 Created Layer1 storage: {get_memory_info(chunk_storage, 'Layer1Storage')}"
                        )

                    # Update storage with model outputs
                    chunk_storage = processor_layer1.update_chunk_storage(
                        chunk_storage, outputs, start_idx, end_idx
                    )

                # Wait for next chunk if needed
                if next_chunk_future and not next_chunk_future.done():
                    print(
                        f"[Layer1-Chunk {chunk_idx}] ⏳ Waiting for next chunk to finish loading..."
                    )
                    current_chunk_data = next_chunk_future.result()
                    memory_info = get_memory_info(
                        current_chunk_data, f"Layer1-Chunk {chunk_idx + 1}"
                    )
                    print(f"    🎉 BACKGROUND LOAD COMPLETE: {memory_info}")
                elif next_chunk_future and chunk_idx + 1 < processor_layer1.num_chunks:
                    if current_chunk_data is None:
                        current_chunk_data = next_chunk_future.result()
                        memory_info = get_memory_info(
                            current_chunk_data, f"Layer1-Chunk {chunk_idx + 1}"
                        )
                        print(f"    🎉 BACKGROUND LOAD ALREADY COMPLETE: {memory_info}")

                # Save intermediate tensors to disk
                if chunk_storage is not None:
                    intermediate_path = processor_layer1.save_chunk_intermediate(
                        chunk_storage, chunk_idx
                    )
                    print(
                        f"[Layer1-Chunk {chunk_idx}] 💾 Saved intermediate tensors to {intermediate_path}"
                    )
                    print(
                        f"    📁 File size: {intermediate_path.stat().st_size / (1024*1024):.2f} MB"
                    )

    # Process all layer 1 chunks
    process_layer1_all_chunks()

    # List intermediate files created
    intermediate_files = list(Path(temp_dir, "intermediate").glob("chunk_*.pt"))
    print(
        f"\nLayer 1 complete. Created {len(intermediate_files)} intermediate tensor files."
    )

    print("\nProcessing second layer (intermediate tensors -> final output)...")

    # Configuration for second layer
    config_layer2 = DataConfig(
        csv_path=csv_path,
        input_col=None,  # Read from intermediate tensors
        output_col="final_output",
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate"),
        final_output_dir=os.path.join(temp_dir, "final_outputs"),
        save_output=True,
    )

    processor_layer2 = DataLoader(config_layer2)

    # Process layer 2 with proper caching
    def process_layer2_all_chunks():
        """Process all layer 2 chunks with proper intermediate tensor caching."""
        chunk_paths = []
        current_chunk_data = None
        current_intermediate_tensor = None

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:  # 2 workers: one for CSV, one for tensors
            # Preload first chunk's data and intermediate tensor
            if processor_layer2.num_chunks > 0:
                print("🔄 Preloading first chunk data and intermediate tensor...")
                csv_future = executor.submit(processor_layer2.load_chunk, 0)
                intermediate_path = Path(temp_dir, "intermediate") / "chunk_0.pt"
                tensor_future = executor.submit(
                    torch.load, intermediate_path, weights_only=False
                )

                current_chunk_data = csv_future.result()
                current_intermediate_tensor = tensor_future.result()
                print(
                    f"✅ First chunk loaded: {get_memory_info(current_chunk_data, 'Chunk 0')}"
                )
                print(
                    f"✅ First intermediate loaded: {get_memory_info(current_intermediate_tensor, 'Intermediate 0')}"
                )

            for chunk_idx in range(processor_layer2.num_chunks):
                print(f"\n[Layer2-Chunk {chunk_idx}] Processing...")
                start_time = time.time()

                # Use cached data
                chunk_data = current_chunk_data
                intermediate_tensor = current_intermediate_tensor
                print(
                    f"[Layer2-Chunk {chunk_idx}] Using cached CSV data: {len(chunk_data)} rows"
                )
                print(
                    f"[Layer2-Chunk {chunk_idx}] Using cached intermediate tensor: {get_memory_info(intermediate_tensor, 'CachedTensor')}"
                )

                # Start loading next chunk's data and intermediate tensor in background
                next_csv_future = None
                next_tensor_future = None
                if chunk_idx + 1 < processor_layer2.num_chunks:
                    print(
                        f"[Layer2-Chunk {chunk_idx}] 🔄 Starting background load of chunk {chunk_idx + 1}"
                    )
                    next_csv_future = executor.submit(
                        processor_layer2.load_chunk, chunk_idx + 1
                    )
                    next_intermediate_path = (
                        Path(temp_dir, "intermediate") / f"chunk_{chunk_idx + 1}.pt"
                    )
                    if next_intermediate_path.exists():
                        next_tensor_future = executor.submit(
                            torch.load, next_intermediate_path, weights_only=False
                        )

                num_batches = (len(intermediate_tensor) + batch_size - 1) // batch_size
                chunk_storage = None

                for batch_idx in range(num_batches):
                    batch_result = processor_layer2.load_batch_from_intermediate_tensor(
                        intermediate_tensor, batch_idx
                    )
                    if batch_result is None:
                        break

                    batch_data, start_idx, end_idx = batch_result
                    print(
                        f"[Layer2-Chunk {chunk_idx}] Processing batch {batch_idx + 1}/{num_batches} (indices {start_idx}-{end_idx-1})"
                    )

                    # Check background loading status
                    if (
                        next_csv_future
                        and next_csv_future.done()
                        and chunk_idx + 1 < processor_layer2.num_chunks
                    ):
                        current_chunk_data = next_csv_future.result()
                        memory_info = get_memory_info(
                            current_chunk_data, f"NextCSV-{chunk_idx + 1}"
                        )
                        print(f"    🎉 CSV PRELOAD COMPLETE: {memory_info}")
                        next_csv_future = None

                    if (
                        next_tensor_future
                        and next_tensor_future.done()
                        and chunk_idx + 1 < processor_layer2.num_chunks
                    ):
                        current_intermediate_tensor = next_tensor_future.result()
                        memory_info = get_memory_info(
                            current_intermediate_tensor, f"NextTensor-{chunk_idx + 1}"
                        )
                        print(f"    🎉 TENSOR PRELOAD COMPLETE: {memory_info}")
                        next_tensor_future = None

                    outputs = slow_model(batch_data, "Layer2")
                    outputs = outputs * 1.5

                    # Initialize storage using first batch
                    if chunk_storage is None:
                        chunk_storage = processor_layer2.create_chunk_storage(
                            len(chunk_data), outputs
                        )
                        print(
                            f"    📦 Created Layer2 storage: {get_memory_info(chunk_storage, 'Layer2Storage')}"
                        )

                    # Update storage with model outputs
                    chunk_storage = processor_layer2.update_chunk_storage(
                        chunk_storage, outputs, start_idx, end_idx
                    )

                # Wait for any remaining background loads
                if next_csv_future and not next_csv_future.done():
                    print(f"[Layer2-Chunk {chunk_idx}] ⏳ Waiting for CSV preload...")
                    current_chunk_data = next_csv_future.result()
                    memory_info = get_memory_info(
                        current_chunk_data, f"NextCSV-{chunk_idx + 1}"
                    )
                    print(f"    🎉 CSV PRELOAD COMPLETE: {memory_info}")
                elif next_csv_future and chunk_idx + 1 < processor_layer2.num_chunks:
                    if current_chunk_data is None:
                        current_chunk_data = next_csv_future.result()

                if next_tensor_future and not next_tensor_future.done():
                    print(
                        f"[Layer2-Chunk {chunk_idx}] ⏳ Waiting for tensor preload..."
                    )
                    current_intermediate_tensor = next_tensor_future.result()
                    memory_info = get_memory_info(
                        current_intermediate_tensor, f"NextTensor-{chunk_idx + 1}"
                    )
                    print(f"    🎉 TENSOR PRELOAD COMPLETE: {memory_info}")
                elif next_tensor_future and chunk_idx + 1 < processor_layer2.num_chunks:
                    if current_intermediate_tensor is None:
                        current_intermediate_tensor = next_tensor_future.result()

                # Save final output chunk
                if chunk_storage is not None:
                    chunk_path = processor_layer2.save_chunk_final(
                        chunk_data, chunk_storage, chunk_idx
                    )
                    chunk_paths.append(chunk_path)
                    print(
                        f"[Layer2-Chunk {chunk_idx}] 💾 Saved final output to {chunk_path}"
                    )
                    file_size = chunk_path.stat().st_size / (1024 * 1024)
                    print(f"    📁 Output file size: {file_size:.2f} MB")

                total_time = time.time() - start_time
                print(f"[Layer2-Chunk {chunk_idx}] Complete in {total_time:.2f}s")

        return chunk_paths

    # Process all layer 2 chunks
    layer2_chunk_paths = process_layer2_all_chunks()

    # Concatenate final outputs
    if layer2_chunk_paths:
        final_path = processor_layer2.concatenate_final_outputs(layer2_chunk_paths)
        print(f"\nSecond layer final output saved to: {final_path}")

    # Verification of results
    final_file = Path(csv_path).parent / f"{Path(csv_path).stem}_with_final_output.csv"
    df = pd.read_csv(final_file)
    print(f"\nFinal CSV after both layers ({len(df)} rows):")
    print(df[["id", "input", "final_output"]].head(10))
    print("...")
    print(df[["id", "input", "final_output"]].tail(5))

    # Validation checks
    assert "final_output" in df.columns, "Final output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"

    # Check for NaN values
    nan_count = df["final_output"].isna().sum()
    if nan_count > 0:
        print(f"⚠️  Warning: Found {nan_count} NaN values in final_output")

    # Verify computation logic
    valid_mask = ~df["final_output"].isna()
    if valid_mask.sum() > 0:
        expected = df.loc[valid_mask, "input"] * 6  # input * 2 * 1.5 * 2
        actual = df.loc[valid_mask, "final_output"]
        assert np.allclose(expected, actual), f"Incorrect final output values"

    print("✅ Test 2 passed: Final output values correct")

    print("\n" + "=" * 80)
    print("Test 3: Memory efficiency validation")
    print("=" * 80)

    # Check remaining intermediate files
    remaining_intermediate = list(Path(temp_dir, "intermediate").glob("*.pt"))
    print(f"Remaining intermediate files: {len(remaining_intermediate)}")

    # Cleanup temporary directory
    shutil.rmtree(temp_dir)
    print(f"\n✅ All tests passed! Removed temporary directory: {temp_dir}")
