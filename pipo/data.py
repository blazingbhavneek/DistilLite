import pandas as pd
import numpy as np
import os
import tempfile
import torch
import shutil
from pathlib import Path

class ChunkedCSVProcessor:
    def __init__(self, csv_path, input_col=None, output_col=None, 
                 chunk_size=10000, batch_size=256, intermediate_dir="intermediate_tensors"):
        self.csv_path = csv_path
        self.input_col = input_col
        self.output_col = output_col
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.intermediate_dir = Path(intermediate_dir)
        
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_chunk = 0
        self.current_batch = 0
        self.processed_rows = 0
        self.total_rows = 0
        self.num_chunks = 0
        self.using_intermediate = False
        
        self.chunk_data = None
        self.tensor_paths = []
        self.temp_csv_path = None
        
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV structure and counters."""
        self.df_header = pd.read_csv(self.csv_path, nrows=0)
        
        with open(self.csv_path, 'r') as f:
            self.total_rows = sum(1 for _ in f) - 1
        
        self.num_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size
        
        self.using_intermediate = self.input_col is None
        
        if self.using_intermediate and 'tensor_path' not in self.df_header.columns:
            self.df_header['tensor_path'] = None
            self.df_header.to_csv(self.csv_path, index=False)
        
        self.temp_csv_path = str(Path(self.csv_path).with_suffix('.tmp'))
    
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
            names=self.df_header.columns
        )
        
        if self.using_intermediate:
            self.tensor_paths = self.chunk_data['tensor_path'].tolist()
        
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
        
        if self.using_intermediate:
            batch_paths = self.tensor_paths[start:end]
            batch_data = [torch.load(self.intermediate_dir / path) for path in batch_paths]
        else:
            batch_data = self.chunk_data.iloc[start:end][self.input_col].values
        
        self.current_batch += 1
        
        return batch_data, start, end
    
    def save_batch_output(self, outputs, start_idx, end_idx):
        """Save model outputs for current batch."""
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        
        if self.output_col is not None:
            if self.output_col not in self.chunk_data.columns:
                self.chunk_data[self.output_col] = None
            
            self.chunk_data.loc[start_idx:end_idx-1, self.output_col] = outputs
        else:
            batch_paths = []
            for i, output in enumerate(outputs):
                global_idx = self.current_chunk * self.chunk_size + start_idx + i
                tensor_path = f"row_{global_idx}.pt"
                full_path = self.intermediate_dir / tensor_path
                
                torch.save(torch.tensor(output), full_path)
                batch_paths.append(tensor_path)
            
            if 'tensor_path' not in self.chunk_data.columns:
                self.chunk_data['tensor_path'] = None
            self.chunk_data.loc[start_idx:end_idx-1, 'tensor_path'] = batch_paths
    
    def save_chunk_to_disk(self):
        """Save current chunk back to CSV using safe file operations."""
        if self.chunk_data is None:
            return
            
        if self.current_chunk == 0:
            original_df = pd.read_csv(self.csv_path)
            
            for col in self.chunk_data.columns:
                if col not in original_df.columns:
                    original_df[col] = None
            
            start_idx = self.current_chunk * self.chunk_size
            end_idx = min(start_idx + len(self.chunk_data), len(original_df))
            
            for col in self.chunk_data.columns:
                values = self.chunk_data[col].values
                if original_df[col].dtype != values.dtype:
                    original_df[col] = original_df[col].astype(values.dtype)
                original_df.loc[start_idx:end_idx-1, col] = values
            
            original_df.to_csv(self.temp_csv_path, index=False)
        else:
            existing_df = pd.read_csv(self.temp_csv_path)
            
            for col in self.chunk_data.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            
            start_idx = self.current_chunk * self.chunk_size
            end_idx = min(start_idx + len(self.chunk_data), len(existing_df))
            
            for col in self.chunk_data.columns:
                values = self.chunk_data[col].values
                if existing_df[col].dtype != values.dtype:
                    existing_df[col] = existing_df[col].astype(values.dtype)
                existing_df.loc[start_idx:end_idx-1, col] = values
            
            existing_df.to_csv(self.temp_csv_path, index=False)
        
        self.chunk_data = None
        self.tensor_paths = []
        self.current_chunk += 1
    
    def finalize(self):
        """Finalize processing by moving temp file to final location."""
        if self.temp_csv_path and os.path.exists(self.temp_csv_path):
            shutil.move(self.temp_csv_path, self.csv_path)

    def clear_intermediate(self):
        """Clear all intermediate tensor files."""
        for file in self.intermediate_dir.glob('*.pt'):
            file.unlink()

if __name__ == "__main__":
    print("Testing ChunkedCSVProcessor with simulated model operations")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    csv_path = os.path.join(temp_dir, "data.csv")
    num_samples = 42
    data = {
        'id': range(1, num_samples+1),
        'input': [float(i) for i in range(1, num_samples+1)]
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Created sample CSV with {num_samples} rows at {csv_path}")
    
    chunk_size = 10
    batch_size = 3
    
    print("\nTest 1: Simple end-to-end processing")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col='input',
        output_col='output',
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate")
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
    
    df = pd.read_csv(csv_path)
    print(f"\nFinal CSV contents after Test 1 ({len(df)} rows):")
    print(df.head(10))
    
    assert 'output' in df.columns, "Output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"
    assert np.allclose(df['input'] * 2, df['output']), "Incorrect output values"
    print("Test 1 passed: Output values correct")
    
    print("\nTest 2: Multi-layer processing")
    
    print("Processing Layer 1...")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col='input',
        output_col=None,
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate")
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
    
    df = pd.read_csv(csv_path)
    assert 'tensor_path' in df.columns, "Tensor path column missing"
    print(f"\nLayer 1 complete. Tensor paths added to CSV ({len(df)} rows).")
    print(f"Number of tensor paths: {len(df['tensor_path'].dropna())}")
    
    print("\nProcessing Layer 2...")
    processor = ChunkedCSVProcessor(
        csv_path=csv_path,
        input_col=None,
        output_col='final_output',
        chunk_size=chunk_size,
        batch_size=batch_size,
        intermediate_dir=os.path.join(temp_dir, "intermediate")
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
            
            if len(batch_data) > 1:
                input_tensor = torch.stack(batch_data)
            else:
                input_tensor = batch_data[0].unsqueeze(0)
                
            outputs = input_tensor * 3
            processor.save_batch_output(outputs, start_idx, end_idx)
        processor.save_chunk_to_disk()
        print(f"Saved chunk {chunks_processed} to disk")
    
    processor.finalize()
    
    df = pd.read_csv(csv_path)
    print(f"\nFinal CSV after both layers ({len(df)} rows):")
    print(df[['id', 'input', 'final_output']].head(10))
    print("\n...")
    print(df[['id', 'input', 'final_output']].tail(10))
    
    assert 'final_output' in df.columns, "Final output column missing"
    assert len(df) == num_samples, f"Expected {num_samples} rows, got {len(df)}"
    assert np.allclose(df['input'] * 6, df['final_output']), "Incorrect final output values"
    print("\nTest 2 passed: Final output values correct")
    
    print("\nTest 3: Intermediate file validation")
    
    tensor_files = list(os.listdir(os.path.join(temp_dir, "intermediate")))
    print(f"Found {len(tensor_files)} intermediate tensor files")
    
    assert len(tensor_files) == num_samples, f"Expected {num_samples} tensor files, found {len(tensor_files)}"
    
    shutil.rmtree(temp_dir)
    print(f"\nAll tests passed! Removed temporary directory: {temp_dir}")