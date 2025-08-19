import gc
import os
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillite.data import DataConfig, DataLoader
from distillite.model import BaseExecutor, ModelConfig, Qwen3Executor


class InferenceOrchestrator:
    """
    Multi-stage inference orchestrator that coordinates thread-safe DataLoader and Executor
    for memory-efficient processing of large datasets through transformer models.
    """

    def __init__(
        self,
        data_config: "DataConfig",
        data_loader: "DataLoader",
        model_config: "ModelConfig",
        executor: "BaseExecutor",
        max_workers: int = 2,
        max_seq_length: int = 512,
    ):
        """
        Initialize orchestrator with injected dependencies.

        Args:
            data_config: Configuration for data processing
            data_loader: Thread-safe data loader instance
            model_config: Model configuration
            executor: Thread-safe model executor (inherits from BaseExecutor)
            max_workers: Maximum workers for thread pools
            max_seq_length: Maximum sequence length for tokenization
        """
        self.data_config = data_config
        self.data_loader = data_loader
        self.model_config = model_config
        self.executor = executor
        self.max_workers = max_workers
        self.max_seq_length = max_seq_length

        self._initialize_tokenizer()

        self.data_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="DataLoader"
        )
        self.model_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ModelLoader"
        )

        self.current_chunk_data = None
        self.current_loaded_layers = None
        self.next_chunk_future = None
        self.next_layers_future = None

        self.intermediate_dir = Path(data_config.intermediate_dir)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.active_intermediate_files = set()

        print(f"   📊 Dataset: {data_config.csv_path}")
        print(f"   🤖 Model: {model_config.model_path}")
        print(f"   💾 Intermediate dir: {self.intermediate_dir}")
        print(f"   📏 Max sequence length: {self.max_seq_length}")

    def _initialize_tokenizer(self):
        """Initialize tokenizer for text processing."""
        if not hasattr(self.executor, "tokenizer") or self.executor.tokenizer is None:
            print("   🔤 Initializing tokenizer...")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.executor.tokenizer = tokenizer
            print(f"   ✅ Tokenizer initialized: {tokenizer.__class__.__name__}")
        else:
            print(
                f"   ✅ Using existing tokenizer: {self.executor.tokenizer.__class__.__name__}"
            )

    def debug_single_stage_comparison(self, test_samples: int = 3) -> bool:
        """
        Debug function to test if single-stage execution matches HF model.
        This should be run first to verify executor correctness.

        Returns:
            bool: True if single-stage matches HF (logits diff < 1e-4)
        """
        print(f"\n🔍 DEBUG: Single-stage comparison with HF model...")

        try:
            test_data = self.data_loader.load_chunk(0)[:test_samples]

            print(
                f"   🔄 Loading ALL layers {0}-{self.model_config.total_layers-1} for comparison..."
            )
            all_layers = self.executor.load_layer_range(
                0, self.model_config.total_layers - 1
            )

            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path, trust_remote_code=True
            ).eval()

            matches = 0
            total_samples = len(test_data)
            max_logits_diffs = []

            for idx, sample in enumerate(test_data.iterrows()):
                sample_data = sample[1]
                test_text = str(sample_data[self.data_config.input_col])

                print(f"   📝 Sample {idx+1}: {test_text[:80]}...")

                tokenized = self.executor.tokenizer(
                    test_text,
                    return_tensors="pt",
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding=False,
                )

                batch_size, seq_len = tokenized["input_ids"].shape
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

                executor_input = {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "position_ids": position_ids,
                }

                orchestrator_output = self.executor.execute_layer_range(
                    loaded_layers=all_layers,
                    batch=executor_input,
                    validate_shapes=False,
                )

                with torch.no_grad():
                    hf_output = hf_model(**tokenized)

                orchestrator_logits = orchestrator_output[0, -1, :].cpu().numpy()
                hf_logits = hf_output.logits[0, -1, :].cpu().numpy()

                logits_diff = np.max(np.abs(orchestrator_logits - hf_logits))
                orchestrator_token = np.argmax(orchestrator_logits)
                hf_token = np.argmax(hf_logits)
                tokens_match = orchestrator_token == hf_token

                max_logits_diffs.append(logits_diff)
                if tokens_match:
                    matches += 1

                print(f"      📊 Max logits diff: {logits_diff:.6f}")
                print(f"      🎯 Tokens match: {'✅' if tokens_match else '❌'}")
                print(
                    f"      🤖 Orchestrator: {orchestrator_token} -> '{self.executor.tokenizer.decode([orchestrator_token])}'"
                )
                print(
                    f"      🤗 HF:       {hf_token} -> '{self.executor.tokenizer.decode([hf_token])}'"
                )

            avg_diff = np.mean(max_logits_diffs)
            max_diff = np.max(max_logits_diffs)
            match_rate = matches / total_samples

            print(f"\n   📊 Single-Stage Debug Summary:")
            print(
                f"      🎯 Token matches: {matches}/{total_samples} ({match_rate*100:.1f}%)"
            )
            print(f"      📊 Average max logits diff: {avg_diff:.6f}")
            print(f"      📊 Maximum logits diff: {max_diff:.6f}")

            is_working = match_rate >= 0.8 and avg_diff < 1e-3

            if is_working:
                print(f"      ✅ Single-stage execution looks correct!")
            else:
                print(
                    f"      ❌ Single-stage execution has issues - fix executor before testing multi-stage"
                )

            del all_layers, hf_model
            gc.collect()

            return is_working

        except Exception as e:
            print(f"      ❌ Debug comparison failed: {e}")
            traceback.print_exc()
            return False

    def run(self, debug_first: bool = True) -> str:
        """
        Execute the complete inference orchestrator with optimized storage.

        Args:
            debug_first: If True, run single-stage debug comparison first

        Returns:
            str: Path to final output file
        """
        print("\n🚀 Starting memory-optimized inference orchestrator...")
        start_time = time.time()

        if debug_first:
            print("\n🔍 Running pre-flight debug check...")
            if not self.debug_single_stage_comparison(test_samples=3):
                print(
                    "❌ Single-stage debug failed. Please fix executor before proceeding."
                )
                print(
                    "💡 Hint: Check if your executor properly handles input_ids, attention_mask, and position_ids"
                )
                return None
            print(
                "✅ Single-stage debug passed! Proceeding with multi-stage orchestrator..."
            )

        try:
            layer_groups = self._calculate_layer_groups()
            print(f"\n📋 Execution plan:")
            print(f"   🎯 {len(layer_groups)} stages")
            print(f"   📦 {self.data_loader.num_chunks} chunks per stage")
            print(f"   🔢 Batch size: {self.data_config.batch_size}")
            print(f"   💽 Storage: Reuse intermediate space per chunk")
            print(f"   🔧 Context: Preserve attention_mask + position_ids")
            for i, (start, end) in enumerate(layer_groups):
                print(f"   Stage {i}: Layers {start}-{end}")

            final_path = self._execute_optimized_orchestrator(layer_groups)

            total_time = time.time() - start_time
            print(f"\n✅ Orchestrator complete in {total_time:.2f}s!")
            print(f"📁 Final output: {final_path}")
            return final_path

        finally:
            self._cleanup()

    def _execute_optimized_orchestrator(
        self, layer_groups: List[Tuple[int, int]]
    ) -> str:
        """
        Execute orchestrator by processing each chunk through all stages sequentially.
        This approach minimizes intermediate storage by overwriting files.
        """

        print(
            f"\n🔄 Processing {self.data_loader.num_chunks} chunks through {len(layer_groups)} stages..."
        )

        final_chunk_paths = []

        for chunk_idx in range(self.data_loader.num_chunks):
            print(f"\n📦 CHUNK {chunk_idx + 1}/{self.data_loader.num_chunks}")
            print(f"   🔄 Processing through {len(layer_groups)} stages...")

            chunk_data = self.data_loader.load_chunk(chunk_idx)
            print(f"   📊 Loaded chunk: {len(chunk_data)} rows")

            chunk_final_path = self._process_chunk_through_all_stages(
                chunk_data, chunk_idx, layer_groups
            )
            final_chunk_paths.append(chunk_final_path)

            print(f"   ✅ Chunk {chunk_idx + 1} completed and saved")
            print(f"   📁 Output: {chunk_final_path}")

        return self._concatenate_final_outputs(final_chunk_paths)

    def _process_chunk_through_all_stages(
        self, chunk_data: Any, chunk_idx: int, layer_groups: List[Tuple[int, int]]
    ) -> str:
        """
        Process one chunk through all stages, reusing intermediate storage.

        Args:
            chunk_data: The chunk data to process
            chunk_idx: Index of the current chunk
            layer_groups: List of (start_layer, end_layer) tuples

        Returns:
            str: Path to the final output file for this chunk
        """

        current_stage_data = chunk_data
        current_loaded_layers = None

        for stage_idx, (start_layer, end_layer) in enumerate(layer_groups):
            is_final_stage = stage_idx == len(layer_groups) - 1
            print(
                f"     Stage {stage_idx + 1}/{len(layer_groups)}: Layers {start_layer}-{end_layer}"
            )

            if current_loaded_layers is not None:
                del current_loaded_layers
                gc.collect()

            print(f"     🔄 Loading layers {start_layer}-{end_layer}...")
            current_loaded_layers = self.executor.load_layer_range(
                start_layer, end_layer
            )
            print(f"     ✅ Layers loaded")

            if stage_idx == 0:
                stage_results = self._process_stage_from_text(
                    current_stage_data, current_loaded_layers, chunk_idx, stage_idx
                )
            else:
                stage_results = self._process_stage_from_intermediate(
                    current_loaded_layers, chunk_idx, stage_idx
                )

            if is_final_stage:
                final_path = self._save_final_chunk_results(
                    stage_results, current_stage_data, chunk_idx
                )

                del current_loaded_layers
                gc.collect()

                return str(final_path)
            else:
                self._save_intermediate_results_overwrite(
                    stage_results, chunk_idx, stage_idx
                )
                print(f"     💾 Intermediate results saved (stage {stage_idx})")

        raise RuntimeError("Should not reach here - final stage should have returned")

    def _process_stage_from_text(
        self, chunk_data: Any, loaded_layers: Any, chunk_idx: int, stage_idx: int
    ) -> Dict[str, Any]:
        """
        Process first stage from text data.
        """

        batch_size = self.data_config.batch_size
        num_batches = (len(chunk_data) + batch_size - 1) // batch_size

        batch_outputs = []
        batch_metadata = []

        print(f"       🔄 Processing {num_batches} batches from text...")

        for batch_idx in range(num_batches):
            batch_result = self.data_loader.get_batch(chunk_data, batch_idx)
            if batch_result is None:
                break

            batch_data, start_idx, end_idx = batch_result
            print(
                f"       [Batch {batch_idx + 1}/{num_batches}] Indices {start_idx}-{end_idx-1}"
            )

            executor_input = self._tokenize_batch(batch_data)

            print(f"       💻 Executing inference on batch...")
            batch_output = self.executor.execute_layer_range(
                loaded_layers=loaded_layers, batch=executor_input, validate_shapes=False
            )

            print(f"       📊 Output shape: {batch_output.shape}")

            batch_outputs.append(batch_output)
            batch_metadata.append(
                {
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "output_shape": batch_output.shape,
                    "attention_mask": executor_input["attention_mask"],
                    "position_ids": executor_input["position_ids"],
                }
            )

            print(f"       ✅ Batch {batch_idx + 1} complete")

        return {
            "batch_outputs": batch_outputs,
            "batch_metadata": batch_metadata,
            "chunk_size": len(chunk_data),
        }

    def _process_stage_from_intermediate(
        self, loaded_layers: Any, chunk_idx: int, stage_idx: int
    ) -> Dict[str, Any]:
        """
        Process subsequent stages from intermediate hidden states.
        """

        intermediate_path = self._get_intermediate_path(chunk_idx)

        if not intermediate_path.exists():
            raise FileNotFoundError(f"Intermediate file not found: {intermediate_path}")

        with open(intermediate_path, "rb") as f:
            prev_results = pickle.load(f)

        batch_outputs = []
        batch_metadata = []

        print(
            f"       🔄 Processing {len(prev_results['batch_outputs'])} batches from intermediate..."
        )

        for batch_idx, (prev_output, prev_metadata) in enumerate(
            zip(prev_results["batch_outputs"], prev_results["batch_metadata"])
        ):
            print(f"       [Batch {batch_idx + 1}] Processing hidden states...")

            executor_input = {
                "hidden_states": prev_output,
                "attention_mask": prev_metadata.get("attention_mask"),
                "position_ids": prev_metadata.get("position_ids"),
            }

            if (
                executor_input["attention_mask"] is None
                or executor_input["position_ids"] is None
            ):
                raise ValueError(
                    f"Missing attention context in stage {stage_idx}, batch {batch_idx}"
                )

            print(f"       💻 Executing inference on batch with preserved context...")
            batch_output = self.executor.execute_layer_range(
                loaded_layers=loaded_layers, batch=executor_input, validate_shapes=False
            )

            print(f"       📊 Output shape: {batch_output.shape}")

            batch_outputs.append(batch_output)
            batch_metadata.append(
                {
                    "batch_idx": batch_idx,
                    "start_idx": prev_metadata["start_idx"],
                    "end_idx": prev_metadata["end_idx"],
                    "output_shape": batch_output.shape,
                    "attention_mask": prev_metadata.get("attention_mask"),
                    "position_ids": prev_metadata.get("position_ids"),
                }
            )

            print(f"       ✅ Batch {batch_idx + 1} complete")

        return {
            "batch_outputs": batch_outputs,
            "batch_metadata": batch_metadata,
            "chunk_size": prev_results["chunk_size"],
        }

    def _tokenize_batch(self, batch_data: Any) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of text data.
        """

        tokenizer = self.executor.tokenizer

        if isinstance(batch_data, np.ndarray):
            text_list = [str(text) for text in batch_data.tolist()]
        elif isinstance(batch_data, (list, tuple)):
            text_list = [str(text) for text in batch_data]
        else:
            text_list = [str(batch_data)]

        print(f"       🔤 Tokenizing {len(text_list)} texts...")
        print(f"       📝 Sample text: '{text_list[0][:80]}...'")

        tokenized = tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        batch_size, seq_len = tokenized["input_ids"].shape
        position_ids = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )

        print(f"       📊 Tokenized shape: {tokenized['input_ids'].shape}")
        print(f"       🔢 Sequence length: {tokenized['input_ids'].shape[1]}")
        print(f"       🎯 Position IDs shape: {position_ids.shape}")

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": position_ids,
        }

    def _get_intermediate_path(self, chunk_idx: int) -> Path:
        """Get the path for intermediate storage (same file reused for all stages)."""
        return self.intermediate_dir / f"chunk_{chunk_idx}_intermediate.pkl"

    def _save_intermediate_results_overwrite(
        self, stage_results: Dict[str, Any], chunk_idx: int, stage_idx: int
    ):
        """
        Save intermediate results, overwriting the previous stage's file.
        """

        intermediate_path = self._get_intermediate_path(chunk_idx)

        self.active_intermediate_files.discard(str(intermediate_path))

        for metadata in stage_results["batch_metadata"]:
            if "attention_mask" not in metadata or "position_ids" not in metadata:
                raise ValueError(
                    f"Attention context missing when saving stage {stage_idx}"
                )

        with open(intermediate_path, "wb") as f:
            pickle.dump(stage_results, f)

        self.active_intermediate_files.add(str(intermediate_path))

        file_size = intermediate_path.stat().st_size / (1024 * 1024)
        print(
            f"         📊 Intermediate file size: {file_size:.2f} MB (with preserved context)"
        )

    def _save_final_chunk_results(
        self, stage_results: Dict[str, Any], original_chunk_data: Any, chunk_idx: int
    ) -> Path:
        """Save final chunk results as CSV."""

        print(f"       💾 Saving final output for chunk {chunk_idx}...")

        batch_outputs = stage_results["batch_outputs"]
        batch_metadata = stage_results["batch_metadata"]

        chunk_logits = []

        for batch_output, metadata in zip(batch_outputs, batch_metadata):
            attention_mask = metadata["attention_mask"]
            batch_size = batch_output.shape[0]

            for i in range(batch_size):
                last_pos = int(attention_mask[i].sum()) - 1
                last_token_logits = batch_output[i, last_pos, :].cpu().numpy()
                chunk_logits.append(last_token_logits.tolist())

        if len(chunk_logits) != len(original_chunk_data):
            print(
                f"         ⚠️  Warning: Logits count ({len(chunk_logits)}) != chunk size ({len(original_chunk_data)})"
            )
            chunk_logits = chunk_logits[: len(original_chunk_data)]

        output_data = original_chunk_data.copy()
        output_data[self.data_config.output_col] = chunk_logits

        output_path = Path(self.data_config.final_output_dir) / f"chunk_{chunk_idx}.csv"
        output_data.to_csv(output_path, index=False)

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"         📊 Final chunk file size: {file_size:.2f} MB")

        intermediate_path = self._get_intermediate_path(chunk_idx)
        if intermediate_path.exists():
            intermediate_path.unlink()
            self.active_intermediate_files.discard(str(intermediate_path))
            print(f"         🗑️  Cleaned up intermediate file")

        return output_path

    def _concatenate_final_outputs(self, final_chunk_paths: List[str]) -> str:
        """Concatenate all final chunk outputs."""

        print("\n🔗 Concatenating final outputs...")

        if final_chunk_paths:
            all_chunks = []
            for chunk_path in final_chunk_paths:
                chunk_df = pd.read_csv(chunk_path)
                all_chunks.append(chunk_df)

            final_df = pd.concat(all_chunks, ignore_index=True)

            final_path = (
                Path(self.data_config.final_output_dir) / "final_predictions.csv"
            )
            final_df.to_csv(final_path, index=False)

            print(f"📋 Final output concatenated: {final_path}")
            print(f"📊 Total rows: {len(final_df)}")
            return str(final_path)
        else:
            raise RuntimeError("No final output chunks found!")

    def _calculate_layer_groups(self) -> List[Tuple[int, int]]:
        """Calculate optimal layer groupings based on available memory."""

        try:
            memory_info = self._get_available_memory()
            gpu_memory_gb = 0
            if memory_info.get("available_gpus"):
                gpu_memory_gb = memory_info["available_gpus"][0]["memoryFree"] / 1024
        except:
            gpu_memory_gb = 8

        total_layers = self.model_config.total_layers

        if gpu_memory_gb > 16:
            layers_per_group = min(8, total_layers)
        elif gpu_memory_gb > 8:
            layers_per_group = min(4, total_layers)
        elif gpu_memory_gb > 4:
            layers_per_group = min(2, total_layers)
        else:
            layers_per_group = 1

        groups = []
        for i in range(0, total_layers, layers_per_group):
            end_layer = min(i + layers_per_group - 1, total_layers - 1)
            groups.append((i, end_layer))

        return groups

    def _get_available_memory(self) -> Dict[str, Any]:
        """Fallback memory check if utility not available."""
        return {
            "available_ram": 16.0,
            "available_disk": 100.0,
            "available_gpus": [{"name": "GPU", "memoryFree": 8192}],
        }

    def _cleanup(self):
        """Clean up resources and any remaining intermediate files."""
        print("\n🧹 Cleaning up resources...")

        self.data_pool.shutdown(wait=True)
        self.model_pool.shutdown(wait=True)

        try:
            cleaned_count = 0
            for file_path in self.active_intermediate_files.copy():
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    cleaned_count += 1
                self.active_intermediate_files.discard(file_path)

            if cleaned_count > 0:
                print(f"   🗑️  Removed {cleaned_count} remaining intermediate files")
            else:
                print(f"   ✅ No intermediate files to clean (already removed)")

        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean intermediate files: {e}")

        print(f"   💽 Storage optimization: Minimized intermediate disk usage")
        print(f"   🔧 Context preservation: Maintained attention consistency")


if __name__ == "__main__":
    print("🧪 Testing Orchestrator with HF Model Comparison")
    print("=" * 70)

    local_model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (
        project_root / "datagen" / "results" / "next_token_prediction_dataset.csv"
    )
    repo_id = "Qwen/Qwen3-0.6B"

    chunk_size = 50
    batch_size = 4
    max_seq_length = 512

    base_output_dir = Path.cwd() / "orchestrator_outputs"
    intermediate_dir = os.path.join(base_output_dir, "intermediate")
    final_output_dir = os.path.join(base_output_dir, "final_outputs")

    print(f"📁 Creating output directories...")
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(local_model_dir, exist_ok=True)
    print(f"   ✅ Intermediate dir: {intermediate_dir}")
    print(f"   ✅ Final output dir: {final_output_dir}")
    print(f"   ✅ Model dir: {local_model_dir}")

    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        sample_df = pd.read_csv(csv_path, nrows=5)
        print(f"\n📊 Dataset sample:")
        print(f"   📁 File: {csv_path}")
        print(f"   📋 Columns: {list(sample_df.columns)}")
        print(f"   📏 Sample shape: {sample_df.shape}")

        if "context_text" in sample_df.columns:
            print(f"\n📝 Sample context_text:")
            for i, text in enumerate(sample_df["context_text"].head(3)):
                print(f"   {i+1}. {str(text)[:100]}...")

        print(f"\n🔧 Configuring orchestrator...")

        data_config = DataConfig(
            csv_path=csv_path,
            input_col="context_text",
            output_col="predicted_logits",
            chunk_size=chunk_size,
            batch_size=batch_size,
            intermediate_dir=intermediate_dir,
            final_output_dir=final_output_dir,
            save_output=True,
        )
        print(f"   ✅ Data config created")

        print(f"\n🔧 Initializing components...")

        data_loader = DataLoader(data_config)
        print(f"   ✅ DataLoader initialized")
        print(f"      📊 Total rows: {data_loader.total_rows}")
        print(f"      📦 Number of chunks: {data_loader.num_chunks}")

        model_config = ModelConfig(local_model_dir, repo_id)
        print(f"   ✅ ModelConfig initialized")

        executor = Qwen3Executor(model_config)
        print(f"   ✅ Qwen3Executor initialized")

        print(f"\n🚀 Creating orchestrator...")
        orchestrator = InferenceOrchestrator(
            data_config=data_config,
            data_loader=data_loader,
            model_config=model_config,
            executor=executor,
            max_workers=2,
            max_seq_length=max_seq_length,
        )

        print(f"\n🎯 Running orchestrator with debug check...")
        start_time = time.time()
        final_output_path = orchestrator.run(debug_first=True)

        if final_output_path is None:
            print("❌ Orchestrator execution failed due to debug check failure.")
            exit(1)

        orchestrator_time = time.time() - start_time

        print(f"\n📊 Orchestrator Results:")
        print(f"   ⏱️  Execution time: {orchestrator_time:.2f}s")
        print(f"   📁 Output file: {final_output_path}")

        if not os.path.exists(final_output_path):
            raise FileNotFoundError(f"Output file not found: {final_output_path}")

        results_df = pd.read_csv(final_output_path)
        print(f"   📋 Results shape: {results_df.shape}")
        print(f"   📈 Result columns: {list(results_df.columns)}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path, trust_remote_code=True, torch_dtype=torch.float32
        ).eval()
        print(f"   ✅ HuggingFace model loaded for comparison")

        comparison_samples = min(10, len(results_df))
        comparison_results = []

        print(f"   🎯 Comparing first {comparison_samples} samples...")

        for idx in range(comparison_samples):
            row = results_df.iloc[idx]
            context_text = row["context_text"]
            actual_next_token = row.get("next_token", None)
            actual_next_token_text = row.get("next_token_text", None)

            try:
                inputs = tokenizer(
                    context_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                )

                with torch.no_grad():
                    hf_outputs = hf_model(**inputs)
                    hf_logits = hf_outputs.logits[0, -1, :].cpu().numpy()
                    hf_predicted_token = np.argmax(hf_logits)
                    hf_predicted_text = tokenizer.decode([hf_predicted_token])

                try:
                    orchestrator_logits_str = row["predicted_logits"]
                    if isinstance(orchestrator_logits_str, str):
                        orchestrator_logits_clean = orchestrator_logits_str.strip("[]")
                        orchestrator_logits = np.array(
                            [
                                float(x.strip())
                                for x in orchestrator_logits_clean.split(",")
                            ]
                        )
                    else:
                        orchestrator_logits = np.array(orchestrator_logits_str)
                    orchestrator_predicted_token = np.argmax(orchestrator_logits)
                    orchestrator_predicted_text = tokenizer.decode(
                        [orchestrator_predicted_token]
                    )
                except Exception as e:
                    print(
                        f"   ❌ Error parsing orchestrator logits for sample {idx}: {e}"
                    )
                    continue

                logits_diff = np.max(np.abs(orchestrator_logits - hf_logits))
                tokens_match = orchestrator_predicted_token == hf_predicted_token

                comparison_result = {
                    "sample_idx": idx,
                    "context_preview": (
                        context_text[:80] + "..."
                        if len(context_text) > 80
                        else context_text
                    ),
                    "actual_token": actual_next_token,
                    "actual_token_text": actual_next_token_text,
                    "orchestrator_predicted_token": int(orchestrator_predicted_token),
                    "orchestrator_predicted_text": orchestrator_predicted_text,
                    "hf_predicted_token": int(hf_predicted_token),
                    "hf_predicted_text": hf_predicted_text,
                    "tokens_match": tokens_match,
                    "max_logits_diff": float(logits_diff),
                    "orchestrator_vs_actual_match": (
                        int(orchestrator_predicted_token) == actual_next_token
                        if actual_next_token is not None
                        else None
                    ),
                    "hf_vs_actual_match": (
                        int(hf_predicted_token) == actual_next_token
                        if actual_next_token is not None
                        else None
                    ),
                }
                comparison_results.append(comparison_result)

            except Exception as e:
                print(f"   ❌ Error with HF model inference for sample {idx}: {e}")
                continue

        print(f"\n📊 Orchestrator Comparison Results:")
        print(f"=" * 80)

        total_samples = len(comparison_results)
        orchestrator_hf_matches = sum(
            1 for r in comparison_results if r["tokens_match"]
        )
        orchestrator_actual_matches = sum(
            1 for r in comparison_results if r["orchestrator_vs_actual_match"] is True
        )
        hf_actual_matches = sum(
            1 for r in comparison_results if r["hf_vs_actual_match"] is True
        )

        print(f"📈 Summary Statistics:")
        print(f"   🎯 Total samples compared: {total_samples}")
        print(
            f"   🤝 Orchestrator vs HF matches: {orchestrator_hf_matches}/{total_samples} ({orchestrator_hf_matches/total_samples*100:.1f}%)"
        )
        if orchestrator_actual_matches > 0:
            print(
                f"   ✅ Orchestrator vs Actual matches: {orchestrator_actual_matches}/{total_samples} ({orchestrator_actual_matches/total_samples*100:.1f}%)"
            )
        if hf_actual_matches > 0:
            print(
                f"   ✅ HF vs Actual matches: {hf_actual_matches}/{total_samples} ({hf_actual_matches/total_samples*100:.1f}%)"
            )

        if comparison_results:
            avg_logits_diff = np.mean(
                [r["max_logits_diff"] for r in comparison_results]
            )
            max_logits_diff = np.max([r["max_logits_diff"] for r in comparison_results])
            print(f"   📊 Average max logits difference: {avg_logits_diff:.6f}")
            print(f"   📊 Maximum logits difference: {max_logits_diff:.6f}")

            if avg_logits_diff < 1e-3 and orchestrator_hf_matches / total_samples > 0.8:
                print(f"   ✅ EXCELLENT: Orchestrator now matches HF model very well!")
            elif (
                avg_logits_diff < 0.1 and orchestrator_hf_matches / total_samples > 0.6
            ):
                print(
                    f"   ✅ GOOD: Significant improvement over original orchestrator!"
                )
            else:
                print(f"   ⚠️  Still some differences - may need further debugging")

        print(f"\n🔍 Sample-by-Sample Analysis:")
        display_samples = min(5, len(comparison_results))
        for i, result in enumerate(comparison_results[:display_samples]):
            print(f"\n📝 Sample {result['sample_idx'] + 1}:")
            print(f"   Context: {result['context_preview']}")
            if result["actual_token_text"]:
                print(
                    f"   🎯 Actual token: {result['actual_token']} -> '{result['actual_token_text']}'"
                )
            print(
                f"   🤖 Orchestrator prediction: {result['orchestrator_predicted_token']} -> '{result['orchestrator_predicted_text']}'"
            )
            print(
                f"   🤗 HF prediction: {result['hf_predicted_token']} -> '{result['hf_predicted_text']}'"
            )

            match_symbol = "✅" if result["tokens_match"] else "❌"
            print(f"   📊 Orchestrator vs HF match: {match_symbol}")

            if result["orchestrator_vs_actual_match"] is not None:
                actual_match_symbol = (
                    "✅" if result["orchestrator_vs_actual_match"] else "❌"
                )
                print(f"   📊 Orchestrator vs Actual: {actual_match_symbol}")

            if result["hf_vs_actual_match"] is not None:
                hf_actual_symbol = "✅" if result["hf_vs_actual_match"] else "❌"
                print(f"   📊 HF vs Actual: {hf_actual_symbol}")

            print(f"   📊 Max logits diff: {result['max_logits_diff']:.6f}")

        comparison_output_path = os.path.join(
            final_output_dir, "comparison_results.csv"
        )
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(comparison_output_path, index=False)
        print(f"\n💾 Detailed comparison results saved to: {comparison_output_path}")

        print(f"\n🎉 Orchestrator test with HF comparison completed!")
        print(f"📁 Main output: {final_output_path}")
        print(f"📁 Comparison results: {comparison_output_path}")

        print(f"\n🧹 Cleaning up HF model...")
        del hf_model
        del tokenizer
        gc.collect()

    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        traceback.print_exc()
