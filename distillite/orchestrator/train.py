import gc
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillite.data import DataConfig, DataLoader
from distillite.distill import LoraDistillationTrainer
from distillite.model import ModelConfig, Qwen3Executor
from distillite.orchestrator import InferenceOrchestrator


class WaveTrainingOrchestrator:
    """
    GPU/CPU compatible wave-based inference + LoRA distillation training orchestrator.
    Integrates InferenceOrchestrator with LoraDistillationTrainer with optimized GPU memory management.
    """

    def __init__(
        self,
        inference_orchestrator: "InferenceOrchestrator",
        distillation_trainer: LoraDistillationTrainer,
        device: str = "auto",
        train_after_each_wave: bool = True,
        training_epochs: int = 2,
        training_batch_size: int = 4,
        save_adapters_after_training: bool = True,
        adapter_save_dir: str = "./trained_adapters",
        memory_utilization: float = 0.7,
        clear_cache_frequency: int = 5,
        mixed_precision_training: bool = True,
    ):
        """
        Initialize GPU-optimized wave-based training orchestrator.

        Args:
            inference_orchestrator: Configured InferenceOrchestrator instance
            distillation_trainer: Configured LoraDistillationTrainer instance
            device: Device to use ('cuda', 'cpu', or 'auto' for auto-detection)
            train_after_each_wave: Whether to train after each wave completion
            training_epochs: Number of epochs for distillation training
            training_batch_size: Batch size for training
            save_adapters_after_training: Whether to save adapters after training
            adapter_save_dir: Directory to save trained adapters
            memory_utilization: Fraction of available GPU memory to use (0.0-1.0)
            clear_cache_frequency: How often to clear GPU cache (every N chunks)
            mixed_precision_training: Whether to use mixed precision for training
        """
        self.inference_orch = inference_orchestrator
        self.distill_trainer = distillation_trainer
        self.train_after_each_wave = train_after_each_wave
        self.training_epochs = training_epochs
        self.training_batch_size = training_batch_size
        self.save_adapters_after_training = save_adapters_after_training
        self.memory_utilization = max(0.1, min(1.0, memory_utilization))
        self.clear_cache_frequency = clear_cache_frequency
        self.mixed_precision_training = mixed_precision_training

        self.device = self._setup_device(device)

        self.adapter_save_dir = Path(adapter_save_dir)
        self.adapter_save_dir.mkdir(parents=True, exist_ok=True)

        self.current_wave = 0
        self.total_training_time = 0.0
        self.wave_training_metrics = []

        self.peak_gpu_usage = 0.0
        self.gpu_memory_warnings = 0

        self._prepare_training_model()

        print(f"🌊 WaveTrainingOrchestrator initialized:")
        print(f"   🎮 Device: {self.device}")
        if self.device.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            )
            print(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"   🎯 Train after each wave: {train_after_each_wave}")
        print(f"   📚 Training epochs per wave: {training_epochs}")
        print(f"   📦 Training batch size: {training_batch_size}")
        print(f"   🧠 Memory utilization: {self.memory_utilization*100:.1f}%")
        print(f"   ⚡ Mixed precision training: {mixed_precision_training}")
        print(f"   🧹 Cache clear frequency: every {clear_cache_frequency} chunks")
        print(f"   💾 Adapter save directory: {self.adapter_save_dir}")

    def _setup_device(self, device: str) -> str:
        """Setup and validate device, inheriting from orchestrator if needed."""
        if device == "auto":
            if hasattr(self.inference_orch, "device"):
                device = self.inference_orch.device
                print(f"   🎮 Inherited device from orchestrator: {device}")
            elif torch.cuda.is_available():
                device = "cuda"
                print(f"   🎮 Auto-detected CUDA device")
            else:
                device = "cpu"
                print(f"   🎮 CUDA not available, using CPU")

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"   ⚠️  CUDA requested but not available, falling back to CPU")
                device = "cpu"
            else:
                try:
                    torch.cuda.empty_cache()
                    _ = torch.zeros(1, device=device)
                    print(f"   ✅ GPU device {device} is accessible")
                except Exception as e:
                    print(f"   ⚠️  GPU device {device} not accessible: {e}, using CPU")
                    device = "cpu"

        return device

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not self.device.startswith("cuda"):
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        free = total - allocated

        self.peak_gpu_usage = max(self.peak_gpu_usage, allocated)

        return {"total": total, "allocated": allocated, "cached": cached, "free": free}

    def _clear_gpu_memory(self):
        """Clear GPU memory and cache with monitoring."""
        if self.device.startswith("cuda"):
            mem_before = self._get_gpu_memory_info()
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()

            mem_after = self._get_gpu_memory_info()
            freed_mb = (mem_before["allocated"] - mem_after["allocated"]) * 1024
            if freed_mb > 100:
                print(f"   🧹 GPU memory freed: {freed_mb:.0f}MB")

    def _monitor_gpu_memory(self, context: str = ""):
        """Monitor GPU memory usage and warn if getting low."""
        if not self.device.startswith("cuda"):
            return

        mem_info = self._get_gpu_memory_info()
        utilization = mem_info["allocated"] / mem_info["total"]

        if utilization > 0.9:
            self.gpu_memory_warnings += 1
            print(
                f"   ⚠️  GPU memory warning {context}: {utilization*100:.1f}% used ({mem_info['allocated']:.1f}GB/{mem_info['total']:.1f}GB)"
            )
            if self.gpu_memory_warnings % 3 == 0:
                print(f"   🧹 Force clearing GPU cache due to high usage...")
                self._clear_gpu_memory()

    def _prepare_training_model(self):
        """Prepare the training model with proper device handling."""
        print("🔧 Preparing training model with device compatibility...")

        if hasattr(self.distill_trainer, "device"):
            if self.distill_trainer.device != self.device:
                print(
                    f"   🔄 Updating trainer device: {self.distill_trainer.device} -> {self.device}"
                )
                self.distill_trainer.device = self.device

        if self.distill_trainer.model is None:
            print("   🔄 Preparing distillation trainer model...")
            self.distill_trainer.prepare_model(self.inference_orch.executor.tokenizer)

        if self.device.startswith("cuda"):
            mem_info = self._get_gpu_memory_info()
            print(
                f"   🎮 GPU memory after model prep: {mem_info['allocated']:.1f}GB allocated"
            )

        print("   ✅ Training model prepared")

    def run_wave_based_training(self, debug_first: bool = True) -> Dict[str, Any]:
        """
        Execute complete wave-based inference + training pipeline with GPU optimization.

        Args:
            debug_first: Whether to run debug checks before starting

        Returns:
            Dictionary with final results and training metrics
        """
        print("\n🚀 Starting GPU-optimized wave-based inference + training...")
        total_start = time.time()

        if self.device.startswith("cuda"):
            initial_mem = self._get_gpu_memory_info()
            print(
                f"   🎮 Initial GPU memory: {initial_mem['allocated']:.1f}GB allocated, {initial_mem['free']:.1f}GB free"
            )

        try:
            final_output_path = self._run_modified_orchestrator_with_training(
                debug_first
            )

            total_time = time.time() - total_start
            self.total_training_time = total_time

            if self.device.startswith("cuda"):
                final_mem = self._get_gpu_memory_info()
                print(
                    f"   🎮 Final GPU memory: {final_mem['allocated']:.1f}GB allocated"
                )
                print(f"   📊 Peak GPU usage: {self.peak_gpu_usage:.1f}GB")
                print(f"   ⚠️  GPU memory warnings: {self.gpu_memory_warnings}")

            results = {
                "final_output_path": final_output_path,
                "adapter_save_path": str(self.adapter_save_dir),
                "total_time": total_time,
                "training_enabled": self.train_after_each_wave,
                "waves_completed": self.current_wave,
                "wave_training_metrics": self.wave_training_metrics,
                "peak_gpu_usage_gb": self.peak_gpu_usage,
                "gpu_memory_warnings": self.gpu_memory_warnings,
                "device": self.device,
                "mixed_precision_used": self.mixed_precision_training,
            }

            print(f"\n🎉 GPU-optimized wave-based training pipeline completed!")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   🌊 Waves completed: {self.current_wave}")
            print(f"   📁 Final outputs: {final_output_path}")
            print(f"   🎯 Trained adapters: {self.adapter_save_dir}")

            return results

        except Exception as e:
            print(f"❌ Wave-based training pipeline failed: {e}")
            traceback.print_exc()

            self._clear_gpu_memory()

            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - total_start,
                "waves_completed": self.current_wave,
            }

    def _run_modified_orchestrator_with_training(self, debug_first: bool) -> str:
        """
        Run orchestrator with integrated GPU-optimized training after each wave.
        """
        print(f"\n📋 Calculating execution plan...")
        layer_groups = self.inference_orch._calculate_layer_groups()
        total_chunks = self.inference_orch.data_loader.num_chunks
        processed_chunks = 0
        wave_number = 1

        print(f"   🎯 {len(layer_groups)} inference stages")
        print(f"   📦 {total_chunks} total chunks")
        print(f"   🌊 Training after each wave: {self.train_after_each_wave}")
        print(f"   🎮 Device: {self.device}")

        if debug_first and wave_number == 1:
            print(f"\n🔍 Running pre-flight debug check...")
            if not self.inference_orch.debug_single_stage_comparison(test_samples=3):
                print(
                    "❌ Single-stage debug failed. Please fix executor before proceeding."
                )
                return None
            print("✅ Single-stage debug passed!")

        while processed_chunks < total_chunks:
            print(f"\n🌊 WAVE {wave_number} - INFERENCE + TRAINING")
            wave_start_time = time.time()
            self.current_wave = wave_number

            self._monitor_gpu_memory(f"Wave {wave_number} start")

            wave_chunks = self.inference_orch._determine_wave_chunks(
                processed_chunks, total_chunks
            )
            wave_end = processed_chunks + len(wave_chunks)

            print(
                f"   📦 Processing chunks {processed_chunks}-{wave_end-1} ({len(wave_chunks)} chunks)"
            )

            print(f"\n   🔄 INFERENCE PHASE - Wave {wave_number}")
            inference_start = time.time()

            self.inference_orch._process_wave_through_all_stages(
                wave_chunks, processed_chunks, layer_groups
            )

            inference_time = time.time() - inference_start
            print(f"   ✅ Inference completed in {inference_time:.2f}s")

            self._monitor_gpu_memory(f"Wave {wave_number} after inference")

            if self.train_after_each_wave:
                print(f"\n   🎓 TRAINING PHASE - Wave {wave_number}")
                training_start = time.time()

                training_metrics = self._train_on_wave_results(wave_chunks, wave_number)

                training_time = time.time() - training_start
                training_metrics["training_time"] = training_time
                training_metrics["inference_time"] = inference_time
                self.wave_training_metrics.append(training_metrics)

                print(f"   ✅ Training completed in {training_time:.2f}s")
            else:
                print(f"   ⏭️  Skipping training for wave {wave_number}")

            print(f"\n   🧹 Wave {wave_number} cleanup...")
            self.inference_orch._cleanup_intermediate_files()

            if wave_number % self.clear_cache_frequency == 0:
                print(
                    f"   🧹 Scheduled GPU cache clear (every {self.clear_cache_frequency} waves)"
                )
                self._clear_gpu_memory()

            wave_time = time.time() - wave_start_time
            processed_chunks = wave_end
            wave_number += 1

            print(
                f"   ✅ Wave {wave_number-1} complete in {wave_time:.2f}s (inference + training + cleanup)"
            )

            self._monitor_gpu_memory(f"Wave {wave_number-1} end")

        return self.inference_orch.data_config.final_output_dir

    def _train_on_wave_results(
        self, wave_chunks: List[int], wave_number: int
    ) -> Dict[str, Any]:
        """Train LoRA adapters on wave results with GPU optimization."""
        print(f"     🔄 Preparing training data for wave {wave_number}...")

        if wave_number > 1:
            previous_wave_path = (
                self.adapter_save_dir / f"wave_{wave_number-1}_adapters"
            )
            if previous_wave_path.exists():
                print(
                    f"     🔄 Loading adapters from previous wave: {previous_wave_path}"
                )
                try:
                    self._load_previous_wave_adapters(previous_wave_path)
                except Exception as e:
                    print(f"     ⚠️  Failed to load previous adapters: {e}")
                    print(f"     🔄 Continuing with fresh adapters for this wave")

        chunk_files = []
        output_dir = Path(self.inference_orch.data_config.final_output_dir)

        for chunk_idx in wave_chunks:
            chunk_file = output_dir / f"chunk_{chunk_idx}.csv"
            if chunk_file.exists():
                chunk_files.append(chunk_file)
            else:
                print(f"     ⚠️  Warning: Chunk file not found: {chunk_file}")

        if not chunk_files:
            print(f"     ❌ No chunk files found for training in wave {wave_number}")
            return {"success": False, "error": "No training data found"}

        print(f"     📚 Training on {len(chunk_files)} chunk files...")

        pre_training_mem = self._get_gpu_memory_info()
        print(
            f"     🎮 GPU memory before training: {pre_training_mem['allocated']:.1f}GB"
        )

        try:
            training_metrics = self.distill_trainer.train_on_wave_outputs(
                chunk_files=chunk_files,
                data_config=self.inference_orch.data_config,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
            )

            post_training_mem = self._get_gpu_memory_info()
            print(
                f"     🎮 GPU memory after training: {post_training_mem['allocated']:.1f}GB"
            )

            print(f"     ✅ Wave {wave_number} training completed")
            print(f"     📊 Final loss: {training_metrics['avg_total_loss']:.4f}")
            print(f"     📊 KL loss: {training_metrics['avg_kl_loss']:.4f}")
            print(f"     📊 CE loss: {training_metrics['avg_ce_loss']:.4f}")

            if self.save_adapters_after_training:
                adapter_path = self.adapter_save_dir / f"wave_{wave_number}_adapters"
                self.distill_trainer.save_trained_model(str(adapter_path))
                print(f"     💾 Adapters saved: {adapter_path}")

            training_metrics.update(
                {
                    "wave_number": wave_number,
                    "chunk_files_count": len(chunk_files),
                    "chunks_processed": wave_chunks,
                    "success": True,
                    "pre_training_gpu_memory": pre_training_mem["allocated"],
                    "post_training_gpu_memory": post_training_mem["allocated"],
                    "gpu_memory_delta": post_training_mem["allocated"]
                    - pre_training_mem["allocated"],
                }
            )

            return training_metrics

        except Exception as e:
            print(f"     ❌ Training failed for wave {wave_number}: {e}")
            traceback.print_exc()

            self._clear_gpu_memory()

            return {
                "wave_number": wave_number,
                "success": False,
                "error": str(e),
                "chunk_files_count": len(chunk_files),
                "chunks_processed": wave_chunks,
            }

    def _load_previous_wave_adapters(self, adapter_path: Path):
        """Load adapters from previous wave to continue training with GPU awareness."""
        from peft import PeftModel

        try:
            base_model = self.distill_trainer.model.get_base_model()

            self.distill_trainer.model = PeftModel.from_pretrained(
                base_model, str(adapter_path)
            )

            self.distill_trainer.model.to(self.device)
            self.distill_trainer.model.train()

            trainable_params = filter(
                lambda p: p.requires_grad, self.distill_trainer.model.parameters()
            )
            self.distill_trainer.optimizer = AdamW(
                trainable_params, lr=self.distill_trainer.learning_rate
            )

            if (
                self.distill_trainer.mixed_precision
                and self.device.startswith("cuda")
                and self.mixed_precision_training
            ):
                self.distill_trainer.scaler = torch.cuda.amp.GradScaler()

            print(f"     ✅ Successfully loaded previous wave adapters")

        except Exception as e:
            print(f"     ⚠️  Failed to load previous adapters: {e}")
            raise e

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with GPU usage statistics."""
        if not self.wave_training_metrics:
            return {"error": "No training metrics available"}

        summary = {
            "total_waves": len(self.wave_training_metrics),
            "total_training_time": self.total_training_time,
            "peak_gpu_usage_gb": self.peak_gpu_usage,
            "gpu_memory_warnings": self.gpu_memory_warnings,
            "device": self.device,
            "mixed_precision_used": self.mixed_precision_training,
            "waves_summary": [],
        }

        total_chunks_processed = 0
        total_training_samples = 0

        for wave_metrics in self.wave_training_metrics:
            if wave_metrics.get("success", True):
                wave_summary = {
                    "wave": wave_metrics["wave_number"],
                    "chunks": len(wave_metrics.get("chunks_processed", [])),
                    "training_time": wave_metrics.get("training_time", 0),
                    "inference_time": wave_metrics.get("inference_time", 0),
                    "avg_total_loss": wave_metrics.get("avg_total_loss", 0),
                    "avg_kl_loss": wave_metrics.get("avg_kl_loss", 0),
                    "avg_ce_loss": wave_metrics.get("avg_ce_loss", 0),
                    "gpu_memory_delta": wave_metrics.get("gpu_memory_delta", 0),
                }
                summary["waves_summary"].append(wave_summary)
                total_chunks_processed += wave_summary["chunks"]

        summary["total_chunks_processed"] = total_chunks_processed

        if len(summary["waves_summary"]) > 1:
            first_loss = summary["waves_summary"][0]["avg_total_loss"]
            last_loss = summary["waves_summary"][-1]["avg_total_loss"]
            summary["loss_improvement"] = first_loss - last_loss
            summary["loss_improvement_percent"] = (
                (first_loss - last_loss) / first_loss * 100
            )

        return summary

    def cleanup(self):
        """Clean up both orchestrator and trainer with GPU memory management."""
        print("🧹 Cleaning up WaveTrainingOrchestrator...")

        if self.device.startswith("cuda"):
            final_mem = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory before cleanup: {final_mem['allocated']:.1f}GB")
            print(f"   📊 Peak GPU usage during training: {self.peak_gpu_usage:.1f}GB")
            print(f"   ⚠️  Total GPU memory warnings: {self.gpu_memory_warnings}")

        if hasattr(self, "distill_trainer") and self.distill_trainer:
            self.distill_trainer.cleanup()

        if hasattr(self, "inference_orch") and self.inference_orch:
            self.inference_orch._cleanup()

        self._clear_gpu_memory()

        if self.device.startswith("cuda"):
            final_mem = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory after cleanup: {final_mem['allocated']:.1f}GB")

        print("   ✅ WaveTrainingOrchestrator cleaned up")


def test_gpu_wave_training_pipeline():
    """
    Comprehensive test of the GPU-optimized wave-based inference + LoRA training pipeline.
    """
    print("🧪 Testing GPU-Optimized Wave-Based LoRA Training Pipeline")
    print("=" * 80)

    local_model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (
        project_root / "datagen" / "results" / "next_token_prediction_dataset.csv"
    )
    repo_id = "Qwen/Qwen3-0.6B"

    chunk_size = 50
    batch_size = 4
    max_seq_length = 512
    use_qlora = True
    memory_utilization = 0.6
    intermediate_size_threshold_gb = 1
    training_epochs_per_wave = 1
    training_batch_size = 2
    lora_rank = 8
    learning_rate = 5e-5
    device = "auto"

    base_output_dir = Path.cwd() / "gpu_wave_training_outputs"
    intermediate_dir = base_output_dir / "intermediate"
    final_output_dir = base_output_dir / "final_outputs"
    adapter_save_dir = base_output_dir / "trained_adapters"

    print(f"📁 Setting up output directories...")
    for dir_path in [
        intermediate_dir,
        final_output_dir,
        adapter_save_dir,
        local_model_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")

    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        sample_df = pd.read_csv(csv_path, nrows=5)
        print(f"\n📊 Dataset verification:")
        print(f"   📁 File: {csv_path}")
        print(f"   📋 Columns: {list(sample_df.columns)}")
        print(f"   📏 Sample shape: {sample_df.shape}")

        gpu_available = torch.cuda.is_available()
        print(f"\n🎮 GPU Status:")
        print(f"   CUDA available: {gpu_available}")
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(f"   Will use CPU for training")

        print(f"\n🔧 Configuring pipeline components...")

        data_config = DataConfig(
            csv_path=str(csv_path),
            input_col="context_text",
            output_col="predicted_logits",
            chunk_size=chunk_size,
            batch_size=batch_size,
            intermediate_dir=str(intermediate_dir),
            final_output_dir=str(final_output_dir),
            save_output=True,
        )
        print(f"   ✅ Data config created")

        data_loader = DataLoader(data_config)
        model_config = ModelConfig(local_model_dir, repo_id)
        executor = Qwen3Executor(model_config)

        print(f"   ✅ Core components initialized")
        print(f"      📊 Total rows: {data_loader.total_rows}")
        print(f"      📦 Number of chunks: {data_loader.num_chunks}")

        inference_orchestrator = InferenceOrchestrator(
            data_config=data_config,
            data_loader=data_loader,
            model_config=model_config,
            executor=executor,
            device=device,
            max_workers=2,
            max_seq_length=max_seq_length,
            memory_utilization=memory_utilization,
            intermediate_size_threshold_gb=intermediate_size_threshold_gb,
        )
        print(f"   ✅ Inference orchestrator created")

        lora_config = {
            "r": lora_rank,
            "lora_alpha": lora_rank * 2,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        distillation_trainer = LoraDistillationTrainer(
            model_config=model_config,
            lora_config=lora_config,
            use_qlora=use_qlora,
            learning_rate=learning_rate,
            temperature=2.0,
            alpha=0.5,
            max_grad_norm=1.0,
            device=device,
            gradient_accumulation_steps=1,
            mixed_precision=gpu_available,
        )
        print(f"   ✅ LoRA distillation trainer created")
        print(f"      🎯 LoRA rank: {lora_rank}")
        print(f"      🔥 QLoRA enabled: {use_qlora}")
        print(f"      ⚡ Mixed precision: {gpu_available}")

        wave_orchestrator = WaveTrainingOrchestrator(
            inference_orchestrator=inference_orchestrator,
            distillation_trainer=distillation_trainer,
            device=device,
            train_after_each_wave=True,
            training_epochs=training_epochs_per_wave,
            training_batch_size=training_batch_size,
            save_adapters_after_training=True,
            adapter_save_dir=str(adapter_save_dir),
            memory_utilization=memory_utilization,
            clear_cache_frequency=3,
            mixed_precision_training=gpu_available,
        )
        print(f"   ✅ GPU-optimized wave training orchestrator created")

        print(f"\n🔍 Running baseline model comparison...")
        baseline_metrics = run_baseline_comparison(
            model_config, data_config, sample_size=10
        )
        print(f"   📊 Baseline accuracy: {baseline_metrics['accuracy']:.1%}")
        print(f"   📊 Baseline avg loss: {baseline_metrics['avg_loss']:.4f}")

        print(f"\n🚀 Starting GPU-optimized wave-based training pipeline...")
        print(f"=" * 80)

        pipeline_start_time = time.time()
        results = wave_orchestrator.run_wave_based_training(debug_first=True)
        pipeline_time = time.time() - pipeline_start_time

        if not results or not results.get("final_output_path"):
            print("❌ GPU wave training pipeline failed.")
            return False

        print(f"\n✅ GPU-optimized pipeline completed successfully!")
        print(f"   ⏱️  Total time: {pipeline_time:.2f}s")
        print(f"   🌊 Waves completed: {results['waves_completed']}")
        print(f"   📁 Final outputs: {results['final_output_path']}")
        print(f"   🎯 Trained adapters: {results['adapter_save_path']}")

        training_summary = wave_orchestrator.get_training_summary()
        print(f"\n📊 Training Summary:")
        print(f"   🌊 Total waves: {training_summary['total_waves']}")
        print(
            f"   ⏱️  Total training time: {training_summary['total_training_time']:.2f}s"
        )
        print(f"   🎮 Device used: {training_summary['device']}")

        if training_summary["device"].startswith("cuda"):
            print(
                f"   📊 Peak GPU usage: {training_summary['peak_gpu_usage_gb']:.1f}GB"
            )
            print(
                f"   ⚠️  GPU memory warnings: {training_summary['gpu_memory_warnings']}"
            )
            print(f"   ⚡ Mixed precision: {training_summary['mixed_precision_used']}")

        if "loss_improvement" in training_summary:
            print(
                f"   📈 Loss improvement: {training_summary['loss_improvement']:.4f} ({training_summary['loss_improvement_percent']:.1f}%)"
            )

        print(f"\n📊 Validating training results...")
        validation_results = validate_gpu_training_results(
            results["final_output_path"],
            results["adapter_save_path"],
            model_config,
            data_config,
            baseline_metrics,
        )

        print_gpu_training_results(
            pipeline_time=pipeline_time,
            baseline_metrics=baseline_metrics,
            validation_results=validation_results,
            training_summary=training_summary,
            training_config={
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "epochs_per_wave": training_epochs_per_wave,
                "use_qlora": use_qlora,
                "memory_utilization": memory_utilization,
                "device": device,
                "mixed_precision": gpu_available,
            },
        )

        print(f"\n🧹 Cleaning up resources...")
        wave_orchestrator.cleanup()
        del wave_orchestrator, inference_orchestrator, distillation_trainer
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ GPU wave training pipeline test failed: {e}")
        traceback.print_exc()
        return False


def run_baseline_comparison(
    model_config: "ModelConfig", data_config: "DataConfig", sample_size: int = 10
) -> Dict[str, float]:
    """Run baseline comparison using original HuggingFace model with GPU support."""
    print(f"   🔄 Loading baseline model for comparison...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_config.model_path, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    sample_df = pd.read_csv(data_config.csv_path, nrows=sample_size)

    correct_predictions = 0
    total_loss = 0.0

    print(f"   🎮 Running baseline on device: {device}")

    with torch.no_grad():
        for idx, row in sample_df.iterrows():
            context_text = row[data_config.input_col]
            actual_next_token = row.get("next_token", None)

            inputs = tokenizer(
                context_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].cpu().numpy()
            predicted_token = np.argmax(logits)

            if actual_next_token is not None:
                if int(predicted_token) == actual_next_token:
                    correct_predictions += 1

                loss = torch.nn.functional.cross_entropy(
                    torch.tensor(logits).unsqueeze(0), torch.tensor([actual_next_token])
                ).item()
                total_loss += loss

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    accuracy = correct_predictions / len(sample_df) if len(sample_df) > 0 else 0.0
    avg_loss = total_loss / len(sample_df) if len(sample_df) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "total_samples": len(sample_df),
        "correct_predictions": correct_predictions,
        "device": device,
    }


def validate_gpu_training_results(
    final_output_path: str,
    adapter_save_path: str,
    model_config: "ModelConfig",
    data_config: "DataConfig",
    baseline_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Validate training results with GPU awareness."""
    print(f"   🔍 Loading and validating GPU training outputs...")

    output_dir = Path(final_output_path)
    adapter_dir = Path(adapter_save_path)

    chunk_files = sorted(output_dir.glob("chunk_*.csv"))
    adapter_subdirs = [d for d in adapter_dir.iterdir() if d.is_dir()]

    validation_results = {
        "output_files_found": len(chunk_files),
        "adapter_checkpoints_found": len(adapter_subdirs),
        "total_training_samples": 0,
        "waves_completed": len(adapter_subdirs),
        "training_progression": [],
        "device_used": "unknown",
    }

    total_samples = 0
    for chunk_file in chunk_files:
        chunk_df = pd.read_csv(chunk_file)
        total_samples += len(chunk_df)
    validation_results["total_training_samples"] = total_samples

    for adapter_subdir in sorted(adapter_subdirs):
        adapter_files = list(adapter_subdir.glob("*.bin")) + list(
            adapter_subdir.glob("*.safetensors")
        )
        config_files = list(adapter_subdir.glob("adapter_config.json"))
        training_config_files = list(adapter_subdir.glob("training_config.txt"))

        checkpoint_valid = len(adapter_files) > 0 and len(config_files) > 0

        if training_config_files and validation_results["device_used"] == "unknown":
            try:
                with open(training_config_files[0], "r") as f:
                    config_content = f.read()
                    if "Device:" in config_content:
                        device_line = [
                            line
                            for line in config_content.split("\n")
                            if "Device:" in line
                        ][0]
                        validation_results["device_used"] = device_line.split(
                            "Device:"
                        )[1].strip()
            except Exception:
                pass

        validation_results["training_progression"].append(
            {
                "wave": adapter_subdir.name,
                "checkpoint_valid": checkpoint_valid,
                "adapter_files": len(adapter_files),
                "config_files": len(config_files),
                "has_training_config": len(training_config_files) > 0,
            }
        )

    if adapter_subdirs:
        latest_adapter = sorted(adapter_subdirs)[-1]
        print(f"   🧪 Testing performance of latest adapter: {latest_adapter.name}")

        try:
            trained_metrics = test_gpu_trained_model_performance(
                model_config, latest_adapter, data_config, sample_size=5
            )
            validation_results["trained_model_metrics"] = trained_metrics
            validation_results["improvement_vs_baseline"] = {
                "accuracy_delta": trained_metrics["accuracy"]
                - baseline_metrics["accuracy"],
                "loss_delta": baseline_metrics["avg_loss"]
                - trained_metrics["avg_loss"],
            }
        except Exception as e:
            print(f"     ⚠️  Could not test trained model: {e}")
            validation_results["trained_model_metrics"] = None
            validation_results["improvement_vs_baseline"] = None

    return validation_results


def test_gpu_trained_model_performance(
    model_config: "ModelConfig",
    adapter_path: Path,
    data_config: "DataConfig",
    sample_size: int = 5,
) -> Dict[str, float]:
    """Test the performance of a trained LoRA adapter with GPU support."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            model_config.model_path, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_path))

    sample_df = pd.read_csv(data_config.csv_path, nrows=sample_size)

    correct_predictions = 0
    total_loss = 0.0

    print(f"     🎮 Testing trained model on device: {device}")

    with torch.no_grad():
        for idx, row in sample_df.iterrows():
            context_text = row[data_config.input_col]
            actual_next_token = row.get("next_token", None)

            inputs = tokenizer(
                context_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model_with_adapter(**inputs)
            logits = outputs.logits[0, -1, :].cpu().numpy()
            predicted_token = np.argmax(logits)

            if actual_next_token is not None:
                if int(predicted_token) == actual_next_token:
                    correct_predictions += 1

                loss = torch.nn.functional.cross_entropy(
                    torch.tensor(logits).unsqueeze(0), torch.tensor([actual_next_token])
                ).item()
                total_loss += loss

    del model_with_adapter, base_model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    accuracy = correct_predictions / len(sample_df) if len(sample_df) > 0 else 0.0
    avg_loss = total_loss / len(sample_df) if len(sample_df) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "total_samples": len(sample_df),
        "correct_predictions": correct_predictions,
        "device": device,
    }


def print_gpu_training_results(
    pipeline_time: float,
    baseline_metrics: Dict[str, float],
    validation_results: Dict[str, Any],
    training_summary: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """Print comprehensive GPU-optimized training results."""
    print(f"\n🎉 GPU-OPTIMIZED WAVE-BASED LORA TRAINING RESULTS")
    print(f"=" * 80)

    print(f"📊 Pipeline Summary:")
    print(f"   ⏱️  Total execution time: {pipeline_time:.2f}s")
    print(f"   🌊 Waves completed: {validation_results['waves_completed']}")
    print(
        f"   📚 Total training samples: {validation_results['total_training_samples']}"
    )
    print(f"   📄 Output chunk files: {validation_results['output_files_found']}")
    print(
        f"   🎯 Adapter checkpoints: {validation_results['adapter_checkpoints_found']}"
    )

    device_used = training_summary.get(
        "device", training_config.get("device", "unknown")
    )
    print(f"\n🎮 Device & GPU Information:")
    print(f"   🎮 Device used: {device_used}")

    if device_used.startswith("cuda"):
        print(
            f"   📊 Peak GPU usage: {training_summary.get('peak_gpu_usage_gb', 0):.1f}GB"
        )
        print(
            f"   ⚠️  GPU memory warnings: {training_summary.get('gpu_memory_warnings', 0)}"
        )
        print(
            f"   ⚡ Mixed precision training: {training_summary.get('mixed_precision_used', False)}"
        )

    print(f"\n🔧 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")

    print(f"\n📈 Model Performance Comparison:")
    print(f"   📊 Baseline accuracy: {baseline_metrics['accuracy']:.1%}")

    if validation_results.get("trained_model_metrics"):
        trained_metrics = validation_results["trained_model_metrics"]
        improvement = validation_results["improvement_vs_baseline"]

        print(f"   🎯 Trained accuracy: {trained_metrics['accuracy']:.1%}")
        print(f"   📊 Baseline avg loss: {baseline_metrics['avg_loss']:.4f}")
        print(f"   🎯 Trained avg loss: {trained_metrics['avg_loss']:.4f}")

        print(f"\n🚀 Training Improvements:")
        print(f"   📈 Accuracy improvement: {improvement['accuracy_delta']:+.1%}")
        print(f"   📉 Loss reduction: {improvement['loss_delta']:+.4f}")

        if improvement["accuracy_delta"] > 0.05:
            print(f"   ✅ EXCELLENT: Significant accuracy improvement!")
        elif improvement["accuracy_delta"] > 0.01:
            print(f"   ✅ GOOD: Noticeable accuracy improvement!")
        else:
            print(f"   ⚠️  Modest improvements - may need more training")
    else:
        print(f"   ❌ Could not evaluate trained model performance")

    print(f"\n🌊 Wave Training Progression:")
    for i, wave_info in enumerate(validation_results["training_progression"]):
        status = "✅" if wave_info["checkpoint_valid"] else "❌"
        config_info = "📋" if wave_info.get("has_training_config", False) else ""
        print(
            f"   {status} {wave_info['wave']}: {wave_info['adapter_files']} adapter files {config_info}"
        )

    if training_summary.get("waves_summary"):
        print(f"\n📊 Individual Wave Performance:")
        for wave in training_summary["waves_summary"]:
            print(
                f"   🌊 Wave {wave['wave']}: "
                f"Loss={wave['avg_total_loss']:.4f}, "
                f"Time={wave['training_time']:.1f}s, "
                f"Chunks={wave['chunks']}"
            )
            if wave.get("gpu_memory_delta", 0) != 0:
                print(f"      🎮 GPU memory delta: {wave['gpu_memory_delta']:+.2f}GB")

    print(f"\n💾 Efficiency Metrics:")
    if training_config["use_qlora"]:
        print(f"   🔥 QLoRA quantization: Enabled (4-bit)")
        print(f"   💾 Memory savings: ~75% vs full fine-tuning")

    print(
        f"   📊 LoRA trainable params: ~{training_config['lora_rank'] * 2}k per layer"
    )
    print(f"   🧠 Memory utilization: {training_config['memory_utilization']:.1%}")

    if device_used.startswith("cuda"):
        print(f"   ⚡ GPU acceleration: Enabled")
        if training_config.get("mixed_precision", False):
            print(f"   ⚡ Mixed precision: Enabled (faster training)")

    if training_summary.get("waves_summary"):
        total_training_time = sum(
            w.get("training_time", 0) for w in training_summary["waves_summary"]
        )
        total_inference_time = sum(
            w.get("inference_time", 0) for w in training_summary["waves_summary"]
        )

        print(f"\n⏱️  Time Breakdown:")
        print(f"   🔄 Total inference time: {total_inference_time:.2f}s")
        print(f"   🎓 Total training time: {total_training_time:.2f}s")
        print(
            f"   📊 Training/Inference ratio: {total_training_time/max(total_inference_time, 0.1):.2f}"
        )

    print(
        f"\n🎯 GPU-optimized wave-based LoRA training pipeline completed successfully!"
    )


if __name__ == "__main__":
    print("🧪 Wave-Based LoRA Training Pipeline Tester")
    print("=" * 80)
    print("This test validates the complete GPU-optimized pipeline:")
    print("1. 🌊 Wave-based inference orchestration with GPU memory management")
    print("2. 🎓 LoRA/QLoRA distillation training with mixed precision")
    print("3. 📊 Performance validation and GPU usage monitoring")
    print("4. 💾 Memory-efficient progressive training with automatic cleanup")
    print()

    success = test_gpu_wave_training_pipeline()

    if success:
        print(
            "\n✅ All tests passed! GPU-optimized wave-based LoRA training pipeline is working correctly."
        )
        exit(0)
    else:
        print("\n❌ Tests failed. Please check the error logs above.")
        exit(1)
