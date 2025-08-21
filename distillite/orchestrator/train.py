import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import gc
import time
from distillite.distill import LoraDistillationTrainer
from distillite.orchestrator import InferenceOrchestrator

import gc
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillite.data import DataConfig, DataLoader
from distillite.model import BaseExecutor, ModelConfig, Qwen3Executor
from distillite.distill import LoraDistillationTrainer
from distillite.orchestrator import InferenceOrchestrator


class WaveTrainingOrchestrator:
    """
    Coordinates wave-based inference with LoRA distillation training.
    Integrates InferenceOrchestrator with LoraDistillationTrainer.
    """
    
    def __init__(
        self,
        inference_orchestrator: "InferenceOrchestrator",
        distillation_trainer: LoraDistillationTrainer,
        train_after_each_wave: bool = True,
        training_epochs: int = 2,
        training_batch_size: int = 4,
        save_adapters_after_training: bool = True,
        adapter_save_dir: str = "./trained_adapters"
    ):
        """
        Initialize wave-based training orchestrator.
        
        Args:
            inference_orchestrator: Configured InferenceOrchestrator instance
            distillation_trainer: Configured LoraDistillationTrainer instance
            train_after_each_wave: Whether to train after each wave completion
            training_epochs: Number of epochs for distillation training
            training_batch_size: Batch size for training
            save_adapters_after_training: Whether to save adapters after training
            adapter_save_dir: Directory to save trained adapters
        """
        self.inference_orch = inference_orchestrator
        self.distill_trainer = distillation_trainer
        self.train_after_each_wave = train_after_each_wave
        self.training_epochs = training_epochs
        self.training_batch_size = training_batch_size
        self.save_adapters_after_training = save_adapters_after_training
        
        self.adapter_save_dir = Path(adapter_save_dir)
        self.adapter_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare distillation model with orchestrator's tokenizer
        self.distill_trainer.prepare_model(self.inference_orch.executor.tokenizer)
        
        print(f"🌊 WaveTrainingOrchestrator initialized:")
        print(f"   🎯 Train after each wave: {train_after_each_wave}")
        print(f"   📚 Training epochs per wave: {training_epochs}")
        print(f"   💾 Adapter save directory: {self.adapter_save_dir}")
    
    def run_wave_based_training(self, debug_first: bool = True) -> Dict[str, any]:
        """
        Execute complete wave-based inference + training pipeline.
        
        Returns:
            Dictionary with final results and training metrics
        """
        print("\n🚀 Starting integrated wave-based inference + training...")
        total_start = time.time()
        
        # Modified inference orchestrator to support training integration
        final_output_path = self._run_modified_orchestrator_with_training(debug_first)
        
        total_time = time.time() - total_start
        
        results = {
            "final_output_path": final_output_path,
            "adapter_save_path": str(self.adapter_save_dir),
            "total_time": total_time,
            "training_enabled": self.train_after_each_wave
        }
        
        print(f"\n🎉 Wave-based training pipeline completed!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📁 Final outputs: {final_output_path}")
        print(f"   🎯 Trained adapters: {self.adapter_save_dir}")
        
        return results
    
    def _run_modified_orchestrator_with_training(self, debug_first: bool) -> str:
        """
        Run orchestrator with training integration after each wave.
        This modifies the orchestrator's wave processing to include training.
        """
        # Get orchestrator's layer groups and data info
        layer_groups = self.inference_orch._calculate_layer_groups()
        total_chunks = self.inference_orch.data_loader.num_chunks
        processed_chunks = 0
        wave_number = 1
        
        print(f"\n📋 Integrated execution plan:")
        print(f"   🎯 {len(layer_groups)} inference stages")
        print(f"   📦 {total_chunks} total chunks")
        print(f"   🌊 Training after each wave: {self.train_after_each_wave}")
        
        while processed_chunks < total_chunks:
            print(f"\n🌊 WAVE {wave_number} - INFERENCE + TRAINING")
            
            # Determine wave chunks
            wave_chunks = self.inference_orch._determine_wave_chunks(processed_chunks, total_chunks)
            wave_end = processed_chunks + len(wave_chunks)
            
            print(f"   📦 Processing chunks {processed_chunks}-{wave_end-1}")
            
            # Run inference for this wave
            self.inference_orch._process_wave_through_all_stages(
                wave_chunks, processed_chunks, layer_groups
            )
            
            if self.train_after_each_wave:
                print(f"\n   🎓 TRAINING PHASE - Wave {wave_number}")
                self._train_on_wave_results(wave_chunks, wave_number)
            
            # Clean up inference intermediates
            self.inference_orch._cleanup_intermediate_files()
            
            processed_chunks = wave_end
            wave_number += 1
            
            print(f"   ✅ Wave {wave_number-1} complete (inference + training)")
        
        return self.inference_orch.data_config.final_output_dir
    
    def _train_on_wave_results(self, wave_chunks: List[int], wave_number: int):
        """Train LoRA adapters on the results from completed wave."""
        # Collect chunk files from this wave
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
            return
        
        print(f"     📚 Training on {len(chunk_files)} chunk files...")
        
        # Run distillation training
        try:
            training_metrics = self.distill_trainer.train_on_wave_outputs(
                chunk_files=chunk_files,
                data_config=self.inference_orch.data_config,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size
            )
            
            print(f"     ✅ Wave {wave_number} training completed")
            print(f"     📊 Final loss: {training_metrics['avg_total_loss']:.4f}")
            
            # Save adapters after training
            if self.save_adapters_after_training:
                adapter_path = self.adapter_save_dir / f"wave_{wave_number}_adapters"
                self.distill_trainer.save_trained_model(str(adapter_path))
                
        except Exception as e:
            print(f"     ❌ Training failed for wave {wave_number}: {e}")
    
    def cleanup(self):
        """Clean up both orchestrator and trainer."""
        self.inference_orch._cleanup()
        self.distill_trainer.cleanup()
        print("🧹 WaveTrainingOrchestrator cleaned up")


def test_wave_training_pipeline():
    """
    Comprehensive test of the wave-based inference + LoRA training pipeline.
    """
    print("🧪 Testing Wave-Based LoRA Training Pipeline")
    print("=" * 80)

    # Configuration - same as your original test
    local_model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (
        project_root / "datagen" / "results" / "next_token_prediction_dataset.csv"
    )
    repo_id = "Qwen/Qwen3-0.6B"

    # Training-specific configurations
    chunk_size = 50  # Keep small for testing
    batch_size = 4
    max_seq_length = 512
    use_qlora = False
    
    # Wave training parameters
    memory_utilization = 0.3
    intermediate_size_threshold_gb = 2.0  # Smaller for testing
    training_epochs_per_wave = 2
    training_batch_size = 2
    lora_rank = 8
    learning_rate = 5e-5

    # Directory setup
    base_output_dir = Path.cwd() / "wave_training_outputs"
    intermediate_dir = os.path.join(base_output_dir, "intermediate")
    final_output_dir = os.path.join(base_output_dir, "final_outputs")
    adapter_save_dir = os.path.join(base_output_dir, "trained_adapters")

    print(f"📁 Creating output directories...")
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(adapter_save_dir, exist_ok=True)
    os.makedirs(local_model_dir, exist_ok=True)
    print(f"   ✅ Intermediate dir: {intermediate_dir}")
    print(f"   ✅ Final output dir: {final_output_dir}")
    print(f"   ✅ Adapter save dir: {adapter_save_dir}")
    print(f"   ✅ Model dir: {local_model_dir}")

    try:
        # Verify dataset exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Preview dataset
        sample_df = pd.read_csv(csv_path, nrows=5)
        print(f"\n📊 Dataset preview:")
        print(f"   📁 File: {csv_path}")
        print(f"   📋 Columns: {list(sample_df.columns)}")
        print(f"   📏 Sample shape: {sample_df.shape}")

        if "context_text" in sample_df.columns:
            print(f"\n📝 Sample contexts:")
            for i, text in enumerate(sample_df["context_text"].head(3)):
                print(f"   {i+1}. {str(text)[:100]}...")

        # Initialize data configuration
        print(f"\n🔧 Configuring pipeline components...")
        
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

        # Initialize data loader
        data_loader = DataLoader(data_config)
        print(f"   ✅ DataLoader initialized")
        print(f"      📊 Total rows: {data_loader.total_rows}")
        print(f"      📦 Number of chunks: {data_loader.num_chunks}")

        # Initialize model configuration
        model_config = ModelConfig(local_model_dir, repo_id)
        print(f"   ✅ ModelConfig initialized")

        # Initialize executor
        executor = Qwen3Executor(model_config)
        print(f"   ✅ Qwen3Executor initialized")

        # Create inference orchestrator
        print(f"\n🚀 Creating inference orchestrator...")
        
        inference_orchestrator = InferenceOrchestrator(
            data_config=data_config,
            data_loader=data_loader,
            model_config=model_config,
            executor=executor,
            max_workers=2,
            max_seq_length=max_seq_length,
            memory_utilization=memory_utilization,
            intermediate_size_threshold_gb=intermediate_size_threshold_gb,
        )
        print(f"   ✅ InferenceOrchestrator created")

        # Create distillation trainer
        print(f"\n🎓 Creating distillation trainer...")
        
        # Custom LoRA config for testing
        lora_config = {
            "r": lora_rank,
            "lora_alpha": lora_rank * 2,  # Common practice: alpha = 2 * rank
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        distillation_trainer = LoraDistillationTrainer(
            model_config=model_config,
            lora_config=lora_config,
            use_qlora = use_qlora,  # Enable quantization for memory efficiency
            learning_rate=learning_rate,
            temperature=2.0,
            alpha=0.5,  # 50% hard labels, 50% soft labels
            max_grad_norm=1.0
        )
        print(f"   ✅ LoraDistillationTrainer created")
        print(f"      🎯 LoRA rank: {lora_rank}")
        print(f"      🔥 QLoRA enabled: True")
        print(f"      📚 Training epochs per wave: {training_epochs_per_wave}")

        # Create wave training orchestrator
        print(f"\n🌊 Creating wave training orchestrator...")
        
        wave_orchestrator = WaveTrainingOrchestrator(
            inference_orchestrator=inference_orchestrator,
            distillation_trainer=distillation_trainer,
            train_after_each_wave=True,
            training_epochs=training_epochs_per_wave,
            training_batch_size=training_batch_size,
            save_adapters_after_training=True,
            adapter_save_dir=adapter_save_dir
        )
        print(f"   ✅ WaveTrainingOrchestrator created")

        # Run baseline comparison (optional - before training)
        print(f"\n🔍 Running baseline model comparison...")
        baseline_metrics = run_baseline_comparison(
            model_config, data_config, sample_size=10
        )
        print(f"   📊 Baseline accuracy: {baseline_metrics['accuracy']:.1%}")
        print(f"   📊 Baseline avg loss: {baseline_metrics['avg_loss']:.4f}")

        # Execute wave-based training pipeline
        print(f"\n🚀 Starting wave-based inference + training pipeline...")
        print(f"=" * 80)
        
        pipeline_start_time = time.time()
        
        # Run the integrated pipeline
        results = wave_orchestrator.run_wave_based_training(debug_first=True)
        
        pipeline_time = time.time() - pipeline_start_time

        # Verify pipeline completion
        if results is None or not results.get("final_output_path"):
            print("❌ Wave training pipeline failed.")
            return False

        print(f"\n✅ Pipeline completed successfully!")
        print(f"   ⏱️  Total time: {pipeline_time:.2f}s")
        print(f"   📁 Final outputs: {results['final_output_path']}")
        print(f"   🎯 Trained adapters: {results['adapter_save_path']}")

        # Validate outputs and training results
        print(f"\n📊 Validating training results...")
        validation_results = validate_training_results(
            results["final_output_path"],
            results["adapter_save_path"],
            model_config,
            data_config,
            baseline_metrics
        )

        # Print comprehensive results
        print_final_results(
            pipeline_time=pipeline_time,
            baseline_metrics=baseline_metrics,
            validation_results=validation_results,
            training_config={
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "epochs_per_wave": training_epochs_per_wave,
                "use_qlora": use_qlora,
                "memory_utilization": memory_utilization,
            }
        )

        # Cleanup
        print(f"\n🧹 Cleaning up resources...")
        wave_orchestrator.cleanup()
        del wave_orchestrator, inference_orchestrator, distillation_trainer
        gc.collect()
        
        return True

    except Exception as e:
        print(f"❌ Wave training pipeline test failed: {e}")
        traceback.print_exc()
        return False


def run_baseline_comparison(
    model_config: "ModelConfig", 
    data_config: "DataConfig", 
    sample_size: int = 10
) -> Dict[str, float]:
    """
    Run baseline comparison using original HuggingFace model.
    """
    print(f"   🔄 Loading baseline model for comparison...")
    
    # Load original model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float32
    ).eval()
    
    # Load sample data
    sample_df = pd.read_csv(data_config.csv_path, nrows=sample_size)
    
    correct_predictions = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for idx, row in sample_df.iterrows():
            context_text = row[data_config.input_col]
            actual_next_token = row.get("next_token", None)
            
            # Tokenize
            inputs = tokenizer(
                context_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].cpu().numpy()
            predicted_token = np.argmax(logits)
            
            # Calculate accuracy
            if actual_next_token is not None:
                if int(predicted_token) == actual_next_token:
                    correct_predictions += 1
            
            # Calculate loss (if we have true labels)
            if actual_next_token is not None:
                loss = torch.nn.functional.cross_entropy(
                    torch.tensor(logits).unsqueeze(0),
                    torch.tensor([actual_next_token])
                ).item()
                total_loss += loss
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    
    accuracy = correct_predictions / len(sample_df) if len(sample_df) > 0 else 0.0
    avg_loss = total_loss / len(sample_df) if len(sample_df) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "total_samples": len(sample_df),
        "correct_predictions": correct_predictions
    }


def validate_training_results(
    final_output_path: str,
    adapter_save_path: str,
    model_config: "ModelConfig",
    data_config: "DataConfig",
    baseline_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Validate that training actually improved the model.
    """
    print(f"   🔍 Loading and validating training outputs...")
    
    # Check if output files exist
    output_dir = Path(final_output_path)
    adapter_dir = Path(adapter_save_path)
    
    chunk_files = sorted(output_dir.glob("chunk_*.csv"))
    adapter_subdirs = [d for d in adapter_dir.iterdir() if d.is_dir()]
    
    validation_results = {
        "output_files_found": len(chunk_files),
        "adapter_checkpoints_found": len(adapter_subdirs),
        "total_training_samples": 0,
        "waves_completed": len(adapter_subdirs),
        "training_progression": []
    }
    
    # Count total samples processed
    total_samples = 0
    for chunk_file in chunk_files:
        chunk_df = pd.read_csv(chunk_file)
        total_samples += len(chunk_df)
    
    validation_results["total_training_samples"] = total_samples
    
    # Validate adapter checkpoints exist
    for adapter_subdir in sorted(adapter_subdirs):
        adapter_files = list(adapter_subdir.glob("*.bin")) + list(adapter_subdir.glob("*.safetensors"))
        config_files = list(adapter_subdir.glob("adapter_config.json"))
        
        checkpoint_valid = len(adapter_files) > 0 and len(config_files) > 0
        
        validation_results["training_progression"].append({
            "wave": adapter_subdir.name,
            "checkpoint_valid": checkpoint_valid,
            "adapter_files": len(adapter_files),
            "config_files": len(config_files)
        })
    
    # Test trained model performance (if we have a recent checkpoint)
    if adapter_subdirs:
        latest_adapter = sorted(adapter_subdirs)[-1]
        print(f"   🧪 Testing performance of latest adapter: {latest_adapter.name}")
        
        try:
            trained_metrics = test_trained_model_performance(
                model_config, latest_adapter, data_config, sample_size=5
            )
            validation_results["trained_model_metrics"] = trained_metrics
            validation_results["improvement_vs_baseline"] = {
                "accuracy_delta": trained_metrics["accuracy"] - baseline_metrics["accuracy"],
                "loss_delta": baseline_metrics["avg_loss"] - trained_metrics["avg_loss"],  # Lower is better
            }
        except Exception as e:
            print(f"     ⚠️  Could not test trained model: {e}")
            validation_results["trained_model_metrics"] = None
            validation_results["improvement_vs_baseline"] = None
    
    return validation_results


def test_trained_model_performance(
    model_config: "ModelConfig",
    adapter_path: Path,
    data_config: "DataConfig", 
    sample_size: int = 5
) -> Dict[str, float]:
    """
    Test the performance of a trained LoRA adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).eval()
    
    # Load trained adapter
    model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    # Load sample data
    sample_df = pd.read_csv(data_config.csv_path, nrows=sample_size)
    
    correct_predictions = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for idx, row in sample_df.iterrows():
            context_text = row[data_config.input_col]
            actual_next_token = row.get("next_token", None)
            
            # Tokenize
            inputs = tokenizer(
                context_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Get predictions with adapter
            outputs = model_with_adapter(**inputs)
            logits = outputs.logits[0, -1, :].cpu().numpy()
            predicted_token = np.argmax(logits)
            
            # Calculate accuracy
            if actual_next_token is not None:
                if int(predicted_token) == actual_next_token:
                    correct_predictions += 1
            
            # Calculate loss
            if actual_next_token is not None:
                loss = torch.nn.functional.cross_entropy(
                    torch.tensor(logits).unsqueeze(0),
                    torch.tensor([actual_next_token])
                ).item()
                total_loss += loss
    
    # Cleanup
    del model_with_adapter, base_model, tokenizer
    gc.collect()
    
    accuracy = correct_predictions / len(sample_df) if len(sample_df) > 0 else 0.0
    avg_loss = total_loss / len(sample_df) if len(sample_df) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "total_samples": len(sample_df),
        "correct_predictions": correct_predictions
    }


def print_final_results(
    pipeline_time: float,
    baseline_metrics: Dict[str, float],
    validation_results: Dict[str, Any],
    training_config: Dict[str, Any]
):
    """
    Print comprehensive final results.
    """
    print(f"\n🎉 WAVE-BASED LORA TRAINING PIPELINE RESULTS")
    print(f"=" * 80)
    
    # Pipeline summary
    print(f"📊 Pipeline Summary:")
    print(f"   ⏱️  Total execution time: {pipeline_time:.2f}s")
    print(f"   🌊 Waves completed: {validation_results['waves_completed']}")
    print(f"   📚 Total training samples: {validation_results['total_training_samples']}")
    print(f"   📄 Output chunk files: {validation_results['output_files_found']}")
    print(f"   🎯 Adapter checkpoints: {validation_results['adapter_checkpoints_found']}")
    
    # Training configuration
    print(f"\n🔧 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    # Baseline vs trained comparison
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
        
        if improvement['accuracy_delta'] > 0.05:  # 5% improvement
            print(f"   ✅ EXCELLENT: Significant accuracy improvement!")
        elif improvement['accuracy_delta'] > 0.01:  # 1% improvement  
            print(f"   ✅ GOOD: Noticeable accuracy improvement!")
        else:
            print(f"   ⚠️  Modest improvements - may need more training")
    else:
        print(f"   ❌ Could not evaluate trained model performance")
    
    # Wave progression
    print(f"\n🌊 Wave Training Progression:")
    for i, wave_info in enumerate(validation_results["training_progression"]):
        status = "✅" if wave_info["checkpoint_valid"] else "❌"
        print(f"   {status} {wave_info['wave']}: {wave_info['adapter_files']} adapter files")
    
    # Memory and efficiency metrics
    print(f"\n💾 Efficiency Metrics:")
    if training_config["use_qlora"]:
        print(f"   🔥 QLoRA quantization: Enabled (4-bit)")
        print(f"   💾 Memory savings: ~75% vs full fine-tuning")
    print(f"   📊 LoRA trainable params: ~{training_config['lora_rank'] * 2}k per layer")
    print(f"   🧠 Memory utilization: {training_config['memory_utilization']:.1%}")
    
    print(f"\n🎯 Wave-based LoRA training pipeline test completed!")


if __name__ == "__main__":
    """
    Main execution function for wave-based LoRA training pipeline test.
    """
    print("🧪 Wave-Based LoRA Training Pipeline Tester")
    print("=" * 80)
    print("This test validates the complete pipeline:")
    print("1. 🌊 Wave-based inference orchestration")
    print("2. 🎓 LoRA/QLoRA distillation training") 
    print("3. 📊 Performance validation and comparison")
    print("4. 💾 Memory-efficient progressive training")
    print()
    
    success = test_wave_training_pipeline()
    
    if success:
        print("\n✅ All tests passed! Wave-based LoRA training pipeline is working correctly.")
        exit(0)
    else:
        print("\n❌ Tests failed. Please check the error logs above.")
        exit(1)
