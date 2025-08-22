import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

# Import handling with fallbacks
try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False

try:
    import bitsandbytes as bnb

    BITSANDBYTES_LIB_AVAILABLE = True
except ImportError:
    BITSANDBYTES_LIB_AVAILABLE = False

try:
    from peft import (LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training)

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from distillite.data import DataConfig
from distillite.model import ModelConfig


class LoraDistillationTrainer:
    """
    GPU/CPU compatible LoRA/QLoRA knowledge distillation trainer using precomputed teacher logits.
    Designed to work with wave-based processing from InferenceOrchestrator.
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        lora_config: Optional[Dict] = None,
        use_qlora: bool = True,
        learning_rate: float = 5e-5,
        temperature: float = 2.0,
        alpha: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "auto",
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
    ):
        """
        Initialize distillation trainer with GPU optimizations and automatic fallbacks.

        Args:
            model_config: Model configuration from orchestrator
            lora_config: LoRA configuration dict (rank, alpha, etc.)
            use_qlora: Whether to use quantized LoRA (4-bit) - will fallback if unavailable
            learning_rate: Learning rate for LoRA adapters
            temperature: Temperature for softmax during distillation
            alpha: Weight balance between hard labels (CE) and soft labels (KL)
            max_grad_norm: Gradient clipping threshold
            device: Device to use ('cuda', 'cpu', or 'auto')
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use automatic mixed precision
        """
        # Check dependencies
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for LoRA training. Install with: pip install peft"
            )

        self.model_config = model_config
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Handle device and precision setup
        self.device = self._setup_device(device)
        self.mixed_precision = mixed_precision and self.device.startswith("cuda")

        # Handle QLoRA capability
        self.use_qlora = (
            use_qlora and BITSANDBYTES_AVAILABLE and BITSANDBYTES_LIB_AVAILABLE
        )
        if use_qlora and not (BITSANDBYTES_AVAILABLE and BITSANDBYTES_LIB_AVAILABLE):
            print(
                "⚠️  QLoRA requested but bitsandbytes not available - falling back to standard LoRA"
            )

        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Initialize training components
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.scaler = None

        print(f"🎯 DistillationTrainer initialized:")
        print(f"   🎮 Device: {self.device}")
        if self.device.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            )
            print(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"   📊 LoRA rank: {self.lora_config['r']}")
        print(f"   🔥 QLoRA enabled: {self.use_qlora}")
        print(f"   🌡️  Temperature: {self.temperature}")
        print(f"   ⚖️  Loss balance (CE/KL): {self.alpha:.1f}/{1-self.alpha:.1f}")
        print(f"   🎯 Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   ⚡ Mixed precision: {self.mixed_precision}")

    def _setup_device(self, device: str) -> str:
        """Setup and validate device for training."""
        if device == "auto":
            if torch.cuda.is_available():
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

        return {"total": total, "allocated": allocated, "cached": cached, "free": free}

    def _clear_gpu_memory(self):
        """Clear GPU memory and cache."""
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()

    def _create_quantization_config(self):
        """Create quantization config if QLoRA is enabled and available."""
        if not self.use_qlora or not BITSANDBYTES_AVAILABLE:
            return None

        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if self.device.startswith("cuda") else torch.float32
                ),
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"   ⚠️  Failed to create quantization config: {e}")
            self.use_qlora = False
            return None

    def prepare_model(self, tokenizer):
        """Prepare quantized model with LoRA adapters for training with GPU optimization."""
        print("🔧 Preparing model for distillation training...")

        self.tokenizer = tokenizer

        if self.device.startswith("cuda"):
            mem_before = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory before loading: {mem_before['allocated']:.1f}GB")

        # Create quantization config if using QLoRA
        bnb_config = self._create_quantization_config()

        # Model loading parameters
        model_kwargs = {
            "trust_remote_code": True,
        }

        if self.use_qlora and bnb_config is not None:
            # QLoRA setup
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = (
                "auto" if self.device == "cuda" else {"": self.device}
            )
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            )
            print("   🔥 Loading model with 4-bit quantization...")
        else:
            # Standard model setup
            model_kwargs["device_map"] = (
                "auto" if self.device == "cuda" else {"": self.device}
            )
            model_kwargs["torch_dtype"] = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
            print("   📊 Loading model in standard precision...")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path, **model_kwargs
            )

            # Prepare for k-bit training if using QLoRA
            if self.use_qlora and bnb_config is not None:
                model = prepare_model_for_kbit_training(model)
                print("   ✅ 4-bit quantization enabled")
            else:
                print("   ✅ Standard precision model loaded")

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🔄 Trying fallback configuration...")

            # Fallback: disable QLoRA and try again
            self.use_qlora = False
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

            # Move to device manually if needed
            if not self.device == "cpu":
                model = model.to(self.device)

            print("   ✅ Fallback model loaded successfully")

        if self.device.startswith("cuda"):
            mem_after = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory after loading: {mem_after['allocated']:.1f}GB")

        # Apply LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(model, lora_config)

        # Setup optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate)

        # Setup mixed precision scaler
        if self.mixed_precision and self.device.startswith("cuda"):
            self.scaler = torch.cuda.amp.GradScaler()
            print("   ⚡ Mixed precision scaler initialized")

        # Print parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"   📊 Total parameters: {total_params:,}")
        print(
            f"   🎯 Trainable parameters: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)"
        )
        print(f"   🔧 LoRA adapters ready for training")

        if self.device.startswith("cuda"):
            mem_final = self._get_gpu_memory_info()
            print(f"   🎮 Final GPU memory: {mem_final['allocated']:.1f}GB allocated")

        return self.model

    def train_on_wave_outputs(
        self,
        chunk_files: List[Path],
        data_config: "DataConfig",
        epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Train LoRA adapters using teacher logits with GPU optimization.

        Args:
            chunk_files: List of CSV files containing inference results
            data_config: Data configuration from orchestrator
            epochs: Number of training epochs
            batch_size: Training batch size (effective batch size = batch_size * gradient_accumulation_steps)
            max_length: Maximum sequence length

        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")

        print(f"\n🚀 Starting distillation training on {len(chunk_files)} chunks...")
        print(f"   📚 Epochs: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🎯 Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   📏 Max length: {max_length}")

        effective_batch_size = batch_size * self.gradient_accumulation_steps
        print(f"   📊 Effective batch size: {effective_batch_size}")

        # Load training data
        training_data = []
        for chunk_file in chunk_files:
            try:
                chunk_df = pd.read_csv(chunk_file)
                training_data.append(chunk_df)
            except Exception as e:
                print(f"   ⚠️  Error loading {chunk_file}: {e}")
                continue

        if not training_data:
            raise ValueError("No valid training data found")

        combined_df = pd.concat(training_data, ignore_index=True)
        print(f"   📊 Total training samples: {len(combined_df)}")

        if self.device.startswith("cuda"):
            mem_start = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory at training start: {mem_start['allocated']:.1f}GB")

        # Training loop
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_batches = 0

        self.model.train()

        for epoch in range(epochs):
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            # Shuffle data
            shuffled_df = combined_df.sample(frac=1.0).reset_index(drop=True)

            # Process batches
            for batch_start in range(0, len(shuffled_df), batch_size):
                batch_end = min(batch_start + batch_size, len(shuffled_df))
                batch_df = shuffled_df.iloc[batch_start:batch_end]

                try:
                    batch_loss, batch_kl, batch_ce = self._train_batch(
                        batch_df, data_config, max_length
                    )

                    total_loss += batch_loss
                    total_kl_loss += batch_kl
                    total_ce_loss += batch_ce
                    total_batches += 1

                    if total_batches % 10 == 0:
                        print(
                            f"   📊 Batch {total_batches}: Loss={batch_loss:.4f}, KL={batch_kl:.4f}, CE={batch_ce:.4f}"
                        )

                        if self.device.startswith("cuda") and total_batches % 50 == 0:
                            mem_current = self._get_gpu_memory_info()
                            print(
                                f"   🎮 GPU memory: {mem_current['allocated']:.1f}GB allocated"
                            )

                except Exception as e:
                    print(f"   ⚠️  Error in batch {total_batches}: {e}")
                    continue

            epoch_time = time.time() - epoch_start
            print(f"   ⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            # Clear cache between epochs
            self._clear_gpu_memory()

        # Calculate final metrics
        if total_batches > 0:
            avg_loss = total_loss / total_batches
            avg_kl = total_kl_loss / total_batches
            avg_ce = total_ce_loss / total_batches
        else:
            avg_loss = avg_kl = avg_ce = 0.0

        metrics = {
            "avg_total_loss": avg_loss,
            "avg_kl_loss": avg_kl,
            "avg_ce_loss": avg_ce,
            "total_batches": total_batches,
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_qlora": self.use_qlora,
        }

        print(f"\n✅ Training completed!")
        print(f"   📊 Average total loss: {avg_loss:.4f}")
        print(f"   📊 Average KL loss: {avg_kl:.4f}")
        print(f"   📊 Average CE loss: {avg_ce:.4f}")

        if self.device.startswith("cuda"):
            mem_final = self._get_gpu_memory_info()
            print(f"   🎮 Final GPU memory: {mem_final['allocated']:.1f}GB allocated")

        return metrics

    def _train_batch(
        self, batch_df: pd.DataFrame, data_config: "DataConfig", max_length: int = 512
    ) -> Tuple[float, float, float]:
        """Train on a single batch with GPU optimization and mixed precision."""

        # Initialize step counter and handle gradient accumulation
        if not hasattr(self, "_step_count"):
            self._step_count = 0

        if self._step_count % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Prepare batch data
        input_texts = batch_df[data_config.input_col].tolist()
        teacher_logits_list = []

        for _, row in batch_df.iterrows():
            logits_str = row[data_config.output_col]
            try:
                if isinstance(logits_str, str):
                    logits_clean = logits_str.strip("[]")
                    logits = np.array(
                        [float(x.strip()) for x in logits_clean.split(",")]
                    )
                else:
                    logits = np.array(logits_str)
                teacher_logits_list.append(torch.tensor(logits, dtype=torch.float32))
            except Exception as e:
                print(f"     ⚠️  Error parsing logits: {e}")
                # Use dummy logits as fallback
                vocab_size = 32000  # Approximate vocab size for most models
                dummy_logits = torch.zeros(vocab_size, dtype=torch.float32)
                teacher_logits_list.append(dummy_logits)

        if not teacher_logits_list:
            return 0.0, 0.0, 0.0

        # Tokenize inputs
        try:
            tokenized = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        except Exception as e:
            print(f"     ⚠️  Tokenization error: {e}")
            return 0.0, 0.0, 0.0

        # Move to device
        device = next(self.model.parameters()).device
        try:
            tokenized = {
                k: v.to(device, non_blocking=True) for k, v in tokenized.items()
            }
            teacher_logits = torch.stack(teacher_logits_list).to(
                device, non_blocking=True
            )
        except Exception as e:
            print(f"     ⚠️  Device transfer error: {e}")
            return 0.0, 0.0, 0.0

        # Forward pass
        try:
            if self.mixed_precision and self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**tokenized)
                    student_logits = outputs.logits[:, -1, :]

                    total_loss, kl_loss, ce_loss = self._compute_losses(
                        student_logits, teacher_logits
                    )

                    total_loss = total_loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**tokenized)
                student_logits = outputs.logits[:, -1, :]

                total_loss, kl_loss, ce_loss = self._compute_losses(
                    student_logits, teacher_logits
                )
                total_loss = total_loss / self.gradient_accumulation_steps
        except Exception as e:
            print(f"     ⚠️  Forward pass error: {e}")
            return 0.0, 0.0, 0.0

        # Backward pass
        try:
            if self.mixed_precision and self.device.startswith("cuda"):
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        except Exception as e:
            print(f"     ⚠️  Backward pass error: {e}")
            return 0.0, 0.0, 0.0

        self._step_count += 1

        # Optimizer step (with gradient accumulation)
        if self._step_count % self.gradient_accumulation_steps == 0:
            try:
                if self.mixed_precision and self.device.startswith("cuda"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
            except Exception as e:
                print(f"     ⚠️  Optimizer step error: {e}")
                return 0.0, 0.0, 0.0

        return (
            total_loss.item() * self.gradient_accumulation_steps,
            kl_loss.item(),
            ce_loss.item(),
        )

    def _compute_losses(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute KL divergence and cross-entropy losses."""
        T = self.temperature

        # Handle dimension mismatch
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[:, :min_vocab]
            teacher_logits = teacher_logits[:, :min_vocab]

        # KL divergence loss
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            T * T
        )

        # Cross-entropy loss
        teacher_tokens = torch.argmax(teacher_logits, dim=-1)
        ce_loss = F.cross_entropy(student_logits, teacher_tokens)

        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return total_loss, kl_loss, ce_loss

    def save_trained_model(self, output_path: str):
        """Save the trained LoRA adapters."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        print(f"💾 Saving trained LoRA adapters...")

        try:
            self.model.save_pretrained(output_path)

            # Save configuration
            config_path = Path(output_path) / "training_config.txt"
            with open(config_path, "w") as f:
                f.write(f"Device: {self.device}\n")
                f.write(f"Mixed Precision: {self.mixed_precision}\n")
                f.write(f"QLoRA: {self.use_qlora}\n")
                f.write(f"LoRA Config: {self.lora_config}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Alpha: {self.alpha}\n")
                f.write(f"Learning Rate: {self.learning_rate}\n")
                f.write(
                    f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}\n"
                )

            print(f"   ✅ Trained LoRA adapters saved to: {output_path}")
            print(f"   📋 Training config saved to: {config_path}")

        except Exception as e:
            print(f"   ❌ Error saving model: {e}")
            raise e

    def cleanup(self):
        """Clean up model and optimizer to free GPU memory."""
        print("🧹 Cleaning up DistillationTrainer...")

        if self.model is not None:
            del self.model
            self.model = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if self.scaler is not None:
            del self.scaler
            self.scaler = None

        self._clear_gpu_memory()

        if self.device.startswith("cuda"):
            mem_final = self._get_gpu_memory_info()
            print(f"   🎮 GPU memory after cleanup: {mem_final['allocated']:.1f}GB")

        print("   ✅ DistillationTrainer cleaned up")
