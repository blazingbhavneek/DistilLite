import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from distillite.data import DataConfig
from distillite.model import ModelConfig


class LoraDistillationTrainer:
    """
    Handles LoRA/QLoRA knowledge distillation training using precomputed teacher logits.
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
    ):
        """
        Initialize distillation trainer.

        Args:
            model_config: Model configuration from orchestrator
            lora_config: LoRA configuration dict (rank, alpha, etc.)
            use_qlora: Whether to use quantized LoRA (4-bit)
            learning_rate: Learning rate for LoRA adapters
            temperature: Temperature for softmax during distillation
            alpha: Weight balance between hard labels (CE) and soft labels (KL)
            max_grad_norm: Gradient clipping threshold
        """
        self.model_config = model_config
        self.use_qlora = use_qlora
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm

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

        self.model = None
        self.optimizer = None
        self.tokenizer = None

        print(f"🎯 DistillationTrainer initialized:")
        print(f"   📊 LoRA rank: {self.lora_config['r']}")
        print(f"   🔥 QLoRA enabled: {self.use_qlora}")
        print(f"   🌡️  Temperature: {self.temperature}")
        print(f"   ⚖️  Loss balance (CE/KL): {self.alpha:.1f}/{1-self.alpha:.1f}")

    def prepare_model(self, tokenizer):
        """Prepare quantized model with LoRA adapters for training."""
        print("🔧 Preparing model for distillation training...")

        self.tokenizer = tokenizer

        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            model = prepare_model_for_kbit_training(model)
            print("   ✅ 4-bit quantization enabled")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            print("   ✅ Standard precision model loaded")

        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(model, lora_config)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"   📊 Total parameters: {total_params:,}")
        print(
            f"   🎯 Trainable parameters: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)"
        )
        print(f"   🔧 LoRA adapters ready for training")

        return self.model

    def train_on_wave_outputs(
        self,
        chunk_files: List[Path],
        data_config: "DataConfig",
        epochs: int = 3,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Train LoRA adapters using teacher logits from completed wave.

        Args:
            chunk_files: List of CSV files containing inference results
            data_config: Data configuration from orchestrator
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")

        print(f"\n🚀 Starting distillation training on {len(chunk_files)} chunks...")
        print(f"   📚 Epochs: {epochs}")
        print(f"   📦 Batch size: {batch_size}")

        training_data = []
        for chunk_file in chunk_files:
            chunk_df = pd.read_csv(chunk_file)
            training_data.append(chunk_df)

        combined_df = pd.concat(training_data, ignore_index=True)
        print(f"   📊 Total training samples: {len(combined_df)}")

        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_batches = 0

        self.model.train()

        for epoch in range(epochs):
            print(f"\n📖 Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            shuffled_df = combined_df.sample(frac=1.0).reset_index(drop=True)

            for batch_start in range(0, len(shuffled_df), batch_size):
                batch_end = min(batch_start + batch_size, len(shuffled_df))
                batch_df = shuffled_df.iloc[batch_start:batch_end]

                batch_loss, batch_kl, batch_ce = self._train_batch(
                    batch_df, data_config
                )

                total_loss += batch_loss
                total_kl_loss += batch_kl
                total_ce_loss += batch_ce
                total_batches += 1

                if total_batches % 10 == 0:
                    print(
                        f"   📊 Batch {total_batches}: Loss={batch_loss:.4f}, KL={batch_kl:.4f}, CE={batch_ce:.4f}"
                    )

            epoch_time = time.time() - epoch_start
            print(f"   ⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f}s")

        avg_loss = total_loss / total_batches
        avg_kl = total_kl_loss / total_batches
        avg_ce = total_ce_loss / total_batches

        metrics = {
            "avg_total_loss": avg_loss,
            "avg_kl_loss": avg_kl,
            "avg_ce_loss": avg_ce,
            "total_batches": total_batches,
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

        print(f"\n✅ Training completed!")
        print(f"   📊 Average total loss: {avg_loss:.4f}")
        print(f"   📊 Average KL loss: {avg_kl:.4f}")
        print(f"   📊 Average CE loss: {avg_ce:.4f}")

        return metrics

    def _train_batch(
        self, batch_df: pd.DataFrame, data_config: "DataConfig"
    ) -> Tuple[float, float, float]:
        """Train on a single batch of examples."""
        self.optimizer.zero_grad()

        input_texts = batch_df[data_config.input_col].tolist()
        teacher_logits_list = []

        for _, row in batch_df.iterrows():
            logits_str = row[data_config.output_col]
            if isinstance(logits_str, str):
                logits_clean = logits_str.strip("[]")
                logits = np.array([float(x.strip()) for x in logits_clean.split(",")])
            else:
                logits = np.array(logits_str)
            teacher_logits_list.append(torch.tensor(logits, dtype=torch.float32))

        tokenized = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        teacher_logits = torch.stack(teacher_logits_list).to(self.model.device)

        outputs = self.model(**tokenized)
        student_logits = outputs.logits[:, -1, :]

        T = self.temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            T * T
        )

        teacher_tokens = torch.argmax(teacher_logits, dim=-1)
        ce_loss = F.cross_entropy(student_logits, teacher_tokens)

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return total_loss.item(), kl_loss.item(), ce_loss.item()

    def save_trained_model(self, output_path: str):
        """Save the trained LoRA adapters."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        self.model.save_pretrained(output_path)
        print(f"💾 Trained LoRA adapters saved to: {output_path}")

    def cleanup(self):
        """Clean up model and optimizer to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 DistillationTrainer cleaned up")
