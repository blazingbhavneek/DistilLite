from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from base import BaseExecutor, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Executor(BaseExecutor):
    """Qwen3-specific implementation of BaseExecutor using official Qwen3 modeling components"""

    def __init__(self, config: ModelConfig, cache_position_embeddings: bool = True):
        super().__init__(config)
        self._rotary_emb = None
        self._cache_position_embeddings = cache_position_embeddings
        self._position_embeddings_cache = {} if cache_position_embeddings else None

        # Initialize rotary embeddings once
        self._initialize_rotary_embeddings()

    def _initialize_rotary_embeddings(self):
        """Initialize Qwen3 rotary embeddings"""
        try:
            # Import Qwen3 components
            from transformers.models.qwen3.modeling_qwen3 import \
                Qwen3RotaryEmbedding

            # Create rotary embedding using config
            self._rotary_emb = Qwen3RotaryEmbedding(config=self.config.config)
        except ImportError as e:
            self._rotary_emb = None

    def clear_position_embeddings_cache(self):
        """Clear position embeddings cache - useful for thread safety or memory management"""
        if self._position_embeddings_cache is not None:
            self._position_embeddings_cache.clear()

    def _compute_position_embeddings(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute position embeddings for the current batch"""
        if self._rotary_emb is None:
            raise RuntimeError("Rotary embeddings not initialized")

        # Get input dimensions
        if "hidden_states" in batch:
            hidden_states = batch["hidden_states"]
            device = hidden_states.device
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        elif "input_ids" in batch:
            input_ids = batch["input_ids"]
            device = input_ids.device
            batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        else:
            raise ValueError("Batch must contain either 'hidden_states' or 'input_ids'")

        # Get or create position_ids
        position_ids = batch.get("position_ids")
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Use caching if enabled
        if (
            self._cache_position_embeddings
            and self._position_embeddings_cache is not None
        ):
            # Create cache key
            cache_key = (
                device.type,
                batch_size,
                seq_len,
                (
                    tuple(position_ids.flatten().tolist())
                    if position_ids.numel() < 100
                    else None
                ),
            )

            # Check cache first
            if cache_key in self._position_embeddings_cache:
                return self._position_embeddings_cache[cache_key]

        # Compute position embeddings using Qwen3RotaryEmbedding
        with torch.no_grad():
            # Create dummy tensor for rotary embedding computation
            dummy_tensor = torch.zeros(
                batch_size,
                seq_len,
                self.config.config.hidden_size,
                device=device,
                dtype=torch.float32,
            )
            cos, sin = self._rotary_emb(dummy_tensor, position_ids)

        # Cache the result if caching is enabled (limit cache size)
        if (
            self._cache_position_embeddings
            and self._position_embeddings_cache is not None
            and len(self._position_embeddings_cache) < 100
        ):
            cache_key = (
                device.type,
                batch_size,
                seq_len,
                (
                    tuple(position_ids.flatten().tolist())
                    if position_ids.numel() < 100
                    else None
                ),
            )
            self._position_embeddings_cache[cache_key] = (cos, sin)

        return cos, sin

    def create_layer(
        self, layer_idx: int, state_dict: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Create a Qwen3 layer from state_dict using official components"""
        layer_type = self._get_layer_type(layer_idx)

        try:
            if layer_type == "embedding":
                return self._create_embedding_layer(state_dict)
            elif layer_type == "transformer":
                return self._create_transformer_layer(layer_idx, state_dict)
            elif layer_type == "norm":
                return self._create_norm_layer(state_dict)
            elif layer_type == "lm_head":
                return self._create_lm_head_layer(state_dict)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {layer_type} layer {layer_idx}: {str(e)}"
            ) from e

    def _create_embedding_layer(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create embedding layer using nn.Embedding"""
        vocab_size = self.config.config.vocab_size
        hidden_size = self.config.config.hidden_size
        padding_idx = getattr(self.config.config, "pad_token_id", None)

        embed_layer = nn.Embedding(vocab_size, hidden_size, padding_idx)

        # Load weights
        for param_name, param_tensor in state_dict.items():
            if param_name.endswith(".weight"):
                set_module_tensor_to_device(
                    embed_layer, "weight", "cpu", value=param_tensor
                )
                break

        return embed_layer

    def _create_transformer_layer(
        self, layer_idx: int, state_dict: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Create transformer layer using Qwen3DecoderLayer"""
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        # Extract relative layer index (subtract 1 for embedding layer)
        relative_layer_idx = layer_idx - 1

        # Create layer with empty weights
        with init_empty_weights():
            layer = Qwen3DecoderLayer(self.config.config, relative_layer_idx)

        # Load weights
        layer_name = self.config.get_layer_name_by_index(layer_idx)
        for param_name, param_tensor in state_dict.items():
            if param_name.startswith(layer_name + "."):
                relative_param_name = param_name[len(layer_name) + 1 :]

                # Navigate to the correct parameter
                param_parts = relative_param_name.split(".")
                current_module = layer

                try:
                    # Navigate through nested modules
                    for part in param_parts[:-1]:
                        current_module = getattr(current_module, part)

                    # Set the final parameter
                    param_attr = param_parts[-1]
                    if hasattr(current_module, param_attr):
                        set_module_tensor_to_device(
                            current_module, param_attr, "cpu", value=param_tensor
                        )
                except AttributeError:
                    continue

        return layer

    def _create_norm_layer(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create normalization layer using Qwen3RMSNorm"""
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        hidden_size = self.config.config.hidden_size
        rms_norm_eps = getattr(self.config.config, "rms_norm_eps", 1e-6)

        norm_layer = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

        # Load weights
        for param_name, param_tensor in state_dict.items():
            if param_name.endswith(".weight"):
                set_module_tensor_to_device(
                    norm_layer, "weight", "cpu", value=param_tensor
                )
                break

        return norm_layer

    def _create_lm_head_layer(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create language model head using nn.Linear"""
        hidden_size = self.config.config.hidden_size
        vocab_size = self.config.config.vocab_size

        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Load weights
        weight_tensor = None
        if "lm_head.weight" in state_dict:
            weight_tensor = state_dict["lm_head.weight"]
        else:
            weight_tensor = list(state_dict.values())[0]

        if weight_tensor is not None:
            set_module_tensor_to_device(lm_head, "weight", "cpu", value=weight_tensor)

        return lm_head

    def execute_layer(
        self, layer: nn.Module, layer_idx: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute a single Qwen3 layer"""
        layer_type = self._get_layer_type(layer_idx)

        if layer_type == "embedding":
            return self._execute_embedding_layer(layer, batch)
        elif layer_type == "transformer":
            return self._execute_transformer_layer(layer, batch)
        elif layer_type == "norm":
            return self._execute_norm_layer(layer, batch)
        elif layer_type == "lm_head":
            return self._execute_lm_head_layer(layer, batch)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _execute_embedding_layer(
        self, layer: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute embedding layer"""
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Embedding layer requires 'input_ids' in batch")
        return layer(input_ids)

    def _execute_transformer_layer(
        self, layer: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute Qwen3 transformer layer with proper position embeddings"""
        hidden_states = batch.get("hidden_states")
        attention_mask = batch.get("attention_mask")
        position_ids = batch.get("position_ids")

        if hidden_states is None:
            raise ValueError("Transformer layer requires 'hidden_states' in batch")

        # Compute position embeddings
        position_embeddings = self._compute_position_embeddings(batch)

        # Prepare attention mask for Qwen3 format
        if attention_mask is not None:
            # Convert to proper device and dtype
            attention_mask = attention_mask.to(device=hidden_states.device)

            # Ensure proper dtype for attention mask
            if attention_mask.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                # Convert integer masks to boolean or float
                if attention_mask.min() >= 0 and attention_mask.max() <= 1:
                    attention_mask = attention_mask.to(torch.bool)
                else:
                    attention_mask = attention_mask.to(hidden_states.dtype)

            # Create causal mask mapping as expected by Qwen3DecoderLayer
            if isinstance(attention_mask, dict):
                causal_mask_mapping = attention_mask
            else:
                # Create the causal mask format that Qwen3 expects
                from transformers.models.qwen3.modeling_qwen3 import \
                    create_causal_mask

                # Prepare mask arguments
                batch_size, seq_len = hidden_states.shape[:2]
                cache_position = torch.arange(seq_len, device=hidden_states.device)

                try:
                    causal_mask = create_causal_mask(
                        config=self.config.config,
                        input_embeds=hidden_states,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=None,
                        position_ids=position_ids,
                    )
                    causal_mask_mapping = {"full_attention": causal_mask}
                except Exception as e:
                    # Fallback: use the original attention mask
                    causal_mask_mapping = {"full_attention": attention_mask}

            # Get the appropriate mask for this layer
            layer_attention_type = getattr(layer, "attention_type", "full_attention")
            final_attention_mask = causal_mask_mapping.get(
                layer_attention_type,
                causal_mask_mapping.get("full_attention", attention_mask),
            )
        else:
            final_attention_mask = None

        # Execute the layer with Qwen3-specific arguments
        try:
            # Try with full Qwen3DecoderLayer signature
            output = layer(
                hidden_states=hidden_states,
                attention_mask=final_attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
        except TypeError as e:
            try:
                # Fallback: try with minimal required arguments
                output = layer(
                    hidden_states,
                    attention_mask=final_attention_mask,
                    position_embeddings=position_embeddings,
                )
            except Exception as e2:
                # Last resort: just hidden states
                output = layer(hidden_states)

        # Handle tuple returns
        if isinstance(output, tuple):
            return output[0]  # Return hidden states only
        return output

    def _execute_norm_layer(
        self, layer: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute normalization layer"""
        hidden_states = batch.get("hidden_states")
        if hidden_states is None:
            raise ValueError("Norm layer requires 'hidden_states' in batch")
        return layer(hidden_states)

    def _execute_lm_head_layer(
        self, layer: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute language model head layer"""
        hidden_states = batch.get("hidden_states")
        if hidden_states is None:
            raise ValueError("LM head layer requires 'hidden_states' in batch")
        return layer(hidden_states)


if __name__ == "__main__":
    repo_id = "Qwen/Qwen3-0.6B"
    local_dir = "models"
    max_new_tokens = 10

    # --- Init config + executor ---
    config = ModelConfig(local_dir, repo_id)
    executor = Qwen3Executor(config)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_prompts = [
        "The future of artificial intelligence depends on",
        "In the next decade, machine learning will",
    ]

    inputs = tokenizer(
        test_prompts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    input_ids_initial = inputs["input_ids"]
    attention_mask_initial = inputs["attention_mask"]

    group_size = 4
    layer_ranges = [
        (i, min(i + group_size - 1, config.total_layers - 1))
        for i in range(0, config.total_layers, group_size)
    ]

    # Pre-load all layer ranges (simulating external pipeline handling memory)
    loaded_layer_ranges = {}
    print("Loading all layer ranges...")
    for start_idx, end_idx in layer_ranges:
        print(f"Loading layers {start_idx}-{end_idx}...")
        loaded_layer_ranges[(start_idx, end_idx)] = executor.load_layer_range(
            start_idx, end_idx
        )

    # -------------------------------
    # Phase 1: Custom Executor run with pre-loaded layers
    # -------------------------------
    custom_logits_list = []
    custom_tokens_list = []

    seq_ids = input_ids_initial.clone()
    attn_mask = attention_mask_initial.clone()

    print("\n--- Phase 1: Custom Executor with Pre-loaded Layers ---")
    for step in range(max_new_tokens):
        pos_ids = (
            torch.arange(seq_ids.shape[1], dtype=torch.long)
            .unsqueeze(0)
            .expand(seq_ids.shape[0], -1)
        )

        batch_step = {
            "input_ids": seq_ids.clone(),
            "attention_mask": attn_mask.clone(),
            "position_ids": pos_ids,
        }

        final_logits = None
        for start_idx, end_idx in layer_ranges:
            # Use pre-loaded layers
            loaded_layers = loaded_layer_ranges[(start_idx, end_idx)]

            final_logits = executor.execute_layer_range(
                loaded_layers=loaded_layers,
                batch=batch_step,
                validate_shapes=False,
            )
            if end_idx < config.total_layers - 1:
                batch_step["hidden_states"] = final_logits
            if "input_ids" in batch_step and start_idx == 0:
                del batch_step["input_ids"]

        last_logits = final_logits[:, -1, :]
        next_tokens = torch.argmax(last_logits, dim=-1)

        custom_logits_list.append(last_logits.cpu())
        custom_tokens_list.append(next_tokens.cpu())

        seq_ids = torch.cat([seq_ids, next_tokens.unsqueeze(1)], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones_like(next_tokens).unsqueeze(1)], dim=1
        )

    # Clean up pre-loaded layers (simulating external pipeline cleanup)
    print("Cleaning up pre-loaded layers...")
    del loaded_layer_ranges
    import gc

    gc.collect()

    # -------------------------------
    # Phase 2: HF run
    # -------------------------------
    hf_logits_list = []
    hf_tokens_list = []

    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True
    ).eval()

    seq_ids = input_ids_initial.clone()
    attn_mask = attention_mask_initial.clone()

    print("\n--- Phase 2: HF Model ---")
    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = hf_model(
                input_ids=seq_ids, attention_mask=attn_mask, use_cache=False
            ).logits

        last_logits = logits[:, -1, :]
        next_tokens = torch.argmax(last_logits, dim=-1)

        hf_logits_list.append(last_logits.cpu())
        hf_tokens_list.append(next_tokens.cpu())

        seq_ids = torch.cat([seq_ids, next_tokens.unsqueeze(1)], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones_like(next_tokens).unsqueeze(1)], dim=1
        )

    # -------------------------------
    # Phase 3: Compare
    # -------------------------------
    print("\n--- Phase 3: Comparison ---")
    for step in range(max_new_tokens):
        logits_diff = torch.max(
            torch.abs(custom_logits_list[step] - hf_logits_list[step])
        ).item()
        tokens_match = torch.equal(custom_tokens_list[step], hf_tokens_list[step])

        print(f"Step {step+1}:")
        print(f"  Max Logit Diff: {logits_diff:.6f}")
        print(f"  Tokens Match: {tokens_match}")
        for i in range(len(test_prompts)):
            print(f"   Prompt {i+1}:")
            print(
                f"     Custom: {custom_tokens_list[step][i].item()} '{tokenizer.decode([custom_tokens_list[step][i].item()])}'"
            )
            print(
                f"     HF:     {hf_tokens_list[step][i].item()} '{tokenizer.decode([hf_tokens_list[step][i].item()])}'"
            )
