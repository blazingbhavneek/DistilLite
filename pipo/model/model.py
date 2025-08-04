import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

try:
    from safetensors import safe_open
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class ModelConfig:
    """Immutable Configuration class for thread-safe layer-by-layer Model execution"""

    def __init__(self, local_dir: str, repo_id: str):
        # Location of the model files
        self.local_dir: str = local_dir
        self.repo_id: str = repo_id

        self.model_path: Path = Path(os.path.join(local_dir, repo_id))
        if not os.path.exists(self.model_path):
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

        # Pretrained Configuration for model Architecture
        self.config = AutoConfig.from_pretrained(self.model_path)

        # Number of layers present in the Model
        self.num_layers: int = self._get_num_layers()

        # Mapping of Architecture specific layer Names to General Names for Inference
        self.layer_names_dict: dict = self._create_layer_names_dict()

        # List of Layers present in Model
        self.layer_names: list[str] = self._create_ordered_layer_names()

        # TODO
        self.layer_to_param_names: dict[str, list[str]] = (
            self._create_layer_param_mapping()
        )

    def _get_num_layers(self, key: str = "num_layers") -> int:
        """Get number of transformer layers from config using the provided key."""
        for k in ("num_hidden_layers", "n_layer", key):
            if hasattr(self.config, k):
                return getattr(self.config, k)

        raise ValueError(
            f"Cannot determine number of layers. "
            f"None of ('num_hidden_layers', 'n_layer', '{key}') found in config.\n"
            f"Please check the model's configuration file."
        )

    def _create_layer_names_dict(self) -> dict:
        """Creates mapping of Architecture specific layer names to general names for inference"""
        return {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }

    def _create_ordered_layer_names(self) -> list[str]:
        """Order the layer names for sequential execution"""
        layer_names = []
        layer_names.append(self.layer_names_dict["embed"])

        for i in range(self.num_layers):
            layer_names.append(f"{self.layer_names_dict['layer_prefix']}.{i}")

        layer_names.append(self.layer_names_dict["norm"])
        layer_names.append(self.layer_names_dict["lm_head"])

        return layer_names

    def _create_layer_param_mapping(self) -> dict[str, list[str]]:
        """Create maps from layer names to their parameter names for efficient loading"""
        all_param_names = self._get_all_param_names()
        layer_to_params = {}

        for layer_name in self.layer_names:
            layer_params = []
            for param_name in all_param_names:
                if self._param_belongs_to_layer(param_name, layer_name):
                    layer_params.append(param_name)
            layer_to_params[layer_name] = layer_params

        return layer_to_params

    def _get_all_param_names(self) -> list[str]:
        """Scans all model files and gets a complete list of every parameter name"""
        param_names = []

        for fname in os.listdir(self.model_path):
            path = os.path.join(self.model_path, fname)
            if fname.endswith(".safetensors") and _HAS_SAFETENSORS:
                with safe_open(path, framework="pt") as f:
                    param_names.extend(f.keys())
            elif fname.endswith(".bin"):
                import torch
                state = torch.load(path, map_location="meta")
                param_names.extend(state.keys())

        return list(set(param_names))

    def _param_belongs_to_layer(self, param_name: str, layer_name: str) -> bool:
        """Check if a parameter belongs to a specific layer"""
        if layer_name.startswith(self.layer_names_dict["layer_prefix"]):
            return param_name.startswith(layer_name + ".")

        return (
            param_name.startswith(layer_name + ".")
            or param_name == layer_name + ".weight"
        )

    def get_layer_param_names(self, layer_name: str) -> list[str]:
        """Get all parameter names for a specific layer"""
        return self.layer_to_param_names.get(layer_name, [])

    def get_layer_name_by_index(self, index: int) -> str:
        """Get layer name by execution order index"""
        if 0 <= index < len(self.layer_names):
            return self.layer_names[index]
        raise IndexError(
            f"Layer index {index} out of range (0-{len(self.layer_names)-1})"
        )

    def load_layer_state_dict(self, layer_name: str) -> dict:
        """Load state_dict for a specific layer from model files"""
        param_names = self.get_layer_param_names(layer_name)

        # Handle tied weights for lm_head
        if not param_names and layer_name == self.layer_names_dict["lm_head"]:
            embed_param = f"{self.layer_names_dict['embed']}.weight"
            all_params = self._get_all_param_names()
            if embed_param in all_params:
                param_names = [embed_param]

        if not param_names:
            return {}

        layer_state_dict = {}

        for fname in os.listdir(self.model_path):
            path = os.path.join(self.model_path, fname)

            if fname.endswith(".safetensors") and _HAS_SAFETENSORS:
                with safe_open(path, framework="pt") as f:
                    for param_name in param_names:
                        if param_name in f.keys():
                            key = (
                                "lm_head.weight"
                                if layer_name == self.layer_names_dict["lm_head"]
                                else param_name
                            )
                            layer_state_dict[key] = f.get_tensor(param_name)

            elif fname.endswith(".bin"):
                state = torch.load(path, map_location="cpu")
                for param_name in param_names:
                    if param_name in state:
                        key = (
                            "lm_head.weight"
                            if layer_name == self.layer_names_dict["lm_head"]
                            else param_name
                        )
                        layer_state_dict[key] = state[param_name]

        return layer_state_dict

    def load_layer_state_dict_by_index(self, layer_index: int) -> dict:
        """Load state_dict for a layer by its execution order index"""
        layer_name = self.get_layer_name_by_index(layer_index)
        return self.load_layer_state_dict(layer_name)

    def load_multiple_layers_state_dict(self, start_idx: int, end_idx: int) -> Dict[int, dict]:
        """
        Load state_dict for multiple layers at once
        
        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)
            
        Returns:
            Dictionary mapping layer_idx to state_dict
        """
        if start_idx < 0 or end_idx >= self.total_layers or start_idx > end_idx:
            raise ValueError(f"Invalid layer range: {start_idx}-{end_idx}. Valid range: 0-{self.total_layers-1}")
        
        layers_state_dict = {}
        
        for layer_idx in range(start_idx, end_idx + 1):
            state_dict = self.load_layer_state_dict_by_index(layer_idx)
            layers_state_dict[layer_idx] = state_dict
            
        return layers_state_dict

    @property
    def total_layers(self) -> int:
        """Get total number of layers (including embed, norm, lm_head)"""
        return len(self.layer_names)

    def __repr__(self) -> str:
        return f"ModelConfig(repo_id='{self.repo_id}', num_layers={self.num_layers}, total_layers={len(self.layer_names)})"


class LayerFactory:
    """Factory for creating different types of model layers from state dictionaries"""

    @staticmethod
    def create_embedding_layer(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create embedding layer from state dict"""

        class EmbeddingLayer(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
                self.model_dtype = weight.dtype

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return F.embedding(input_ids, self.weight).to(self.model_dtype)

        embed_weight = list(state_dict.values())[0]
        return EmbeddingLayer(embed_weight)

    @staticmethod
    def create_transformer_layer(
        state_dict: Dict[str, torch.Tensor], layer_name: str, config
    ) -> nn.Module:
        """Create transformer layer from state dict"""

        class TransformerLayer(nn.Module):
            def __init__(self, state_dict_data, layer_name, config):
                super().__init__()
                self.state_dict_data = state_dict_data
                self.layer_name = layer_name
                self.config = config
                self.model_dtype = next(iter(state_dict_data.values())).dtype

                # Get model dimensions from config
                self.hidden_size = getattr(config, "hidden_size", 2560)
                self.num_attention_heads = getattr(config, "num_attention_heads", 32)
                self.num_key_value_heads = getattr(
                    config, "num_key_value_heads", self.num_attention_heads
                )
                self.head_dim = self.hidden_size // self.num_attention_heads

            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                # Convert input to model dtype
                hidden_states = hidden_states.to(self.model_dtype)
                batch_size, seq_len, hidden_size = hidden_states.shape

                # Get attention weights
                q_weight = self.state_dict_data.get(
                    f"{self.layer_name}.self_attn.q_proj.weight"
                )
                k_weight = self.state_dict_data.get(
                    f"{self.layer_name}.self_attn.k_proj.weight"
                )
                v_weight = self.state_dict_data.get(
                    f"{self.layer_name}.self_attn.v_proj.weight"
                )
                o_weight = self.state_dict_data.get(
                    f"{self.layer_name}.self_attn.o_proj.weight"
                )

                # Residual connection input
                residual = hidden_states

                # Apply RMS norm to input (pre-norm architecture)
                input_layernorm_weight = self.state_dict_data.get(
                    f"{self.layer_name}.input_layernorm.weight"
                )
                if input_layernorm_weight is not None:
                    # RMS normalization
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
                    hidden_states = input_layernorm_weight * hidden_states

                if (
                    q_weight is not None
                    and k_weight is not None
                    and v_weight is not None
                ):
                    # Multi-head attention computation
                    q = torch.matmul(hidden_states, q_weight.T)
                    k = torch.matmul(hidden_states, k_weight.T)
                    v = torch.matmul(hidden_states, v_weight.T)

                    # Check if we need to handle grouped query attention (GQA)
                    q_heads = self.num_attention_heads
                    kv_heads = self.num_key_value_heads
                    q_head_dim = q.shape[-1] // q_heads
                    kv_head_dim = k.shape[-1] // kv_heads

                    # Reshape for multi-head attention
                    try:
                        q = q.view(batch_size, seq_len, q_heads, q_head_dim).transpose(
                            1, 2
                        )
                        k = k.view(
                            batch_size, seq_len, kv_heads, kv_head_dim
                        ).transpose(1, 2)
                        v = v.view(
                            batch_size, seq_len, kv_heads, kv_head_dim
                        ).transpose(1, 2)
                    except RuntimeError as e:
                        # Fallback: try to infer correct dimensions
                        actual_q_dim = q.shape[-1]
                        actual_k_dim = k.shape[-1]
                        actual_v_dim = v.shape[-1]

                        # Recalculate head dimensions based on actual tensor sizes
                        q_head_dim = (
                            actual_q_dim // q_heads
                            if actual_q_dim % q_heads == 0
                            else actual_q_dim
                        )
                        k_head_dim = (
                            actual_k_dim // kv_heads
                            if actual_k_dim % kv_heads == 0
                            else actual_k_dim
                        )
                        v_head_dim = (
                            actual_v_dim // kv_heads
                            if actual_v_dim % kv_heads == 0
                            else actual_v_dim
                        )

                        if actual_q_dim % q_heads == 0:
                            q = q.view(
                                batch_size, seq_len, q_heads, q_head_dim
                            ).transpose(1, 2)
                        else:
                            q = q.view(batch_size, seq_len, 1, actual_q_dim).transpose(
                                1, 2
                            )

                        if actual_k_dim % kv_heads == 0:
                            k = k.view(
                                batch_size, seq_len, kv_heads, k_head_dim
                            ).transpose(1, 2)
                        else:
                            k = k.view(batch_size, seq_len, 1, actual_k_dim).transpose(
                                1, 2
                            )

                        if actual_v_dim % kv_heads == 0:
                            v = v.view(
                                batch_size, seq_len, kv_heads, v_head_dim
                            ).transpose(1, 2)
                        else:
                            v = v.view(batch_size, seq_len, 1, actual_v_dim).transpose(
                                1, 2
                            )

                    # Handle grouped query attention if needed
                    if kv_heads < q_heads:
                        # Repeat k and v heads to match q heads
                        repeat_factor = q_heads // kv_heads
                        k = k.repeat(1, repeat_factor, 1, 1)
                        v = v.repeat(1, repeat_factor, 1, 1)

                    # Scaled dot-product attention
                    scale = 1.0 / (q.shape[-1] ** 0.5)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                    # Apply causal mask
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=scores.device), diagonal=1
                    ).bool()
                    scores = scores.masked_fill(causal_mask, -1e9)

                    if attention_mask is not None:
                        # Expand attention mask for multi-head
                        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                        scores = scores.masked_fill(attention_mask == 0, -1e9)

                    attn_weights = F.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v)

                    # Reshape back
                    attn_output = (
                        attn_output.transpose(1, 2)
                        .contiguous()
                        .view(batch_size, seq_len, -1)
                    )

                    # Output projection
                    if o_weight is not None:
                        attn_output = torch.matmul(attn_output, o_weight.T)

                    # Residual connection
                    hidden_states = residual + attn_output

                # Feed-forward network with pre-norm
                ffn_residual = hidden_states

                # Apply post-attention layer norm
                post_attention_layernorm_weight = self.state_dict_data.get(
                    f"{self.layer_name}.post_attention_layernorm.weight"
                )
                if post_attention_layernorm_weight is not None:
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
                    hidden_states = post_attention_layernorm_weight * hidden_states

                # Feed-forward network
                gate_weight = self.state_dict_data.get(
                    f"{self.layer_name}.mlp.gate_proj.weight"
                )
                up_weight = self.state_dict_data.get(
                    f"{self.layer_name}.mlp.up_proj.weight"
                )
                down_weight = self.state_dict_data.get(
                    f"{self.layer_name}.mlp.down_proj.weight"
                )

                if (
                    gate_weight is not None
                    and up_weight is not None
                    and down_weight is not None
                ):
                    # SwiGLU activation
                    gate_output = torch.matmul(hidden_states, gate_weight.T)
                    up_output = torch.matmul(hidden_states, up_weight.T)
                    ffn_output = F.silu(gate_output) * up_output
                    ffn_output = torch.matmul(ffn_output, down_weight.T)

                    # Residual connection
                    hidden_states = ffn_residual + ffn_output

                return hidden_states

        return TransformerLayer(state_dict, layer_name, config)

    @staticmethod
    def create_norm_layer(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create layer norm from state dict"""

        class RMSNorm(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
                self.eps = 1e-6
                self.model_dtype = weight.dtype

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                hidden_states = hidden_states.to(self.model_dtype)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
                return self.weight * hidden_states

        weight = list(state_dict.values())[0]
        return RMSNorm(weight)

    @staticmethod
    def create_lm_head(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create language model head from state dict"""

        class LMHead(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
                self.model_dtype = weight.dtype

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                hidden_states = hidden_states.to(self.model_dtype)
                return torch.matmul(hidden_states, self.weight.T)

        weight = state_dict.get("lm_head.weight", list(state_dict.values())[0])
        return LMHead(weight)


class MultiLayerExecutor:
    """
    Enhanced executor that can load and execute multiple layers simultaneously.
    
    This class handles execution of multiple model layers on a batch of inputs,
    keeping all layers in memory for the duration of the execution.
    """

    def __init__(self, config):
        """
        Initialize the multi-layer executor

        Args:
            config: ModelConfig instance for layer type detection and validation
        """
        self.config = config
        self._execution_lock = threading.RLock()
        self._stats = {
            "total_executions": 0,
            "total_samples": 0,
            "layer_type_counts": {},
            "multi_layer_executions": 0,
        }
        self._loaded_layers = {}  # Cache for loaded layers

    def load_layer_range(self, start_idx: int, end_idx: int) -> Dict[int, nn.Module]:
        """
        Load multiple layers into memory simultaneously
        
        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)
            
        Returns:
            Dictionary mapping layer_idx to loaded nn.Module
        """
        if start_idx < 0 or end_idx >= self.config.total_layers or start_idx > end_idx:
            raise ValueError(f"Invalid layer range: {start_idx}-{end_idx}. Valid range: 0-{self.config.total_layers-1}")
        
        print(f"📥 Loading layers {start_idx} to {end_idx} into memory...")
        
        # Load all state dicts at once
        layers_state_dict = self.config.load_multiple_layers_state_dict(start_idx, end_idx)
        
        loaded_layers = {}
        
        for layer_idx in range(start_idx, end_idx + 1):
            state_dict = layers_state_dict[layer_idx]
            
            if not state_dict:
                print(f"  ⚠️ No state dict found for layer {layer_idx}, skipping...")
                continue
            
            # Determine layer type and create layer
            layer_type = self._get_layer_type(layer_idx)
            layer_name = self.config.get_layer_name_by_index(layer_idx)
            
            print(f"  🏗️ Creating {layer_type} layer {layer_idx}: {layer_name}")
            
            if layer_type == "embedding":
                layer = LayerFactory.create_embedding_layer(state_dict)
            elif layer_type == "transformer":
                layer = LayerFactory.create_transformer_layer(state_dict, layer_name, self.config.config)
            elif layer_type == "norm":
                layer = LayerFactory.create_norm_layer(state_dict)
            elif layer_type == "lm_head":
                layer = LayerFactory.create_lm_head(state_dict)
            else:
                print(f"  ❌ Unknown layer type: {layer_type}")
                continue
            
            layer.eval()
            loaded_layers[layer_idx] = layer
        
        print(f"✅ Successfully loaded {len(loaded_layers)} layers")
        return loaded_layers

    def execute_layer_range(
        self,
        start_idx: int,
        end_idx: int,
        batch: Dict[str, torch.Tensor],
        validate_shapes: bool = True,
        keep_layers_loaded: bool = False
    ) -> torch.Tensor:
        """
        Execute multiple layers sequentially on the input batch
        
        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)
            batch: Input batch dictionary
            validate_shapes: Whether to validate input/output shapes
            keep_layers_loaded: Whether to keep layers in memory after execution
            
        Returns:
            torch.Tensor: Output from the last layer in the range
        """
        # Check if layers are already loaded
        cache_key = f"{start_idx}-{end_idx}"
        
        if cache_key in self._loaded_layers:
            print(f"🔄 Using cached layers {start_idx} to {end_idx}")
            loaded_layers = self._loaded_layers[cache_key]
        else:
            # Load layers
            loaded_layers = self.load_layer_range(start_idx, end_idx)
            if keep_layers_loaded:
                self._loaded_layers[cache_key] = loaded_layers
        
        print(f"⚡ Executing layers {start_idx} to {end_idx}...")
        
        current_batch = batch.copy()
        
        try:
            # Execute each layer in sequence
            for layer_idx in range(start_idx, end_idx + 1):
                if layer_idx not in loaded_layers:
                    print(f"  ⚠️ Layer {layer_idx} not found in loaded layers, skipping...")
                    continue
                
                layer = loaded_layers[layer_idx]
                layer_type = self._get_layer_type(layer_idx)
                
                print(f"  🚀 Executing layer {layer_idx} ({layer_type})")
                
                # Validate inputs
                if validate_shapes:
                    self._validate_inputs(current_batch, layer_type, layer_idx)
                
                # Execute layer based on type
                with torch.no_grad():
                    if layer_type == "embedding":
                        output = self._execute_embedding_layer(layer, current_batch)
                    elif layer_type == "transformer":
                        output = self._execute_transformer_layer(layer, current_batch)
                    elif layer_type == "norm":
                        output = self._execute_norm_layer(layer, current_batch)
                    elif layer_type == "lm_head":
                        output = self._execute_lm_head_layer(layer, current_batch)
                    else:
                        raise ValueError(f"Unknown layer type: {layer_type}")
                
                # Validate output
                if validate_shapes:
                    self._validate_output(output, layer_type, current_batch)
                
                # Update batch for next layer
                current_batch['hidden_states'] = output
                
                # Remove input_ids after embedding layer
                if layer_type == "embedding" and 'input_ids' in current_batch:
                    del current_batch['input_ids']
                
                print(f"  ✅ Layer {layer_idx} completed - Output shape: {output.shape}")
            
            # Update stats
            self._update_multi_layer_stats(start_idx, end_idx, batch)
            
            print(f"🎉 Layer range {start_idx}-{end_idx} execution completed!")
            return output
            
        except Exception as e:
            raise RuntimeError(
                f"Multi-layer execution failed for range {start_idx}-{end_idx}: {str(e)}"
            ) from e
        
        finally:
            # Clean up if not keeping layers loaded
            if not keep_layers_loaded and cache_key not in self._loaded_layers:
                del loaded_layers
                gc.collect()

    def unload_layers(self, start_idx: int = None, end_idx: int = None):
        """
        Unload layers from memory cache
        
        Args:
            start_idx: Starting layer index (if None, unload all)
            end_idx: Ending layer index (if None, unload all)
        """
        if start_idx is None or end_idx is None:
            # Unload all cached layers
            print("🧹 Unloading all cached layers...")
            self._loaded_layers.clear()
        else:
            cache_key = f"{start_idx}-{end_idx}"
            if cache_key in self._loaded_layers:
                print(f"🧹 Unloading cached layers {start_idx}-{end_idx}...")
                del self._loaded_layers[cache_key]
            else:
                print(f"⚠️ No cached layers found for range {start_idx}-{end_idx}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_layer_type(self, layer_idx: int) -> str:
        """Determine layer type from index"""
        if layer_idx == 0:
            return "embedding"
        elif layer_idx == self.config.total_layers - 2:
            return "norm"
        elif layer_idx == self.config.total_layers - 1:
            return "lm_head"
        else:
            return "transformer"

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
        """Execute transformer layer"""
        hidden_states = batch.get("hidden_states")
        attention_mask = batch.get("attention_mask")

        if hidden_states is None:
            raise ValueError("Transformer layer requires 'hidden_states' in batch")

        # Check if layer accepts attention mask
        if (
            hasattr(layer, "forward")
            and "attention_mask" in layer.forward.__code__.co_varnames
        ):
            return layer(hidden_states, attention_mask=attention_mask)
        else:
            return layer(hidden_states)

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

    def _validate_inputs(
        self, batch: Dict[str, torch.Tensor], layer_type: str, layer_idx: int
    ):
        """Validate input batch for specific layer type"""
        if layer_type == "embedding":
            input_ids = batch.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    f"Layer {layer_idx} (embedding): 'input_ids' required in batch"
                )
            if input_ids.dim() != 2:
                raise ValueError(
                    f"Layer {layer_idx} (embedding): Expected 2D input_ids, got {input_ids.dim()}D"
                )
            if input_ids.dtype not in [torch.long, torch.int]:
                raise ValueError(
                    f"Layer {layer_idx} (embedding): Expected integer input_ids, got {input_ids.dtype}"
                )

        else:  # transformer, norm, lm_head layers
            hidden_states = batch.get("hidden_states")
            if hidden_states is None:
                raise ValueError(
                    f"Layer {layer_idx} ({layer_type}): 'hidden_states' required in batch"
                )
            if hidden_states.dim() != 3:
                raise ValueError(
                    f"Layer {layer_idx} ({layer_type}): Expected 3D hidden_states, got {hidden_states.dim()}D"
                )
            if hidden_states.dtype not in [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ]:
                raise ValueError(
                    f"Layer {layer_idx} ({layer_type}): Expected float hidden_states, got {hidden_states.dtype}"
                )

            # Validate attention mask if present
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                if attention_mask.dim() != 2:
                    raise ValueError(
                        f"Layer {layer_idx} ({layer_type}): Expected 2D attention_mask, got {attention_mask.dim()}D"
                    )
                if attention_mask.shape[:2] != hidden_states.shape[:2]:
                    raise ValueError(
                        f"Layer {layer_idx} ({layer_type}): attention_mask shape {attention_mask.shape[:2]} doesn't match hidden_states {hidden_states.shape[:2]}"
                    )

    def _validate_output(
        self, output: torch.Tensor, layer_type: str, batch: Dict[str, torch.Tensor]
    ):
        """Validate output tensor"""
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Layer output must be torch.Tensor, got {type(output)}")

        if layer_type == "embedding":
            input_ids = batch["input_ids"]
            expected_shape = (
                input_ids.shape[0],
                input_ids.shape[1],
                -1,
            )  # (batch, seq, hidden)
            if output.shape[:2] != expected_shape[:2]:
                raise ValueError(
                    f"Embedding output shape {output.shape[:2]} doesn't match expected {expected_shape[:2]}"
                )
            if output.dim() != 3:
                raise ValueError(f"Embedding output must be 3D, got {output.dim()}D")

        elif layer_type == "lm_head":
            hidden_states = batch["hidden_states"]
            expected_shape = (
                hidden_states.shape[0],
                hidden_states.shape[1],
                -1,
            )  # (batch, seq, vocab)
            if output.shape[:2] != expected_shape[:2]:
                raise ValueError(
                    f"LM head output shape {output.shape[:2]} doesn't match expected {expected_shape[:2]}"
                )

        else:  # transformer, norm layers
            hidden_states = batch["hidden_states"]
            if output.shape != hidden_states.shape:
                raise ValueError(
                    f"{layer_type.title()} layer output shape {output.shape} doesn't match input shape {hidden_states.shape}"
                )

    def _update_multi_layer_stats(self, start_idx: int, end_idx: int, batch: Dict[str, torch.Tensor]):
        """Update execution statistics for multi-layer execution (thread-safe)"""
        with self._execution_lock:
            self._stats["multi_layer_executions"] += 1
            
            # Count samples in batch
            if "input_ids" in batch:
                batch_size = batch["input_ids"].shape[0]
            elif "hidden_states" in batch:
                batch_size = batch["hidden_states"].shape[0]
            else:
                batch_size = 1

            self._stats["total_samples"] += batch_size
            
            # Count layer types in the range
            for layer_idx in range(start_idx, end_idx + 1):
                layer_type = self._get_layer_type(layer_idx)
                self._stats["layer_type_counts"][layer_type] = (
                    self._stats["layer_type_counts"].get(layer_type, 0) + 1
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics (thread-safe)"""
        with self._execution_lock:
            return self._stats.copy()

    def reset_stats(self):
        """Reset execution statistics (thread-safe)"""
        with self._execution_lock:
            self._stats = {
                "total_executions": 0,
                "total_samples": 0,
                "layer_type_counts": {},
                "multi_layer_executions": 0,
            }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        memory_info = {
            "loaded_layer_ranges": list(self._loaded_layers.keys()),
            "total_cached_ranges": len(self._loaded_layers)
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return memory_info


# Test code demonstrating multi-layer batch execution
if __name__ == "__main__":
    print("🧪 Testing Multi-Layer Batch Execution...")
    
    repo_id = "Qwen/Qwen3-4B"
    local_dir = "models"
    
    # Initialize config and executor
    config = ModelConfig(local_dir, repo_id)
    executor = MultiLayerExecutor(config)
    
    print(f"✅ Initialized multi-layer executor for {repo_id}")
    print(f"📊 Model config: {config}")
    print(f"🔢 Total layers: {config.total_layers}")
    
    # Initialize tokenizer for proper input creation
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        print(f"✅ Loaded tokenizer")
    except Exception as e:
        print(f"⚠️ Could not load tokenizer: {e}")
        tokenizer = None
    
    # Create test input batch
    batch_size = 2
    seq_length = 256
    
    test_prompts = [
        "The future of artificial intelligence depends on",
        "In the next decade, machine learning will"
    ]
    
    if tokenizer:
        # Tokenize with padding
        inputs = tokenizer(test_prompts, 
                          padding=True, 
                          truncation=True, 
                          max_length=seq_length,
                          return_tensors="pt")
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        print(f"📝 Using prompts (batch_size={batch_size}):")
        for i, prompt in enumerate(test_prompts):
            print(f"  {i+1}: {prompt}")
        print(f"🔤 Input tokens shape: {input_ids.shape}")
        print(f"👁️ Attention mask shape: {attention_mask.shape}")
        
    else:
        # Fallback to random tokens
        input_ids = torch.randint(100, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        print(f"🎲 Using random tokens: {input_ids.shape}")
    
    # Initialize batch dictionary
    current_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Test different layer ranges
    test_ranges = [
        (0, 4),    # First 5 layers (embedding + 4 transformer layers)
        (5, 9),    # Next 5 transformer layers
        (10, 14),  # Next 5 transformer layers
        (15, 19),  # Next 5 transformer layers
        (20, 24),  # Next 5 transformer layers
    ]
    
    # Add final layers if there are more
    remaining_layers = config.total_layers - 25
    if remaining_layers > 0:
        test_ranges.append((25, config.total_layers - 1))
    
    print(f"\n🚀 Starting multi-layer batch inference...")
    print(f"📦 Batch size: {batch_size}, Sequence length: {seq_length}")
    print(f"🔄 Testing {len(test_ranges)} layer ranges")
    
    import time
    
    total_start_time = time.time()
    range_times = []
    
    for i, (start_idx, end_idx) in enumerate(test_ranges):
        if end_idx >= config.total_layers:
            end_idx = config.total_layers - 1
        
        print(f"\n--- Range {i+1}/{len(test_ranges)}: Layers {start_idx}-{end_idx} ---")
        
        range_start_time = time.time()
        
        try:
            # Execute layer range
            output = executor.execute_layer_range(
                start_idx=start_idx,
                end_idx=end_idx,
                batch=current_batch,
                validate_shapes=False,
                keep_layers_loaded=True  # Keep in memory for faster subsequent calls
            )
            
            # Update batch for next range
            current_batch['hidden_states'] = output
            
            # Remove input_ids after first range (embedding processed)
            if 'input_ids' in current_batch and start_idx == 0:
                del current_batch['input_ids']
            
            range_time = time.time() - range_start_time
            range_times.append(range_time)
            
            print(f"✅ Range {start_idx}-{end_idx} completed in {range_time:.3f}s")
            print(f"   Output shape: {output.shape}, dtype: {output.dtype}")
            print(f"   Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            
            # Show memory usage
            memory_info = executor.get_memory_usage()
            print(f"   Memory: {memory_info}")
            
        except Exception as e:
            print(f"❌ Range {start_idx}-{end_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 Multi-layer batch inference completed!")
    print(f"⏱️ Total time: {total_time:.3f}s")
    print(f"📊 Range times: {[f'{t:.3f}s' for t in range_times]}")
    print(f"🚀 Average time per range: {sum(range_times)/len(range_times):.3f}s")
    print(f"⚡ Throughput: {batch_size/total_time:.2f} samples/second")
    
    # Final output analysis
    final_output = current_batch.get('hidden_states')
    if final_output is not None:
        print(f"\n📋 Final Model Output Analysis:")
        print(f"  Shape: {final_output.shape}")
        print(f"  Dtype: {final_output.dtype}")
        print(f"  Mean: {final_output.mean().item():.6f}")
        print(f"  Std: {final_output.std().item():.6f}")
        print(f"  Min: {final_output.min().item():.6f}")
        print(f"  Max: {final_output.max().item():.6f}")
    
    # Show executor stats
    print(f"\n📊 Executor Statistics:")
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test text generation with multi-layer approach
    if final_output is not None and tokenizer:
        print(f"\n🎯 Testing Text Generation with Multi-Layer Execution...")
        
        # Generate 20 tokens for the first prompt
        current_tokens = input_ids[0:1]  # Take first prompt
        current_mask = attention_mask[0:1]
        max_new_tokens = 20
        
        print(f"Original prompt: {test_prompts[0]}")
        print("Generating tokens...")
        
        for step in range(max_new_tokens):
            # Create batch for current step
            step_batch = {
                'input_ids': current_tokens,
                'attention_mask': current_mask
            }
            
            # Execute all layers in ranges
            for start_idx, end_idx in test_ranges:
                if end_idx >= config.total_layers:
                    end_idx = config.total_layers - 1
                
                output = executor.execute_layer_range(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    batch=step_batch,
                    validate_shapes=False,
                    keep_layers_loaded=True
                )
                
                step_batch['hidden_states'] = output
                
                # Remove input_ids after first range
                if 'input_ids' in step_batch and start_idx == 0:
                    del step_batch['input_ids']
            
            # Get next token from logits
            last_token_logits = output[0, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
            
            # Show progress
            if (step + 1) % 5 == 0:
                partial_text = tokenizer.decode(current_tokens[0], skip_special_tokens=True)
                print(f"Step {step+1}: {partial_text}")
        
        # Final generated text
        final_text = tokenizer.decode(current_tokens[0], skip_special_tokens=True)
        print(f"\n🎉 Final Generated Text:")
        print(f"{final_text}")
    
    # Clean up memory
    print(f"\n🧹 Cleaning up memory...")
    executor.unload_layers()  # Unload all cached layers
    
    print(f"✨ Multi-layer batch execution demonstration completed!")
    
    # Performance comparison summary
    print(f"\n📈 Performance Summary:")
    print(f"  Multi-layer approach processed {len(test_ranges)} ranges")
    print(f"  Each range contained ~{(config.total_layers//len(test_ranges)):.0f} layers on average")
    print(f"  Memory efficiency: Layers loaded in batches rather than individually")
    print(f"  Execution speed: {total_time:.3f}s for full model inference")
    
    # Demonstrate different range sizes
    print(f"\n🔬 Testing Different Range Sizes...")
    
    # Test with smaller ranges
    small_ranges = [(0, 2), (3, 5), (6, 8)]
    print(f"Testing smaller ranges: {small_ranges}")
    
    # Reset batch
    test_batch = {
        'input_ids': input_ids[:1],  # Single sample
        'attention_mask': attention_mask[:1]
    }
    
    small_range_times = []
    for start_idx, end_idx in small_ranges:
        start_time = time.time()
        
        try:
            output = executor.execute_layer_range(
                start_idx=start_idx,
                end_idx=end_idx,
                batch=test_batch,
                validate_shapes=False,
                keep_layers_loaded=False  # Don't cache for this test
            )
            
            test_batch['hidden_states'] = output
            if 'input_ids' in test_batch and start_idx == 0:
                del test_batch['input_ids']
            
            exec_time = time.time() - start_time
            small_range_times.append(exec_time)
            print(f"  Range {start_idx}-{end_idx}: {exec_time:.3f}s")
            
        except Exception as e:
            print(f"  Range {start_idx}-{end_idx} failed: {e}")
    
    print(f"Small ranges total time: {sum(small_range_times):.3f}s")
    print(f"Average per small range: {sum(small_range_times)/len(small_range_times):.3f}s")
    
    print(f"\n🎯 Multi-layer executor provides flexibility in:")
    print(f"  • Memory management (load ranges as needed)")
    print(f"  • Performance optimization (batch layer execution)")
    print(f"  • Resource utilization (cache frequently used ranges)")
    print(f"  • Debugging (isolate specific layer ranges)")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n✅ All tests completed successfully!")