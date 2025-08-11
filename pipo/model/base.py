import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig

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
            snapshot_download(repo_id=repo_id, local_dir=self.model_path)

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

    def load_multiple_layers_state_dict(
        self, start_idx: int, end_idx: int
    ) -> Dict[int, dict]:
        """
        Load state_dict for multiple layers at once

        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)

        Returns:
            Dictionary mapping layer_idx to state_dict
        """
        if start_idx < 0 or end_idx >= self.total_layers or start_idx > end_idx:
            raise ValueError(
                f"Invalid layer range: {start_idx}-{end_idx}. Valid range: 0-{self.total_layers-1}"
            )

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


class BaseExecutor:
    """Base class for layer-by-layer model execution"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._execution_lock = threading.RLock()
        self._stats = {
            "total_executions": 0,
            "total_samples": 0,
            "layer_type_counts": {},
            "multi_layer_executions": 0,
        }
        self._loaded_layers = {}  # Cache for loaded layers

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

    def create_layer(
        self, layer_idx: int, state_dict: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Create a layer from state_dict. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_layer method")

    def execute_layer(
        self, layer: nn.Module, layer_idx: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Execute a single layer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute_layer method")

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
            raise ValueError(
                f"Invalid layer range: {start_idx}-{end_idx}. Valid range: 0-{self.config.total_layers-1}"
            )

        # print(f"📥 Loading layers {start_idx} to {end_idx} into memory...")

        # Load all state dicts at once
        layers_state_dict = self.config.load_multiple_layers_state_dict(
            start_idx, end_idx
        )

        loaded_layers = {}

        for layer_idx in range(start_idx, end_idx + 1):
            state_dict = layers_state_dict[layer_idx]

            if not state_dict:
                # print(f"  ⚠️ No state dict found for layer {layer_idx}, skipping...")
                continue

            layer_type = self._get_layer_type(layer_idx)
            layer_name = self.config.get_layer_name_by_index(layer_idx)

            # print(f"  🏗️ Creating {layer_type} layer {layer_idx}: {layer_name}")

            # Use subclass implementation to create layer
            layer = self.create_layer(layer_idx, state_dict)
            layer.eval()
            loaded_layers[layer_idx] = layer

        # print(f"✅ Successfully loaded {len(loaded_layers)} layers")
        return loaded_layers

    def execute_layer_range(
        self,
        start_idx: int,
        end_idx: int,
        batch: Dict[str, torch.Tensor],
        validate_shapes: bool = True,
        keep_layers_loaded: bool = False,
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
            # print(f"🔄 Using cached layers {start_idx} to {end_idx}")
            loaded_layers = self._loaded_layers[cache_key]
        else:
            # Load layers
            loaded_layers = self.load_layer_range(start_idx, end_idx)
            if keep_layers_loaded:
                self._loaded_layers[cache_key] = loaded_layers

        # print(f"⚡ Executing layers {start_idx} to {end_idx}...")

        current_batch = batch.copy()

        # Ensure position_ids are present for transformer models
        if "position_ids" not in current_batch and (
            "input_ids" in current_batch or "hidden_states" in current_batch
        ):
            if "input_ids" in current_batch:
                seq_len = current_batch["input_ids"].shape[1]
                batch_size = current_batch["input_ids"].shape[0]
                device = current_batch["input_ids"].device
            else:
                seq_len = current_batch["hidden_states"].shape[1]
                batch_size = current_batch["hidden_states"].shape[0]
                device = current_batch["hidden_states"].device

            current_batch["position_ids"] = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        try:
            # Execute each layer in sequence
            for layer_idx in range(start_idx, end_idx + 1):
                if layer_idx not in loaded_layers:
                    # print(f"  ⚠️ Layer {layer_idx} not found in loaded layers, skipping...")
                    continue

                layer = loaded_layers[layer_idx]
                layer_type = self._get_layer_type(layer_idx)

                # print(f"  🚀 Executing layer {layer_idx} ({layer_type})")

                # Validate inputs
                if validate_shapes:
                    self._validate_inputs(current_batch, layer_type, layer_idx)

                # Execute layer using subclass implementation
                with torch.no_grad():
                    output = self.execute_layer(layer, layer_idx, current_batch)

                # Handle tuple returns (some layers return (hidden_states, attentions, etc.))
                if isinstance(output, tuple):
                    output = output[0]

                # Validate output
                if validate_shapes:
                    self._validate_output(output, layer_type, current_batch)

                # Update batch for next layer
                current_batch["hidden_states"] = output

                # Remove input_ids after embedding layer
                if layer_type == "embedding" and "input_ids" in current_batch:
                    del current_batch["input_ids"]

                # print(f"  ✅ Layer {layer_idx} completed - Output shape: {output.shape}")

            # Update stats
            self._update_multi_layer_stats(start_idx, end_idx, batch)

            # print(f"🎉 Layer range {start_idx}-{end_idx} execution completed!")
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
            # print("🧹 Unloading all cached layers...")
            self._loaded_layers.clear()
        else:
            cache_key = f"{start_idx}-{end_idx}"
            if cache_key in self._loaded_layers:
                # print(f"🧹 Unloading cached layers {start_idx}-{end_idx}...")
                del self._loaded_layers[cache_key]
            # else:
            # print(f"⚠️ No cached layers found for range {start_idx}-{end_idx}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _update_multi_layer_stats(
        self, start_idx: int, end_idx: int, batch: Dict[str, torch.Tensor]
    ):
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
            "total_cached_ranges": len(self._loaded_layers),
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated"] = (
                torch.cuda.memory_allocated() / 1024**3
            )  # GB
            memory_info["gpu_memory_reserved"] = (
                torch.cuda.memory_reserved() / 1024**3
            )  # GB

        return memory_info
