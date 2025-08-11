import gc
import os
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

        # Mapping from layer names to parameter names
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
    """Base class for thread-safe layer-by-layer model execution"""

    def __init__(self, config: ModelConfig):
        self.config = config
        # Simple stats without threading locks - let external systems handle thread safety if needed
        self._stats = {
            "total_executions": 0,
            "total_samples": 0,
            "layer_type_counts": {},
            "multi_layer_executions": 0,
        }

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
        Load multiple layers into memory - pure loading function

        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)

        Returns:
            Dictionary mapping layer_idx to loaded nn.Module

        Note:
            This is a pure function - no caching or cleanup.
            External pipeline should handle memory management.
        """
        if start_idx < 0 or end_idx >= self.config.total_layers or start_idx > end_idx:
            raise ValueError(
                f"Invalid layer range: {start_idx}-{end_idx}. Valid range: 0-{self.config.total_layers-1}"
            )

        # Load all state dicts at once
        layers_state_dict = self.config.load_multiple_layers_state_dict(
            start_idx, end_idx
        )

        loaded_layers = {}

        for layer_idx in range(start_idx, end_idx + 1):
            state_dict = layers_state_dict[layer_idx]

            if not state_dict:
                continue

            # Use subclass implementation to create layer
            layer = self.create_layer(layer_idx, state_dict)
            layer.eval()
            loaded_layers[layer_idx] = layer

        return loaded_layers

    def execute_layer_range(
        self,
        loaded_layers: Dict[int, nn.Module],
        batch: Dict[str, torch.Tensor],
        validate_shapes: bool = True,
    ) -> torch.Tensor:
        """
        Execute multiple layers sequentially using pre-loaded layers

        Args:
            loaded_layers: Dictionary mapping layer_idx to loaded nn.Module
            batch: Input batch dictionary
            validate_shapes: Whether to validate input/output shapes

        Returns:
            torch.Tensor: Output from the last layer in the range
        """
        if not loaded_layers:
            raise ValueError("loaded_layers cannot be empty")

        layer_indices = sorted(loaded_layers.keys())
        start_idx, end_idx = layer_indices[0], layer_indices[-1]

        # Verify we have all layers in the range
        for layer_idx in range(start_idx, end_idx + 1):
            if layer_idx not in loaded_layers:
                raise ValueError(f"Missing layer {layer_idx} in loaded_layers")

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
                layer = loaded_layers[layer_idx]
                layer_type = self._get_layer_type(layer_idx)

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

            # Update stats (non-thread-safe - let external systems handle if needed)
            self._update_stats(start_idx, end_idx, batch)

            return output

        except Exception as e:
            raise RuntimeError(
                f"Multi-layer execution failed for range {start_idx}-{end_idx}: {str(e)}"
            ) from e

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

    def _update_stats(
        self, start_idx: int, end_idx: int, batch: Dict[str, torch.Tensor]
    ):
        """Update execution statistics (simple, non-thread-safe)"""
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
        """Get execution statistics (non-thread-safe)"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset execution statistics (non-thread-safe)"""
        self._stats = {
            "total_executions": 0,
            "total_samples": 0,
            "layer_type_counts": {},
            "multi_layer_executions": 0,
        }
