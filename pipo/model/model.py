import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
import inspect

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
    """Factory for creating different types of model layers using transformers library implementations"""

    @staticmethod
    def _get_model_class_and_components(config):
        """
        Dynamically get the appropriate model class and components based on config.
        This approach is more flexible and can handle new architectures automatically.
        """
        from transformers import AutoModelForCausalLM
        
        try:
            # Try to get the model class from AutoModelForCausalLM mapping
            try:
                model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
            except TypeError:
                # Handle newer transformers versions with different mapping API
                model_class = AutoModelForCausalLM._model_mapping[type(config)] if type(config) in AutoModelForCausalLM._model_mapping else None
            
            if model_class is None:
                # Fallback: try to instantiate directly
                with init_empty_weights():
                    temp_model = AutoModelForCausalLM.from_config(config)
                model_class = type(temp_model)
        except Exception as e:
            # Final fallback: try to detect based on config class name
            config_name = config.__class__.__name__.lower()
            if 'qwen' in config_name:
                # For Qwen models (including Qwen3), try to use Qwen2 as fallback
                try:
                    from transformers import Qwen2ForCausalLM
                    model_class = Qwen2ForCausalLM
                    print(f"Using Qwen2ForCausalLM as fallback for {config.__class__.__name__}")
                except ImportError:
                    raise ValueError(f"Could not determine model class for config type {type(config)}: {e}")
            elif 'llama' in config_name:
                try:
                    from transformers import LlamaForCausalLM
                    model_class = LlamaForCausalLM
                except ImportError:
                    raise ValueError(f"Could not determine model class for config type {type(config)}: {e}")
            else:
                raise ValueError(f"Could not determine model class for config type {type(config)}: {e}")
        
        # Get the base model class (without the LM head)
        base_model_class = getattr(model_class, 'base_model_prefix', None)
        if base_model_class and hasattr(model_class, base_model_class):
            base_model_attr = getattr(model_class, base_model_class)
            if hasattr(base_model_attr, 'fget'):
                # It's a property, need to get the actual class differently
                with init_empty_weights():
                    temp_full_model = model_class(config)
                    base_model = getattr(temp_full_model, base_model_class)
                    base_model_class = type(base_model)
            else:
                base_model_class = base_model_attr
        else:
            # Try to infer base model class from the full model
            with init_empty_weights():
                temp_full_model = model_class(config)
                # Common base model attribute names
                for attr_name in ['model', 'transformer', 'base_model']:
                    if hasattr(temp_full_model, attr_name):
                        base_model = getattr(temp_full_model, attr_name)
                        base_model_class = type(base_model)
                        break
                else:
                    base_model_class = model_class
        
        return model_class, base_model_class

    @staticmethod
    def _get_decoder_layer_class(config):
        """
        Dynamically get the decoder layer class for any transformer architecture.
        """
        try:
            _, base_model_class = LayerFactory._get_model_class_and_components(config)
            
            # Create a temporary base model to inspect its layers
            with init_empty_weights():
                temp_model = base_model_class(config)
                
                # Common layer container names
                layer_containers = ['layers', 'h', 'blocks', 'transformer_blocks']
                decoder_layer_class = None
                
                for container_name in layer_containers:
                    if hasattr(temp_model, container_name):
                        layers = getattr(temp_model, container_name)
                        if hasattr(layers, '__len__') and len(layers) > 0:
                            # Get the class of the first layer
                            first_layer = layers[0] if hasattr(layers, '__getitem__') else next(iter(layers))
                            decoder_layer_class = type(first_layer)
                            break
                
                if decoder_layer_class is None:
                    raise ValueError("Could not find decoder layer class")
                    
                return decoder_layer_class, container_name
                
        except Exception as e:
            # Fallback: try to use known decoder layers based on config name
            config_name = config.__class__.__name__.lower()
            if 'qwen' in config_name:
                try:
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                    print(f"Using Qwen2DecoderLayer as fallback for {config.__class__.__name__}")
                    return Qwen2DecoderLayer, 'layers'
                except ImportError:
                    pass
            elif 'llama' in config_name:
                try:
                    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                    return LlamaDecoderLayer, 'layers'
                except ImportError:
                    pass
            
            raise ValueError(f"Could not determine decoder layer class: {e}")

    @staticmethod
    def _get_norm_class(config):
        """
        Dynamically get the normalization class used by the model.
        """
        try:
            _, base_model_class = LayerFactory._get_model_class_and_components(config)
            
            with init_empty_weights():
                temp_model = base_model_class(config)
                
                # Common norm attribute names
                norm_attrs = ['norm', 'final_layer_norm', 'layer_norm', 'ln_f']
                
                for norm_attr in norm_attrs:
                    if hasattr(temp_model, norm_attr):
                        norm_layer = getattr(temp_model, norm_attr)
                        if norm_layer is not None:
                            return type(norm_layer)
                
                # If no final norm found, check inside a decoder layer
                decoder_layer_class, _ = LayerFactory._get_decoder_layer_class(config)
                with init_empty_weights():
                    temp_layer = decoder_layer_class(config, 0)
                    
                    # Common norm names in decoder layers
                    layer_norm_attrs = ['input_layernorm', 'post_attention_layernorm', 'ln_1', 'ln_2']
                    for norm_attr in layer_norm_attrs:
                        if hasattr(temp_layer, norm_attr):
                            norm_layer = getattr(temp_layer, norm_attr)
                            if norm_layer is not None:
                                return type(norm_layer)
                
                raise ValueError("Could not find normalization class")
                
        except Exception as e:
            # Fallback to generic implementation
            return None

    @staticmethod
    def create_embedding_layer(state_dict: Dict[str, torch.Tensor], config) -> nn.Module:
        """Create embedding layer using transformers implementation"""
        
        try:
            _, base_model_class = LayerFactory._get_model_class_and_components(config)
            
            # Create temporary model to extract embedding layer
            with init_empty_weights():
                temp_model = base_model_class(config)
                
                # Common embedding attribute names
                embed_attrs = ['embed_tokens', 'embeddings', 'wte', 'token_embedding']
                embed_layer = None
                
                for embed_attr in embed_attrs:
                    if hasattr(temp_model, embed_attr):
                        embed_layer = getattr(temp_model, embed_attr)
                        if embed_layer is not None:
                            break
                
                if embed_layer is None:
                    raise ValueError("Could not find embedding layer")
                    
                # Create a new instance of the same class
                embed_layer = type(embed_layer)(embed_layer.num_embeddings, embed_layer.embedding_dim)
                
        except Exception as e:
            # Fallback to generic embedding
            print(f"Warning: Using generic embedding due to: {e}")
            vocab_size = getattr(config, 'vocab_size', 32000)
            hidden_size = getattr(config, 'hidden_size', 4096)
            embed_layer = nn.Embedding(vocab_size, hidden_size)

        # Load the weights
        for param_name, param_tensor in state_dict.items():
            if hasattr(embed_layer, 'weight'):
                set_module_tensor_to_device(embed_layer, 'weight', 'cpu', value=param_tensor)
                break

        return embed_layer

    @staticmethod
    def create_transformer_layer(
        state_dict: Dict[str, torch.Tensor], layer_name: str, config
    ) -> nn.Module:
        """Create transformer layer using transformers implementation"""
        
        try:
            decoder_layer_class, _ = LayerFactory._get_decoder_layer_class(config)
            layer_idx = int(layer_name.split('.')[-1])  # Extract layer index
            
            # Create the layer
            with init_empty_weights():
                layer = decoder_layer_class(config, layer_idx)
                
        except Exception as e:
            raise ValueError(f"Could not create transformer layer: {e}")

        # Load weights into the layer
        for param_name, param_tensor in state_dict.items():
            if param_name.startswith(layer_name + "."):
                relative_param_name = param_name[len(layer_name) + 1:]
                
                # Navigate to the correct parameter
                param_parts = relative_param_name.split('.')
                current_module = layer
                
                try:
                    # Navigate through nested modules
                    for part in param_parts[:-1]:
                        current_module = getattr(current_module, part)
                    
                    # Set the final parameter
                    param_attr = param_parts[-1]
                    if hasattr(current_module, param_attr):
                        set_module_tensor_to_device(
                            current_module, param_attr, 'cpu', value=param_tensor
                        )
                except AttributeError:
                    print(f"Warning: Could not load parameter {param_name}")
                    continue

        return layer

    @staticmethod
    def create_norm_layer(state_dict: Dict[str, torch.Tensor], config) -> nn.Module:
        """Create normalization layer using transformers implementation"""
        
        norm_class = LayerFactory._get_norm_class(config)
        
        if norm_class is not None:
            try:
                # Try to create the norm layer with common parameters
                hidden_size = getattr(config, 'hidden_size', 4096)
                
                # Try different constructor signatures
                try:
                    # Most common: (hidden_size, eps)
                    eps = getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))
                    norm_layer = norm_class(hidden_size, eps=eps)
                except TypeError:
                    try:
                        # Some models: (hidden_size,)
                        norm_layer = norm_class(hidden_size)
                    except TypeError:
                        # Fallback: try with config
                        norm_layer = norm_class(config)
                        
            except Exception as e:
                print(f"Warning: Could not create {norm_class.__name__}: {e}")
                norm_class = None
        
        if norm_class is None:
            # Generic RMS norm fallback
            hidden_size = getattr(config, 'hidden_size', 4096)
            rms_norm_eps = getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))
            
            class GenericRMSNorm(nn.Module):
                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(hidden_size))
                    self.variance_epsilon = eps

                def forward(self, hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                    return self.weight * hidden_states.to(input_dtype)
            
            norm_layer = GenericRMSNorm(hidden_size, eps=rms_norm_eps)

        # Load weights
        for param_name, param_tensor in state_dict.items():
            if hasattr(norm_layer, 'weight'):
                set_module_tensor_to_device(norm_layer, 'weight', 'cpu', value=param_tensor)
                break

        return norm_layer

    @staticmethod
    def create_lm_head(state_dict: Dict[str, torch.Tensor], config) -> nn.Module:
        """Create language model head using transformers implementation"""
        
        try:
            model_class, _ = LayerFactory._get_model_class_and_components(config)
            
            # Create temporary full model to extract LM head
            with init_empty_weights():
                temp_model = model_class(config)
                
                # Common LM head attribute names
                head_attrs = ['lm_head', 'head', 'classifier', 'score']
                lm_head = None
                
                for head_attr in head_attrs:
                    if hasattr(temp_model, head_attr):
                        head = getattr(temp_model, head_attr)
                        if head is not None:
                            # Create new instance of the same type
                            if hasattr(head, 'in_features') and hasattr(head, 'out_features'):
                                lm_head = type(head)(head.in_features, head.out_features, bias=hasattr(head, 'bias') and head.bias is not None)
                            break
                
                if lm_head is None:
                    raise ValueError("Could not find LM head")
                    
        except Exception as e:
            # Fallback to generic linear layer
            print(f"Warning: Using generic LM head due to: {e}")
            vocab_size = getattr(config, 'vocab_size', 32000)
            hidden_size = getattr(config, 'hidden_size', 4096)
            lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Load weights
        weight_tensor = None
        if 'lm_head.weight' in state_dict:
            weight_tensor = state_dict['lm_head.weight']
        else:
            weight_tensor = list(state_dict.values())[0]
        
        if weight_tensor is not None:
            set_module_tensor_to_device(lm_head, 'weight', 'cpu', value=weight_tensor)
        
        return lm_head

class MultiLayerExecutor:
    def __init__(self, config):
        self.config = config
        self._execution_lock = threading.RLock()
        self._stats = {
            "total_executions": 0,
            "total_samples": 0,
            "layer_type_counts": {},
            "multi_layer_executions": 0,
        }
        self._loaded_layers = {}  # Cache for loaded layers
        self._rotary_emb = None  # Cache for rotary embeddings


    def _initialize_rotary_embeddings(self, head_dim: Optional[int] = None):
        """Init rotary emb, tolerant of different Qwen ctor signatures."""
        if getattr(self, "_rotary_emb", None) is not None:
            return

        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        except Exception as e:
            print("⚠️ Qwen2RotaryEmbedding import failed:", e)
            self._rotary_emb = None
            return

        ctor_sig = inspect.signature(Qwen2RotaryEmbedding.__init__)
        params = set(ctor_sig.parameters.keys())  # includes 'self'

        # derive a sensible head_dim if not provided
        if head_dim is None:
            hidden_size = getattr(self.config.config, "hidden_size", None) or 4096
            num_heads = getattr(self.config.config, "num_attention_heads", None) or 32
            head_dim = int(hidden_size) // int(num_heads)

        max_pos = getattr(self.config.config, "max_position_embeddings", 32768)
        rope_theta = getattr(self.config.config, "rope_theta", 1000000.0)

        # Choose constructor pattern depending on parameters:
        try:
            if "config" in params:
                # Many implementations expect the HF config object
                conf = getattr(self.config, "config", None)
                if conf is None:
                    raise RuntimeError("No config object available to pass to Qwen2RotaryEmbedding")
                # Try passing the whole config (positional or keyword)
                try:
                    self._rotary_emb = Qwen2RotaryEmbedding(conf)
                except TypeError:
                    self._rotary_emb = Qwen2RotaryEmbedding(config=conf)
            else:
                # param-based constructor; only pass supported kwargs
                kw = {}
                if "dim" in params:
                    kw["dim"] = head_dim
                elif "head_dim" in params:
                    kw["head_dim"] = head_dim
                if "max_position_embeddings" in params:
                    kw["max_position_embeddings"] = max_pos
                if "base" in params:
                    kw["base"] = rope_theta
                self._rotary_emb = Qwen2RotaryEmbedding(**kw)
        except Exception as e:
            print("⚠️ Could not initialize rotary embeddings:", e)
            self._rotary_emb = None
            return

        # record the head_dim the rotary was created for
        self._rotary_head_dim = head_dim
        # small wrapper to call the rotary consistently
        self._rotary_call = lambda position_ids, seq_len, device: self._call_rotary(position_ids, seq_len, device)
        print(f"✅ Initialized rotary with head_dim={head_dim}")



    def _call_rotary(self, position_ids: Optional[torch.LongTensor], seq_len: int, device: torch.device):
        if self._rotary_emb is None:
            raise RuntimeError("rotary not initialized")

        last_exc = None
        # Try common call signatures in this order:
        attempts = []

        if position_ids is not None:
            attempts += [
                lambda: self._rotary_emb(position_ids),                  # common: forward(position_ids)
                lambda: self._rotary_emb(position_ids.to(device)),      # same, ensure device
            ]
        # some impls accept just seq_len or (seq_len, device)
        attempts += [
            lambda: self._rotary_emb(seq_len),
            lambda: self._rotary_emb(seq_len, device),
            lambda: self._rotary_emb(device=device),  # unlikely but harmless
            lambda: self._rotary_emb(),               # try no-arg
        ]

        for fn in attempts:
            try:
                out = fn()
                if isinstance(out, tuple) and len(out) == 2:
                    return out
                if isinstance(out, dict) and 'cos' in out and 'sin' in out:
                    return out['cos'], out['sin']
            except TypeError as e:
                # signature mismatch — try next
                last_exc = e
                continue
            except Exception as e:
                last_exc = e
                continue

        raise last_exc or RuntimeError("Rotary call failed with unknown error")


    def _compute_position_embeddings(self, batch: Dict[str, torch.Tensor], layer: Optional[torch.nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos, sin tensors shaped (batch, seq_len, head_dim).
        Try to call the rotary object; if that fails, compute cos/sin from rope_theta (robust fallback).
        """
        # extract device/seq_len/batch_size/hidden_size
        if 'hidden_states' in batch:
            hidden_states = batch['hidden_states']
            batch_size, seq_len, hidden_size = hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
            device = hidden_states.device
        elif 'input_ids' in batch:
            input_ids = batch['input_ids']
            batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
            device = input_ids.device
            hidden_size = getattr(self.config.config, 'hidden_size', None) or 4096
        else:
            raise ValueError("Batch must contain 'hidden_states' or 'input_ids' to compute position embeddings")

        # Determine head_dim (prefer layer if available)
        num_attention_heads = getattr(self.config.config, 'num_attention_heads', None) or 32
        head_dim = None
        if layer is not None:
            sa = getattr(layer, "self_attn", None)
            if sa is not None:
                head_dim = getattr(sa, "head_dim", None)
        if head_dim is None:
            head_dim = int(hidden_size) // int(num_attention_heads)

        # position_ids
        position_ids = batch.get('position_ids')
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Attempt to call existing rotary object if available
        if getattr(self, "_rotary_emb", None) is not None:
            try:
                # Inspect forward signature and try to call sensibly based on param names.
                sig = None
                try:
                    sig = inspect.signature(self._rotary_emb.forward)
                except Exception:
                    try:
                        sig = inspect.signature(self._rotary_emb.__call__)
                    except Exception:
                        sig = None

                if sig is not None:
                    params = [p for p in list(sig.parameters.keys()) if p != "self"]
                    # Build kwargs by parameter name mapping
                    call_kwargs = {}
                    for p in params:
                        lp = p.lower()
                        if 'position' in lp or 'pos' in lp:
                            call_kwargs[p] = position_ids
                        elif 'seq' in lp or 'seqlen' in lp or 'length' in lp or 'n' == lp:
                            # some implementations accept seq_len (positional)
                            call_kwargs[p] = seq_len
                        elif lp in ('device',):
                            call_kwargs[p] = device
                        elif lp in ('x', 'input', 'tensor', 'hidden_states', 'inp'):
                            # some forward implementations expect an 'x' input tensor.
                            # Provide a small dummy shaped (batch, seq_len, head_dim) OR the hidden_states projected shape.
                            # We will attempt to give a tensor with the right last dim (head_dim).
                            # Prefer a sliced view of hidden_states if head_dim matches last dim of hidden_states; otherwise provide zeros.
                            try:
                                if hidden_states.shape[-1] == head_dim:
                                    call_kwargs[p] = hidden_states
                                else:
                                    call_kwargs[p] = torch.zeros(batch_size, seq_len, head_dim, device=device, dtype=hidden_states.dtype)
                            except Exception:
                                call_kwargs[p] = torch.zeros(batch_size, seq_len, head_dim, device=device)
                    # Try calling with kwargs
                    out = self._rotary_emb(**call_kwargs)
                    if isinstance(out, tuple) and len(out) == 2:
                        cos, sin = out
                        # ensure shape -> (batch, seq_len, head_dim)
                        cos = _ensure_batch_seq_head(cos, batch_size, seq_len, head_dim, device)
                        sin = _ensure_batch_seq_head(sin, batch_size, seq_len, head_dim, device)
                        return cos, sin
                    if isinstance(out, dict) and 'cos' in out and 'sin' in out:
                        cos, sin = out['cos'], out['sin']
                        cos = _ensure_batch_seq_head(cos, batch_size, seq_len, head_dim, device)
                        sin = _ensure_batch_seq_head(sin, batch_size, seq_len, head_dim, device)
                        return cos, sin
                    # otherwise fallthrough to fallback
            except Exception as e:
                # don't fail hard — we'll compute cos/sin manually
                print(f"⚠️ Rotary call failed: {e}")

        # --- Fallback: compute cos & sin manually using rotary (standard implementation) ---
        # rope_theta / base
        rope_theta = getattr(self.config.config, 'rope_theta', None) or getattr(self.config.config, 'rope_base', None) or 10000.0

        # inv_freq: shape head_dim/2
        half_dim = head_dim // 2
        dtype = torch.get_default_dtype()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        # positions shape: seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # outer product -> (seq_len, half_dim)
        freqs = torch.einsum('i,j->ij', positions, inv_freq)  # seq_len x half_dim
        # turn into (seq_len, head_dim) by interleaving the freq twice
        # create (seq_len, half_dim, 1) expand then reshape
        # compute sin/cos on repeated pattern
        sin_part = torch.sin(freqs)  # seq_len x half_dim
        cos_part = torch.cos(freqs)  # seq_len x half_dim
        # interleave to full head dim: [cos0, cos0, cos1, cos1, ...] same for sin
        # One simple way: stack and reshape
        cos_full = torch.stack([cos_part, cos_part], dim=-1).reshape(seq_len, half_dim * 2)
        sin_full = torch.stack([sin_part, sin_part], dim=-1).reshape(seq_len, half_dim * 2)

        # if head_dim is odd (rare) pad/truncate
        if cos_full.shape[-1] != head_dim:
            cos_full = cos_full[:, :head_dim]
            sin_full = sin_full[:, :head_dim]

        # expand to batch shape: (batch, seq_len, head_dim)
        cos = cos_full.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        sin = sin_full.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        return cos, sin


    def _ensure_batch_seq_head(t: torch.Tensor, batch_size: int, seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
        """
        Normalize tensor t into shape (batch, seq_len, head_dim).
        Accepts shapes: (seq_len, head_dim), (batch, seq_len, head_dim), (1, seq_len, head_dim).
        """
        if t is None:
            return None
        if t.dim() == 2 and t.shape == (seq_len, head_dim):
            return t.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        if t.dim() == 3:
            if t.shape[0] == 1:
                return t.expand(batch_size, -1, -1).to(device)
            if t.shape[0] == batch_size and t.shape[1] == seq_len and t.shape[2] == head_dim:
                return t.to(device)
        # fallback: try reshape
        try:
            return t.reshape(batch_size, seq_len, head_dim).to(device)
        except Exception:
            # last resort: zeros
            return torch.zeros(batch_size, seq_len, head_dim, device=device)





    def _execute_transformer_layer(self, layer: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute transformer layer with proper position embeddings and robust dtype/shape handling."""
        hidden_states = batch.get("hidden_states")
        attention_mask = batch.get("attention_mask")

        if hidden_states is None:
            raise ValueError("Transformer layer requires 'hidden_states' in batch")

        # create kwargs early so we can safely assign into it
        kwargs: Dict[str, torch.Tensor] = {}

        # Basic shapes / device info
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_size = hidden_states.shape[-1]
        device = hidden_states.device

        # --- Normalize attention_mask: device + dtype acceptable to SDPA ---
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if attention_mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                a_min = int(attention_mask.min().item())
                a_max = int(attention_mask.max().item())
                if a_min >= 0 and a_max <= 1:
                    attention_mask = attention_mask.to(torch.bool)
                else:
                    attention_mask = attention_mask.to(hidden_states.dtype)
            else:
                if attention_mask.dtype != hidden_states.dtype and attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.to(hidden_states.dtype)

            # reshape mask to be broadcastable by SDPA: (batch, seq_len) -> (batch, 1, 1, seq_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.unsqueeze(2)

            batch['attention_mask'] = attention_mask
            kwargs['attention_mask'] = attention_mask

        # position_ids (if present) can be passed through
        if 'position_ids' in batch:
            kwargs['position_ids'] = batch['position_ids']

        # Compute position embeddings for Qwen models (pass layer to let it derive head_dim)
        cos, sin = self._compute_position_embeddings(batch, layer=layer)

        # Determine expected head_dim from layer if possible (preferred) else from config/hidden states
        num_attention_heads = getattr(self.config.config, 'num_attention_heads', None) or 32
        head_dim = None
        sa = getattr(layer, "self_attn", None)
        if sa is not None:
            head_dim = getattr(sa, "head_dim", None)
        if head_dim is None:
            try:
                head_dim = int(hidden_size) // int(num_attention_heads)
            except Exception:
                head_dim = hidden_size

        # Helper to expand/normalize cos/sin to (batch, seq_len, head_dim)
        def _expand_rotary(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if t.dim() == 2:
                return t.unsqueeze(0).expand(batch_size, -1, -1).to(device)
            if t.dim() == 3:
                if t.shape[0] == 1:
                    return t.expand(batch_size, -1, -1).to(device)
                return t.to(device)
            try:
                return t.reshape(batch_size, seq_len, head_dim).to(device)
            except Exception:
                return None

        # Validate/normalize cos & sin and attach them to kwargs
        if cos is not None and sin is not None:
            cos_e = _expand_rotary(cos)
            sin_e = _expand_rotary(sin)
            ok = (cos_e is not None and sin_e is not None and cos_e.shape[-1] == head_dim and sin_e.shape[-1] == head_dim)
            if not ok:
                print(f"⚠️ Rotary shape mismatch: expected head_dim={head_dim}, got cos={getattr(cos,'shape',None)}, sin={getattr(sin,'shape',None)}. Using zero-fallback.")
                cos_e = torch.zeros(batch_size, seq_len, head_dim, device=device, dtype=hidden_states.dtype)
                sin_e = torch.zeros_like(cos_e)
            kwargs['position_embeddings'] = (cos_e, sin_e)

        # Debug prints (temporary)
        if getattr(self, "_debug", True):
            if 'attention_mask' in kwargs:
                am = kwargs['attention_mask']
            pe = kwargs.get('position_embeddings')
            if pe:
                if isinstance(pe, (list, tuple)):
                    cos_shape = getattr(pe[0], "shape", None) if pe[0] is not None else None
                    sin_shape = getattr(pe[1], "shape", None) if pe[1] is not None else None

        # Now attempt to call layer with graceful fallbacks
        try:
            result = layer(hidden_states, **kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result
        except Exception as e:
            print(f"⚠️ Standard execution failed: {e}")
            try:
                result = layer(
                    hidden_states,
                    attention_mask=kwargs.get('attention_mask', None),
                    position_ids=batch.get('position_ids'),
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    position_embeddings=kwargs.get('position_embeddings', None)
                )
                if isinstance(result, tuple):
                    return result[0]
                return result
            except Exception as e2:
                print(f"⚠️ Extended parameter execution failed: {e2}")
                try:
                    result = layer(hidden_states)
                    if isinstance(result, tuple):
                        return result[0]
                    return result
                except Exception as e3:
                    raise RuntimeError(f"All transformer layer execution attempts failed. Last error: {e3}") from e




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
        
        # Ensure position_ids are present for transformer models
        if 'position_ids' not in current_batch and ('input_ids' in current_batch or 'hidden_states' in current_batch):
            if 'input_ids' in current_batch:
                seq_len = current_batch['input_ids'].shape[1]
                batch_size = current_batch['input_ids'].shape[0]
                device = current_batch['input_ids'].device
            else:
                seq_len = current_batch['hidden_states'].shape[1]
                batch_size = current_batch['hidden_states'].shape[0]
                device = current_batch['hidden_states'].device
                
            current_batch['position_ids'] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
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
                
                # Handle tuple returns (some layers return (hidden_states, attentions, etc.))
                if isinstance(output, tuple):
                    output = output[0]
                
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
                layer = LayerFactory.create_embedding_layer(state_dict, self.config.config)
            elif layer_type == "transformer":
                layer = LayerFactory.create_transformer_layer(state_dict, layer_name, self.config.config)
            elif layer_type == "norm":
                layer = LayerFactory.create_norm_layer(state_dict, self.config.config)
            elif layer_type == "lm_head":
                layer = LayerFactory.create_lm_head(state_dict, self.config.config)
            else:
                print(f"  ❌ Unknown layer type: {layer_type}")
                continue
            
            layer.eval()
            loaded_layers[layer_idx] = layer
        
        print(f"✅ Successfully loaded {len(loaded_layers)} layers")
        return loaded_layers

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


if __name__ == "__main__":
    import time
    repo_id = "Qwen/Qwen3-0.6B" # Example model
    local_dir = "models"
    max_new_tokens = 5

    config = ModelConfig(local_dir, repo_id)
    executor = MultiLayerExecutor(config)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    # Ensure tokenizer has a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_prompts = [
        "人工智能的未来取决于",
        "在下一个十年，机器学习将"
    ]

    inputs = tokenizer(test_prompts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids_initial = inputs['input_ids'] # Store initial input_ids
    attention_mask_initial = inputs['attention_mask'] # Store initial attention_mask

    # Create groups of layers for execution
    group_size = 4
    test_ranges = [(i, min(i + group_size - 1, config.total_layers - 1)) for i in range(0, config.total_layers, group_size)]

    # --- 1. Run Your Layer-by-Layer Implementation ---
    print("--- Running Layer-by-Layer Implementation ---")
    # Start with the initial inputs
    current_sequence_ids = input_ids_initial.clone() # This will grow with generated tokens
    current_attention_mask = attention_mask_initial.clone() # This will grow with generated tokens

    # Store logits from the last autoregressive step for comparison
    our_logits_at_end = None

    start_time = time.time()
    # Autoregressive generation loop
    for step in range(max_new_tokens):
        print(f"\n--- Layer-by-Layer Generation Step {step + 1}/{max_new_tokens} ---")
        # Prepare the batch for this *full* sequence step
        # Position IDs need to be for the entire current sequence
        current_position_ids = torch.arange(current_sequence_ids.shape[1], dtype=torch.long, device=current_sequence_ids.device).unsqueeze(0).expand(current_sequence_ids.shape[0], -1)

        current_batch_step = {
            'input_ids': current_sequence_ids.clone(), # Pass the *entire* sequence for embedding
            'attention_mask': current_attention_mask.clone(),
            'position_ids': current_position_ids
        }

        # --- Execute all layer groups for the current full sequence ---
        final_logits_or_hidden_states = None # To store the output of the last group
        for start_idx, end_idx in test_ranges:
            if end_idx >= config.total_layers:
                 end_idx = config.total_layers - 1
            # print(f"Executing range {start_idx}-{end_idx}") # Optional debug
            final_logits_or_hidden_states = executor.execute_layer_range(
                start_idx=start_idx,
                end_idx=end_idx,
                batch=current_batch_step,
                validate_shapes=False, # Set to True if you want strict shape checks
                keep_layers_loaded=True # Keep layers loaded for efficiency in this loop
            )
            # After executing a group, the output is either logits or hidden_states.
            # Update the batch for the *next* group *within the same step*.
            # Crucially, we don't update current_sequence_ids here.
            # We pass the full output (which becomes input hidden_states) to the next group.
            # Check if this was the last group (containing lm_head)
            if end_idx == config.total_layers - 1:
                 # The output from this group is the logits for the entire sequence
                 logits_full_sequence = final_logits_or_hidden_states
            else:
                 # The output is hidden_states for the next group
                 current_batch_step['hidden_states'] = final_logits_or_hidden_states

            # Remove input_ids after the embedding layer group has been processed (assuming group 0 contains embedding)
            if 'input_ids' in current_batch_step and start_idx == 0:
                del current_batch_step['input_ids']

        # --- End of step: Get logits and determine next token ---
        # At this point, logits_full_sequence contains the logits for the *entire* sequence of this step.
        # final_logits_or_hidden_states also holds the same value if the last group was executed.

        # Store logits only from the last autoregressive step for comparison
        # We want the logits for the *entire sequence* at the final step.
        if step == max_new_tokens - 1:
            our_logits_at_end = logits_full_sequence.clone().detach() # Shape: [batch_size, final_seq_len, vocab_size]

        # Get next tokens (greedy) - based on the logits of the *last token* in the sequence
        last_token_logits = logits_full_sequence[:, -1, :] # Get logits for the last generated token: [batch_size, vocab_size]
        next_tokens = torch.argmax(last_token_logits, dim=-1) # Greedy sampling: [batch_size]

        # --- Prepare inputs for the *next* autoregressive step ---
        # Append the newly generated tokens to the running sequence
        current_sequence_ids = torch.cat([current_sequence_ids, next_tokens.unsqueeze(1)], dim=1)
        # Extend the attention mask to cover the new token (assuming it's not masked)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_tokens).unsqueeze(1)], dim=1)
        # Note: current_position_ids for the *next* step will be recalculated based on the new sequence length above.

        # Optional: Print intermediate tokens
        for i, prompt in enumerate(test_prompts):
            next_token_id = next_tokens[i].item()
            next_token_text = tokenizer.decode([next_token_id])
            print(f"  Step {step+1}, Prompt {i+1}: ... -> Next token: '{next_token_text}' (ID: {next_token_id})")


    end_time = time.time()
    # Decode final output from your method
    final_texts_ours = tokenizer.batch_decode(current_sequence_ids, skip_special_tokens=True)
    print("\n--- Layer-by-Layer Generation Completed ---")
    for i, text in enumerate(final_texts_ours):
        print(f"Final Generated Text {i+1} (Ours): {text}")

    print(f"\nTotal Layers: {config.total_layers}")
    print(f"Layer-by-Layer Execution Time: {end_time - start_time:.2f} seconds")
    executor.unload_layers()


    # --- 2. Run Official Hugging Face Implementation ---
    print("\n--- Running Official Hugging Face Implementation ---")
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(config.model_path, trust_remote_code=True)
    hf_model.eval()
    print(f"Official HF Model Loaded: {type(hf_model)}")

    # Use the same initial inputs
    hf_input_ids = input_ids_initial.clone()
    hf_attention_mask = attention_mask_initial.clone()

    # Store logits from the last autoregressive step for comparison
    hf_logits_at_end = None

    start_time_hf = time.time()

    with torch.no_grad():
        for step in range(max_new_tokens):
            print(f"\n--- HF Generation Step {step + 1}/{max_new_tokens} ---")
            outputs = hf_model(input_ids=hf_input_ids, attention_mask=hf_attention_mask)
            logits = outputs.logits

            # Store logits only from the last autoregressive step for comparison
            # We want the logits for the *entire sequence* at the final step.
            if step == max_new_tokens - 1:
                hf_logits_at_end = logits.clone().detach() # Shape: [batch_size, final_seq_len, vocab_size]

            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            hf_input_ids = torch.cat([hf_input_ids, next_tokens.unsqueeze(1)], dim=-1)
            hf_attention_mask = torch.cat([hf_attention_mask, torch.ones_like(next_tokens).unsqueeze(1)], dim=-1)

            # Optional: Print intermediate tokens
            for i, prompt in enumerate(test_prompts):
                 next_token_id = next_tokens[i].item()
                 next_token_text = tokenizer.decode([next_token_id])
                 print(f"  Step {step+1}, Prompt {i+1}: ... -> Next token: '{next_token_text}' (ID: {next_token_id})")

    end_time_hf = time.time()
    final_texts_hf = tokenizer.batch_decode(hf_input_ids, skip_special_tokens=True)
    print("\n--- Official HF Generation Completed ---")
    for i, text in enumerate(final_texts_hf):
        print(f"Final Generated Text {i+1} (HF): {text}")

    print(f"HF Execution Time: {end_time_hf - start_time_hf:.2f} seconds")


    # --- 3. Compare Final Logits ---
    print("\n--- Comparing Final Step Logits ---")
    if our_logits_at_end is not None and hf_logits_at_end is not None:
         print(f"  Our logits shape: {our_logits_at_end.shape}")
         print(f"  HF logits shape:  {hf_logits_at_end.shape}")

         if our_logits_at_end.shape == hf_logits_at_end.shape:
             # Calculate difference metrics
             diff = torch.abs(our_logits_at_end - hf_logits_at_end)
             max_diff = torch.max(diff).item()
             mean_diff = torch.mean(diff).item()
             are_close = torch.allclose(our_logits_at_end, hf_logits_at_end, atol=1e-3, rtol=1e-4) # Slightly relaxed tolerance

             print(f"  Max absolute difference in logits: {max_diff:.6f}")
             print(f"  Mean absolute difference in logits: {mean_diff:.6f}")
             print(f"  Are logits approximately equal (atol=1e-3, rtol=1e-4)? {are_close}")

             if not are_close:
                 max_diff_idx = torch.argmax(diff)
                 max_diff_loc = torch.unravel_index(max_diff_idx, diff.shape)
                 print(f"  Max difference occurs at batch={max_diff_loc[0]}, seq={max_diff_loc[1]}, vocab={max_diff_loc[2]}")
                 print(f"    Our value:  {our_logits_at_end[max_diff_loc]:.6f}")
                 print(f"    HF value:   {hf_logits_at_end[max_diff_loc]:.6f}")
             else:
                 print("  ✅ Logits match within tolerance!")

         else:
             print("  ❌ Error: Logit shapes do not match!")
             # If shapes don't match, maybe compare the last token logits only?
             if our_logits_at_end.shape[2] == hf_logits_at_end.shape[2]: # Vocab size matches
                 our_last_logits = our_logits_at_end[:, -1, :] # [batch, vocab]
                 hf_last_logits = hf_logits_at_end[:, -1, :] # [batch, vocab]
                 if our_last_logits.shape == hf_last_logits.shape:
                     diff_last = torch.abs(our_last_logits - hf_last_logits)
                     max_diff_last = torch.max(diff_last).item()
                     mean_diff_last = torch.mean(diff_last).item()
                     are_close_last = torch.allclose(our_last_logits, hf_last_logits, atol=1e-3, rtol=1e-4)
                     print(f"  Comparing only LAST TOKEN logits:")
                     print(f"    Our last token logits shape: {our_last_logits.shape}")
                     print(f"    HF last token logits shape:  {hf_last_logits.shape}")
                     print(f"    Max abs diff (last token): {max_diff_last:.6f}")
                     print(f"    Mean abs diff (last token): {mean_diff_last:.6f}")
                     print(f"    Are last token logits close? {are_close_last}")
                     if not are_close_last:
                         max_diff_idx_last = torch.argmax(diff_last)
                         max_diff_loc_last = torch.unravel_index(max_diff_idx_last, diff_last.shape)
                         print(f"    Max diff (last token) at batch={max_diff_loc_last[0]}, vocab={max_diff_loc_last[1]}")
                         print(f"      Our value:  {our_last_logits[max_diff_loc_last]:.6f}")
                         print(f"      HF value:   {hf_last_logits[max_diff_loc_last]:.6f}")

    else:
        print("  ❌ Error: Could not retrieve logits from one or both methods for comparison.")
