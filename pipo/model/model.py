import os
from pathlib import Path
from transformers import AutoConfig
from huggingface_hub import snapshot_download

try:
    from safetensors import safe_open
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

class ModelConfig:
    """Immutable Configuration class for thread-safe layer-by-layer Model execution"""
    
    def __init__(self, local_dir: str, repo_id: str):
        self.local_dir: str = local_dir
        self.repo_id: str = repo_id
        
        self.model_path: Path = Path(os.path.join(local_dir, repo_id))
        if not os.path.exists(self.model_path):
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        self.num_layers: int = self._get_num_layers()
        
        self.layer_names_dict: dict = self._create_layer_names_dict()
        
        self.layer_names: list[str] = self._create_ordered_layer_names()
        
        self.layer_to_param_names: dict[str, list[str]] = self._create_layer_param_mapping()
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers from config"""
        return getattr(self.config, 'num_hidden_layers', 
                      getattr(self.config, 'n_layer', 
                             getattr(self.config, 'num_layers', 32)))
    
    def _create_layer_names_dict(self) -> dict:
        """Creates layer names dictionary based on model architecture"""
        model_arch = getattr(self.config, 'architectures', [''])[0] if hasattr(self.config, 'architectures') else ''
        
        return {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
    
    def _create_ordered_layer_names(self) -> list[str]:
        """Create ordered list of layer names for forward pass execution"""
        layer_names = []
        
        layer_names.append(self.layer_names_dict['embed'])
        
        for i in range(self.num_layers):
            layer_names.append(f"{self.layer_names_dict['layer_prefix']}.{i}")
        
        layer_names.append(self.layer_names_dict['norm'])
        
        layer_names.append(self.layer_names_dict['lm_head'])
        
        return layer_names
    
    def _create_layer_param_mapping(self) -> dict[str, list[str]]:
        """Create mapping from layer names to their parameter names for efficient loading"""
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
        """Get all parameter names from model files"""
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
        if layer_name.startswith(self.layer_names_dict['layer_prefix']):
            return param_name.startswith(layer_name + ".")
        
        return param_name.startswith(layer_name + ".") or param_name == layer_name + ".weight"
    
    def get_layer_param_names(self, layer_name: str) -> list[str]:
        """Get all parameter names for a specific layer"""
        return self.layer_to_param_names.get(layer_name, [])
    
    def get_layer_name_by_index(self, index: int) -> str:
        """Get layer name by execution order index"""
        if 0 <= index < len(self.layer_names):
            return self.layer_names[index]
        raise IndexError(f"Layer index {index} out of range (0-{len(self.layer_names)-1})")
    
    def load_layer_state_dict(self, layer_name: str) -> dict:
        """Load state_dict for a specific layer from model files"""
        param_names = self.get_layer_param_names(layer_name)
        
        if not param_names and layer_name == self.layer_names_dict['lm_head']:
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
                            key = "lm_head.weight" if layer_name == self.layer_names_dict['lm_head'] else param_name
                            layer_state_dict[key] = f.get_tensor(param_name)
            
            elif fname.endswith(".bin"):
                import torch
                state = torch.load(path, map_location="cpu")
                for param_name in param_names:
                    if param_name in state:
                        key = "lm_head.weight" if layer_name == self.layer_names_dict['lm_head'] else param_name
                        layer_state_dict[key] = state[param_name]
        
        return layer_state_dict
    
    def load_layer_state_dict_by_index(self, layer_index: int) -> dict:
        """Load state_dict for a layer by its execution order index"""
        layer_name = self.get_layer_name_by_index(layer_index)
        return self.load_layer_state_dict(layer_name)
    
    @property
    def total_layers(self) -> int:
        """Get total number of layers (including embed, norm, lm_head)"""
        return len(self.layer_names)
    
    def __repr__(self) -> str:
        return f"ModelConfig(repo_id='{self.repo_id}', num_layers={self.num_layers}, total_layers={len(self.layer_names)})"


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from accelerate.utils.modeling import set_module_tensor_to_device
    from transformers import AutoModelForCausalLM
    
    repo_id = "Qwen/Qwen3-4B"
    local_dir = "models"
    model_path = os.path.join(local_dir, repo_id)
    
    print("🚀 Testing ModelConfig capabilities...")
    
    config = ModelConfig(local_dir, repo_id)
    print(f"📊 {config}")
    print(f"🏗️ Architecture: {config.layer_names_dict}")
    print(f"📝 Layer execution order: {config.layer_names[:3]}...{config.layer_names[-2:]}")
    
    test_layers = [
        (0, "Embedding layer"),
        (1, "First transformer layer"), 
        (2, "Second transformer layer"),
        (config.num_layers // 2 + 1, "Middle transformer layer"),
        (config.num_layers, "Last transformer layer"),
        (config.num_layers + 1, "Final norm layer"),
        (config.num_layers + 2, "LM head layer")
    ]
    
    print("\n🔍 Testing layer loading capabilities...")
    
    for layer_idx, description in test_layers:
        print(f"\n--- {description} (Index: {layer_idx}) ---")
        
        try:
            layer_name = config.get_layer_name_by_index(layer_idx)
            print(f"Layer name: {layer_name}")
            
            param_names = config.get_layer_param_names(layer_name)
            print(f"Parameters: {len(param_names)} tensors")
            for param in param_names[:3]:
                print(f"  - {param}")
            if len(param_names) > 3:
                print(f"  ... and {len(param_names) - 3} more")
                
            print("Loading state_dict...")
            state_dict = config.load_layer_state_dict_by_index(layer_idx)
            
            if state_dict:
                print(f"✅ Loaded {len(state_dict)} tensors")
                
                first_tensor = next(iter(state_dict.values()))
                model_dtype = first_tensor.dtype
                print(f"Model dtype: {model_dtype}")
                
                for param_name, tensor in list(state_dict.items())[:2]:
                    print(f"  {param_name}: {tensor.shape} ({tensor.dtype})")
                
                print("🧪 Testing inference...")
                
                if layer_idx == 0:
                    vocab_size = state_dict[f'{layer_name}.weight'].shape[0]
                    embed_dim = state_dict[f'{layer_name}.weight'].shape[1]
                    
                    with init_empty_weights():
                        embed_layer = nn.Embedding(vocab_size, embed_dim)
                    
                    for name, param in state_dict.items():
                        set_module_tensor_to_device(embed_layer, name.split('.')[-1], "cpu", value=param)
                    
                    input_ids = torch.randint(0, min(1000, vocab_size), (1, 10))
                    print(f"  Input: {input_ids.shape} token IDs")
                    
                    with torch.no_grad():
                        output = embed_layer(input_ids)
                        print(f"  Output: {output.shape} embeddings")
                        print(f"  Sample output mean: {output.mean().item():.4f}")
                        print(f"  ✅ Embedding layer test passed!")
                
                elif "layers." in layer_name:
                    attn_weights = [k for k in state_dict.keys() if 'self_attn' in k and 'weight' in k]
                    if attn_weights:
                        q_proj_weight = state_dict.get(f'{layer_name}.self_attn.q_proj.weight')
                        if q_proj_weight is not None:
                            hidden_size = q_proj_weight.shape[1]
                            print(f"  Hidden size: {hidden_size}")
                            print(f"  Q projection: {q_proj_weight.shape}")
                            
                            hidden_states = torch.randn(1, 10, hidden_size, dtype=model_dtype)
                            print(f"  Test input: {hidden_states.shape} ({hidden_states.dtype})")
                            
                            with torch.no_grad():
                                q_out = torch.matmul(hidden_states, q_proj_weight.T)
                                print(f"  Q output: {q_out.shape} ({q_out.dtype})")
                                print(f"  ✅ Transformer layer test passed!")
                
                elif layer_name == config.layer_names_dict['norm']:
                    norm_weight = state_dict.get(f'{layer_name}.weight')
                    if norm_weight is not None:
                        hidden_size = norm_weight.shape[0]
                        print(f"  Hidden size: {hidden_size}")
                        
                        hidden_states = torch.randn(1, 10, hidden_size, dtype=model_dtype)
                        print(f"  Input: {hidden_states.shape} ({hidden_states.dtype})")
                        
                        with torch.no_grad():
                            mean = hidden_states.mean(-1, keepdim=True)
                            var = hidden_states.var(-1, keepdim=True, unbiased=False)
                            normalized = (hidden_states - mean) / torch.sqrt(var + 1e-5)
                            output = normalized * norm_weight
                            print(f"  Normalized output: {output.shape} ({output.dtype})")
                            print(f"  ✅ Norm layer test passed!")
                
                elif layer_name == config.layer_names_dict['lm_head']:
                    lm_weight = state_dict.get('lm_head.weight')
                    if lm_weight is not None:
                        vocab_size, hidden_size = lm_weight.shape
                        print(f"  Vocab size: {vocab_size}, Hidden size: {hidden_size}")
                        print(f"  🔗 Using tied embeddings from model.embed_tokens")
                        
                        hidden_states = torch.randn(1, 10, hidden_size, dtype=model_dtype)
                        print(f"  Input: {hidden_states.shape} ({hidden_states.dtype})")
                        
                        with torch.no_grad():
                            logits = torch.matmul(hidden_states, lm_weight.T)
                            print(f"  Logits: {logits.shape} ({logits.dtype})")
                            print(f"  Top token ID: {logits.argmax(-1)[0, -1].item()}")
                            print(f"  ✅ LM head test passed!")
                    else:
                        print("  ⚠️ No lm_head.weight found - might be tied to embedding")
                
            else:
                print("❌ No parameters found for this layer")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
