import os

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from safetensors import safe_open

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def download_model(repo_id: str, local_dir: str = "models") -> str:
    """
    Download a model from HF Hub into cache_dir and return the local path.
    """
    model_path = os.path.join(local_dir, repo_id)
    if not os.path.exists(model_path):
        snapshot_download(repo_id=repo_id, local_dir=model_path)
    return model_path


def inspect_model_layers(model_dir: str) -> list[dict]:
    """
    Inspect all parameter shards in model_dir (.safetensors or .bin) and return per-layer metadata.
    Each entry contains:
      - name
      - shape
      - dtype
      - size_bytes
    """
    infos = []
    for fname in os.listdir(model_dir):
        path = os.path.join(model_dir, fname)
        if fname.endswith(".safetensors") and _HAS_SAFETENSORS:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    shape = tuple(tensor.shape)
                    dtype = str(tensor.dtype)
                    numel = int(np.prod(shape))
                    size_bytes = numel * tensor.element_size()
                    infos.append(
                        {
                            "shard": fname,
                            "name": key,
                            "shape": shape,
                            "dtype": dtype,
                            "size_bytes": size_bytes,
                        }
                    )

        elif fname.endswith(".bin"):
            state = torch.load(path, map_location="meta")
            for key, tensor in state.items():
                shape = tuple(tensor.shape)
                dtype = tensor.dtype
                numel = int(np.prod(shape))
                size_bytes = numel * torch.tensor([], dtype=dtype).element_size()
                infos.append(
                    {
                        "shard": fname,
                        "name": key,
                        "shape": shape,
                        "dtype": str(dtype),
                        "size_bytes": size_bytes,
                    }
                )

    return infos


def load_model_subset(
    repo_id: str,
    start_layer: int,
    end_layer: int,
    local_dir: str = "models",
) -> AutoModelForCausalLM:
    """
    Lazily initialize an empty model, then load only layers in [start_layer, end_layer) onto CPU.
    Other layers are offloaded to disk.
    """
    model_path = os.path.join(local_dir, repo_id)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = {}
    for i in range(config.num_hidden_layers):
        layer_name = f"model.layers.{i}"
        if start_layer <= i < end_layer:
            device_map[layer_name] = "cpu"
        else:
            device_map[layer_name] = "disk"

    device_map.update(
        {
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
            "model.norm": "cpu",
        }
    )

    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map=device_map,
        no_split_module_classes=[model.__class__.__name__],
        offload_folder=os.path.join(model_path, "offload"),
    )

    return model


if __name__ == "__main__":
    repo_id = "Qwen/Qwen3-4B"
    local_dir = "models"
    model_path = os.path.join(local_dir, repo_id)

    print("Downloading model (if not cached)...")
    model_path = download_model(repo_id, local_dir)
    print(f"Model downloaded to: {model_path}")

    print("\nInspecting model layers...")
    layer_infos = inspect_model_layers(model_path)
    print(f"Found {len(layer_infos)} parameters.\n")

    for info in layer_infos:
        print(
            f"{info['name']} | shape: {info['shape']} | dtype: {info['dtype']} | size: {info['size_bytes'] / 1e6:.2f} MB"
        )

    print("\nLoading subset of model layers onto CPU (layers 0–2)...")
    try:
        model = load_model_subset(
            repo_id, local_dir=local_dir, start_layer=0, end_layer=2
        )
        print("Model subset loaded successfully.")
    except Exception as e:
        print(f"Error loading model subset: {e}")
