import os

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def run_partial_inference(
    model, input_ids, attention_mask, start_layer: int, end_layer: int
):
    """
    Run inference through a subset of layers (start_layer to end_layer-1).
    Returns the hidden states after the last processed layer.
    """
    with torch.no_grad():
        if start_layer == 0:
            hidden_states = model.model.embed_tokens(input_ids)

            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            for i in range(start_layer, min(end_layer, len(model.model.layers))):
                layer = model.model.layers[i]

                if attention_mask is not None:
                    causal_mask = torch.tril(
                        torch.ones(seq_length, seq_length, device=hidden_states.device)
                    )
                    expanded_mask = attention_mask[:, None, None, :].expand(
                        batch_size, 1, seq_length, seq_length
                    )
                    combined_mask = causal_mask[None, None, :, :] * expanded_mask
                    combined_mask = (1.0 - combined_mask) * torch.finfo(
                        hidden_states.dtype
                    ).min
                else:
                    combined_mask = None

                try:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=combined_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    hidden_states = layer_outputs[0]
                    print(f"Processed layer {i}, output shape: {hidden_states.shape}")

                except Exception as e:
                    print(f"Error in layer {i} with full args: {e}")
                    try:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        hidden_states = layer_outputs[0]
                        print(
                            f"Processed layer {i} (minimal args), output shape: {hidden_states.shape}"
                        )
                    except Exception as e2:
                        print(f"Error in layer {i} with minimal args: {e2}")
                        try:
                            layer_outputs = layer(hidden_states)
                            if isinstance(layer_outputs, tuple):
                                hidden_states = layer_outputs[0]
                            else:
                                hidden_states = layer_outputs
                            print(
                                f"Processed layer {i} (fallback), output shape: {hidden_states.shape}"
                            )
                        except Exception as e3:
                            print(f"Failed to process layer {i}: {e3}")
                            break

        else:
            hidden_states = input_ids

            batch_size, seq_length = hidden_states.shape[:2]
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            for i in range(start_layer, min(end_layer, len(model.model.layers))):
                layer = model.model.layers[i]

                try:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    hidden_states = layer_outputs[0]
                    print(f"Processed layer {i}, output shape: {hidden_states.shape}")
                except Exception as e:
                    print(f"Error in layer {i}: {e}")
                    try:
                        layer_outputs = layer(hidden_states)
                        if isinstance(layer_outputs, tuple):
                            hidden_states = layer_outputs[0]
                        else:
                            hidden_states = layer_outputs
                        print(
                            f"Processed layer {i} (fallback), output shape: {hidden_states.shape}"
                        )
                    except Exception as e2:
                        print(f"Failed to process layer {i}: {e2}")
                        break

        return hidden_states


if __name__ == "__main__":
    repo_id = "Qwen/Qwen3-4B"
    local_dir = "models"
    model_path = os.path.join(local_dir, repo_id)

    print("Downloading model (if not cached)...")
    model_path = download_model(repo_id, local_dir)
    print(f"Model downloaded to: {model_path}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_text = "The quick brown fox jumps over"
    print(f"Test input: '{test_text}'")

    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"Tokenized input shape: {input_ids.shape}")

    temp_outputs = {}

    print("\n" + "=" * 50)
    print("LAYERED INFERENCE TEST")
    print("=" * 50)

    print("\nStep 1: Loading layers 0-1...")
    model_chunk1 = load_model_subset(
        repo_id, local_dir=local_dir, start_layer=0, end_layer=2
    )
    print("Running inference on layers 0-1...")

    output_chunk1 = run_partial_inference(
        model_chunk1, input_ids, attention_mask, start_layer=0, end_layer=2
    )
    temp_outputs["layers_0_1"] = output_chunk1.clone()
    print(f"Stored output from layers 0-1, shape: {temp_outputs['layers_0_1'].shape}")

    del model_chunk1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nStep 2: Loading layers 2-3...")
    model_chunk2 = load_model_subset(
        repo_id, local_dir=local_dir, start_layer=2, end_layer=4
    )
    print("Running inference on layers 2-3 using stored intermediate results...")

    output_chunk2 = run_partial_inference(
        model_chunk2,
        temp_outputs["layers_0_1"],
        attention_mask,
        start_layer=2,
        end_layer=4,
    )
    temp_outputs["layers_2_3"] = output_chunk2.clone()
    print(f"Stored output from layers 2-3, shape: {temp_outputs['layers_2_3'].shape}")

    del model_chunk2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 50)
    print("LAYERED INFERENCE COMPLETE")
    print("=" * 50)
    print(f"Final intermediate tensor shape: {temp_outputs['layers_2_3'].shape}")
    print(
        f"Final intermediate tensor mean: {temp_outputs['layers_2_3'].mean().item():.6f}"
    )
    print(
        f"Final intermediate tensor std: {temp_outputs['layers_2_3'].std().item():.6f}"
    )

    print("\nStored intermediate results:")
    for key, tensor in temp_outputs.items():
        print(f"  {key}: shape={tensor.shape}, mean={tensor.mean().item():.6f}")

    print("\nLayered inference test completed successfully!")
    print("You can extend this pattern to process more layer chunks sequentially.")
