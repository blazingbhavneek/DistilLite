import math
import os
import random

import dask
import nltk
import pandas as pd
import requests
from dask import delayed
from dask.distributed import Client
from datasets import load_dataset
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download("stopwords")

japanese_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt"
response = requests.get(japanese_stopwords_url)
japanese_stopwords = set(response.text.splitlines())

english_stopwords = set(stopwords.words("english"))

punctuation = set('.,!?:"";\'()[]{}-_=+*/\\')
numbers = set("0123456789")
spaces = set([" ", "\n", "\t"])

COMMON_TOKENS_TO_AVOID = (
    japanese_stopwords | english_stopwords | punctuation | numbers | spaces
)

DATASET_SELECTION_PERCENT = 0.1
MAX_SEQ_LENGTH = 100
MIN_OUTPUT_TOKENS = 10
MIN_TOTAL_TOKENS = 90
AVOID_COMMON_TOKENS = True
MAX_SELECTION_ATTEMPTS = 20


@delayed
def process_chunk(indices_chunk, dataset, tokenizer, chunk_id, total_chunks):
    """Process a chunk of selected indices with tokenization for next token prediction"""

    processed_rows = []
    skipped_common_tokens = 0

    for idx in indices_chunk:
        try:
            instruction = str(dataset[idx]["instruction"])
            input_text = str(dataset[idx]["input"])
            output_text = str(dataset[idx]["output"])

            instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
            input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
            output_tokens = tokenizer.encode(output_text, add_special_tokens=False)

            context_length = len(instruction_tokens) + len(input_tokens)
            if context_length >= MAX_SEQ_LENGTH:
                continue

            available_output_space = MAX_SEQ_LENGTH - context_length

            if len(output_tokens) < MIN_OUTPUT_TOKENS:
                continue

            target_position = None
            next_token = None
            selection_attempts = 0

            if AVOID_COMMON_TOKENS and len(output_tokens) > MIN_OUTPUT_TOKENS:
                available_positions = []

                if len(output_tokens) > available_output_space:
                    max_pos = min(MIN_OUTPUT_TOKENS - 1, available_output_space - 1)
                    start_pos = max(0, max_pos - 50)
                    end_pos = min(len(output_tokens) - 1, max_pos + 50)
                else:
                    start_pos = 0
                    end_pos = len(output_tokens) - 1

                for pos in range(start_pos, end_pos + 1):
                    token_text = tokenizer.decode([output_tokens[pos]]).strip()
                    if token_text not in COMMON_TOKENS_TO_AVOID and len(token_text) > 1:
                        available_positions.append(pos)

                if available_positions:
                    target_position = random.choice(available_positions)
                    next_token = output_tokens[target_position]
                else:
                    skipped_common_tokens += 1
                    if len(output_tokens) > available_output_space:
                        target_position = min(
                            MIN_OUTPUT_TOKENS - 1, available_output_space - 1
                        )
                    else:
                        target_position = random.randint(0, len(output_tokens) - 1)
                    next_token = output_tokens[target_position]
            else:
                if len(output_tokens) > available_output_space:
                    target_position = min(
                        MIN_OUTPUT_TOKENS - 1, available_output_space - 1
                    )
                else:
                    target_position = random.randint(0, len(output_tokens) - 1)
                next_token = output_tokens[target_position]

            partial_output_tokens = output_tokens[:target_position]

            context_tokens = instruction_tokens + input_tokens + partial_output_tokens

            total_tokens = len(context_tokens) + 1
            if total_tokens < MIN_TOTAL_TOKENS:
                continue

            context_text = tokenizer.decode(context_tokens)
            next_token_text = tokenizer.decode([next_token])

            is_common_token = (
                next_token_text.strip() in COMMON_TOKENS_TO_AVOID
                or len(next_token_text.strip()) <= 1
            )

            processed_rows.append(
                {
                    "context_text": context_text,
                    "next_token_text": next_token_text,
                    "total_tokens": total_tokens,
                    "original_instruction": instruction,
                    "original_input": input_text,
                    "original_output": output_text,
                    "target_position": target_position,
                    "output_length": len(output_tokens),
                    "is_common_token": is_common_token,
                    "token_length": len(next_token_text.strip()),
                }
            )

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

    chunk_df = pd.DataFrame(processed_rows)

    print(
        f"Processed chunk {chunk_id}/{total_chunks} - Kept {len(chunk_df)} rows (Skipped {skipped_common_tokens} due to common tokens)"
    )

    return chunk_df


def main():
    """Main function to run the processing"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Starting Dask client...")
    client = Client(processes=True, n_workers=2, threads_per_worker=2)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset("izumi-lab/llm-japanese-dataset")
    dataset = dataset["train"]

    total_rows = len(dataset)
    selected_rows = int(total_rows * DATASET_SELECTION_PERCENT / 100)
    print(
        f"Selecting {selected_rows} rows ({DATASET_SELECTION_PERCENT}%) from {total_rows} total rows"
    )

    random.seed(42)
    selected_indices = random.sample(range(total_rows), selected_rows)
    selected_indices.sort()

    chunk_size = min(5000, len(selected_indices) // 4)
    n_chunks = math.ceil(len(selected_indices) / chunk_size)

    chunks = []
    for i in range(0, len(selected_indices), chunk_size):
        end_idx = min(i + chunk_size, len(selected_indices))
        indices_chunk = selected_indices[i:end_idx]
        chunk_id = (i // chunk_size) + 1
        chunk_task = process_chunk(
            indices_chunk, dataset, tokenizer, chunk_id, n_chunks
        )
        chunks.append(chunk_task)

    print(f"Created {len(chunks)} tasks for {len(selected_indices)} selected samples")

    print("Computing chunks in parallel...")
    try:
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:

            def update_progress(*args):
                pbar.update(1)

            for chunk in chunks:
                chunk.then(update_progress)

            filtered_chunks = dask.compute(*chunks)
    except Exception as e:
        print(f"Error during parallel computation: {e}")
        print("Falling back to sequential processing...")
        filtered_chunks = []
        for chunk in tqdm(chunks, desc="Processing chunks sequentially"):
            try:
                result = chunk.compute()
                filtered_chunks.append(result)
            except Exception as chunk_error:
                print(f"Error in chunk: {chunk_error}")
                continue

    print("Combining processed chunks...")
    processed_df = pd.concat(
        [chunk for chunk in filtered_chunks if len(chunk) > 0], ignore_index=True
    )

    print(f"\n=== PROCESSING STATISTICS ===")
    print(f"Original dataset size: {total_rows:,}")
    print(
        f"Selected for processing: {len(selected_indices):,} ({DATASET_SELECTION_PERCENT}%)"
    )
    print(f"Successfully processed: {len(processed_df):,}")
    if len(selected_indices) > 0:
        print(f"Success rate: {len(processed_df)/len(selected_indices)*100:.2f}%")

    csv_path = os.path.join(results_dir, "next_token_prediction_dataset.csv")
    print("\nSaving processed dataset...")
    processed_df.to_csv(csv_path, index=False)

    if len(processed_df) > 0:
        print(
            f"\nAnalysis complete! Saved {len(processed_df):,} samples for next token prediction training."
        )
        print("Files saved:")
        print(f"- {csv_path} (main dataset)")
    else:
        print("No data was successfully processed. Check your filtering criteria.")

    client.close()


if __name__ == "__main__":
    main()
