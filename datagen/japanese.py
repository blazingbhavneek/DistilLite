import math
import random

import dask
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
import seaborn as sns
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

DATASET_SELECTION_PERCENT = 1
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
                    "context_tokens": context_tokens,
                    "next_token": next_token,
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

    print("\nSaving processed dataset...")
    processed_df.to_csv("results/next_token_prediction_dataset.csv", index=False)

    print("Saving tokenized dataset (pickle)...")
    processed_df.to_pickle("results/next_token_prediction_dataset.pkl")

    if len(processed_df) > 0:
        print("\n" + "=" * 50)
        print("ANALYZING GENERATED DATASET")
        print("=" * 50)

        print("Calculating component token lengths...")
        processed_df["instruction_tokens"] = processed_df["original_instruction"].apply(
            lambda x: len(tokenizer.encode(str(x), add_special_tokens=False))
        )
        processed_df["input_tokens"] = processed_df["original_input"].apply(
            lambda x: len(tokenizer.encode(str(x), add_special_tokens=False))
        )
        processed_df["output_tokens"] = processed_df["output_length"]
        processed_df["context_tokens_len"] = processed_df["context_tokens"].apply(len)

        print("\n=== TOKEN LENGTH STATISTICS ===")
        token_stats = processed_df[
            [
                "instruction_tokens",
                "input_tokens",
                "output_tokens",
                "context_tokens_len",
                "total_tokens",
            ]
        ].describe()
        print(token_stats)

        print("\n=== SAMPLE PROCESSED ROWS ===")
        sample_cols = [
            "context_tokens_len",
            "next_token",
            "next_token_text",
            "target_position",
            "total_tokens",
            "is_common_token",
        ]
        print(processed_df[sample_cols].head(10))

        print("\n=== TOKEN QUALITY ANALYSIS ===")
        common_tokens_count = len(processed_df[processed_df["is_common_token"] == True])
        rare_tokens_count = len(processed_df[processed_df["is_common_token"] == False])

        print(
            f"Common/boring tokens selected: {common_tokens_count} ({common_tokens_count/len(processed_df)*100:.1f}%)"
        )
        print(
            f"Rare/interesting tokens selected: {rare_tokens_count} ({rare_tokens_count/len(processed_df)*100:.1f}%)"
        )

        avg_common_token_len = processed_df[processed_df["is_common_token"] == True][
            "token_length"
        ].mean()
        avg_rare_token_len = processed_df[processed_df["is_common_token"] == False][
            "token_length"
        ].mean()

        print(f"Average common token length: {avg_common_token_len:.2f} chars")
        print(f"Average rare token length: {avg_rare_token_len:.2f} chars")

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle("Token Length Distributions for Next Token Prediction Dataset")

        axes[0, 0].hist(
            processed_df["total_tokens"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[0, 0].axvline(
            MAX_SEQ_LENGTH,
            color="red",
            linestyle="--",
            label=f"Max Length ({MAX_SEQ_LENGTH})",
        )
        axes[0, 0].set_title("Total Token Length Distribution")
        axes[0, 0].set_xlabel("Total Tokens")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        axes[0, 1].hist(
            processed_df["context_tokens_len"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[0, 1].set_title("Context Token Length Distribution")
        axes[0, 1].set_xlabel("Context Tokens")
        axes[0, 1].set_ylabel("Frequency")

        axes[0, 2].hist(
            processed_df["target_position"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[0, 2].axvline(
            MIN_OUTPUT_TOKENS - 1,
            color="red",
            linestyle="--",
            label=f"Min Position ({MIN_OUTPUT_TOKENS-1})",
        )
        axes[0, 2].set_title("Target Token Position Distribution")
        axes[0, 2].set_xlabel("Position in Output")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].legend()

        axes[1, 0].hist(
            processed_df["instruction_tokens"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[1, 0].set_title("Instruction Token Length Distribution")
        axes[1, 0].set_xlabel("Tokens")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].hist(
            processed_df["input_tokens"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[1, 1].set_title("Input Token Length Distribution")
        axes[1, 1].set_xlabel("Tokens")
        axes[1, 1].set_ylabel("Frequency")

        axes[1, 2].hist(
            processed_df["output_tokens"], bins=50, edgecolor="black", alpha=0.7
        )
        axes[1, 2].axvline(
            MIN_OUTPUT_TOKENS,
            color="red",
            linestyle="--",
            label=f"Min Required ({MIN_OUTPUT_TOKENS})",
        )
        axes[1, 2].set_title("Original Output Token Length Distribution")
        axes[1, 2].set_xlabel("Tokens")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].legend()

        common_mask = processed_df["is_common_token"] == True
        rare_mask = processed_df["is_common_token"] == False

        axes[2, 0].hist(
            [
                processed_df[common_mask]["token_length"],
                processed_df[rare_mask]["token_length"],
            ],
            bins=30,
            alpha=0.7,
            label=["Common tokens", "Rare tokens"],
            edgecolor="black",
        )
        axes[2, 0].set_title("Target Token Length by Quality")
        axes[2, 0].set_xlabel("Token Length (chars)")
        axes[2, 0].set_ylabel("Frequency")
        axes[2, 0].legend()

        axes[2, 1].hist(
            [
                processed_df[common_mask]["target_position"],
                processed_df[rare_mask]["target_position"],
            ],
            bins=30,
            alpha=0.7,
            label=["Common tokens", "Rare tokens"],
            edgecolor="black",
        )
        axes[2, 1].set_title("Target Position by Token Quality")
        axes[2, 1].set_xlabel("Position in Output")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].legend()

        axes[2, 2].hist(
            [
                processed_df[common_mask]["total_tokens"],
                processed_df[rare_mask]["total_tokens"],
            ],
            bins=30,
            alpha=0.7,
            label=["Common tokens", "Rare tokens"],
            edgecolor="black",
        )
        axes[2, 2].set_title("Total Tokens by Target Quality")
        axes[2, 2].set_xlabel("Total Tokens")
        axes[2, 2].set_ylabel("Frequency")
        axes[2, 2].legend()

        plt.tight_layout()
        plt.savefig("results/token_distributions.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\n=== TOKEN LENGTH CORRELATIONS ===")
        correlation_cols = [
            "instruction_tokens",
            "input_tokens",
            "output_tokens",
            "context_tokens_len",
            "total_tokens",
        ]
        correlation_matrix = processed_df[correlation_cols].corr()
        print(correlation_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".3f",
            square=True,
        )
        plt.title("Correlation Matrix of Token Lengths")
        plt.tight_layout()
        plt.savefig("results/token_correlations.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\n=== TARGET POSITION ANALYSIS ===")
        random_positions = processed_df[
            processed_df["target_position"] < (MIN_OUTPUT_TOKENS - 1)
        ]
        fixed_positions = processed_df[
            processed_df["target_position"] == (MIN_OUTPUT_TOKENS - 1)
        ]

        print(
            f"Random position selections: {len(random_positions)} ({len(random_positions)/len(processed_df)*100:.1f}%)"
        )
        print(
            f"Fixed position (100th token): {len(fixed_positions)} ({len(fixed_positions)/len(processed_df)*100:.1f}%)"
        )

        print("\n=== EXTREME CASES ===")
        print("Longest total token sequences:")
        longest = processed_df.nlargest(5, "total_tokens")[
            ["context_tokens_len", "total_tokens", "target_position", "next_token_text"]
        ]
        print(longest)

        print("\nShortest total token sequences:")
        shortest = processed_df.nsmallest(5, "total_tokens")[
            ["context_tokens_len", "total_tokens", "target_position", "next_token_text"]
        ]
        print(shortest)

        print("\n=== DATA QUALITY CHECKS ===")
        empty_instructions = len(
            processed_df[
                processed_df["original_instruction"].astype(str).str.strip() == ""
            ]
        )
        empty_inputs = len(
            processed_df[processed_df["original_input"].astype(str).str.strip() == ""]
        )
        empty_outputs = len(
            processed_df[processed_df["original_output"].astype(str).str.strip() == ""]
        )

        print(f"Empty instructions: {empty_instructions}")
        print(f"Empty inputs: {empty_inputs}")
        print(f"Empty outputs: {empty_outputs}")

        print("\n=== TOKEN EFFICIENCY ANALYSIS ===")
        context_ratio = (
            processed_df["context_tokens_len"] / processed_df["total_tokens"]
        )
        print(f"Average context ratio: {context_ratio.mean():.3f}")
        print(
            f"Context ratio range: {context_ratio.min():.3f} - {context_ratio.max():.3f}"
        )

        print("\n=== NEXT TOKEN ANALYSIS ===")
        token_freq = processed_df["next_token_text"].value_counts().head(20)
        print("Most frequent next tokens:")
        print(token_freq)

        unique_tokens = processed_df["next_token"].nunique()
        total_samples = len(processed_df)
        print(
            f"\nToken diversity: {unique_tokens:,} unique next tokens out of {total_samples:,} samples"
        )
        print(f"Diversity ratio: {unique_tokens/total_samples:.3f}")

        print("\n=== CHARACTER-TO-TOKEN RATIO ANALYSIS ===")
        char_lengths = (
            processed_df["original_instruction"].astype(str).str.len()
            + processed_df["original_input"].astype(str).str.len()
            + processed_df["original_output"].astype(str).str.len()
        )

        total_original_tokens = (
            processed_df["instruction_tokens"]
            + processed_df["input_tokens"]
            + processed_df["output_tokens"]
        )

        char_to_token_ratio = char_lengths / total_original_tokens
        print(f"Average characters per token: {char_to_token_ratio.mean():.2f}")
        print(
            f"Character-to-token ratio range: {char_to_token_ratio.min():.2f} - {char_to_token_ratio.max():.2f}"
        )

        with open("results/dataset_analysis.txt", "w", encoding="utf-8") as f:
            f.write("NEXT TOKEN PREDICTION DATASET ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Selection: {DATASET_SELECTION_PERCENT}% of original\n")
            f.write(f"Total Processed Samples: {len(processed_df):,}\n")
            f.write(
                f"Average Total Tokens: {processed_df['total_tokens'].mean():.1f}\n"
            )
            f.write(
                f"Average Context Length: {processed_df['context_tokens_len'].mean():.1f}\n"
            )
            f.write(f"Random Position Selections: {len(random_positions):,}\n")
            f.write(f"Fixed Position Selections: {len(fixed_positions):,}\n")
            f.write(f"Token Diversity: {unique_tokens:,} unique tokens\n")
            f.write(f"Average Context Ratio: {context_ratio.mean():.3f}\n")
            f.write(f"Average Chars per Token: {char_to_token_ratio.mean():.2f}\n\n")
            f.write("Token Length Statistics:\n")
            f.write(str(token_stats))

        print(
            f"\nAnalysis complete! Saved {len(processed_df):,} samples for next token prediction training."
        )
        print("Files saved:")
        print("- next_token_prediction_dataset.csv (main dataset)")
        print("- next_token_prediction_dataset.pkl (tokenized data)")
        print("- token_distributions.png (distribution plots)")
        print("- token_correlations.png (correlation heatmap)")
        print("- dataset_analysis.txt (summary statistics)")

    else:
        print("No data was successfully processed. Check your filtering criteria.")

    client.close()


if __name__ == "__main__":
    main()
