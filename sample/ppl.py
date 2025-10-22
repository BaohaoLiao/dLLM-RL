"""
Perplexity evaluation script for block diffusion models.

Usage:
    python ppl.py --model_name_or_path /path/to/model --dataset_path test.json --block_length 4
"""

import json
import argparse
from typing import List, Dict
import numpy as np
import datasets
from jetengine_ext import LLM


def load_dataset(
    dataset_path: str, tokenizer, max_samples: int = None, max_length: int = 512
):
    """
    Load evaluation dataset and tokenize.

    Args:
        dataset_path: path to a json file
        tokenizer: Tokenizer instance
        max_samples: Maximum number of samples to evaluate (None = all)
        max_length: Maximum sequence length

    Returns:
        List of tokenized sequences
    """
    sequences = []

    dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        text = example["text"].strip()
        if len(text) > 0:  # Skip empty lines
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 0:
                # Truncate if too long
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                sequences.append(tokens)

    print(f"Loaded {len(sequences)} sequences from {dataset_path}")
    return sequences


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from results."""
    if not results:
        return {}

    ppls = [r["perplexity"] for r in results]
    nlls = [r["nll"] for r in results]
    num_tokens = [len(r["tokens"]) for r in results]

    # Compute corpus-level perplexity (weighted by sequence length)
    total_tokens = sum(num_tokens)
    total_nll = sum(nll * n for nll, n in zip(nlls, num_tokens))
    corpus_nll = total_nll / total_tokens
    corpus_ppl = np.exp(corpus_nll)

    stats = {
        "corpus_perplexity": corpus_ppl,
        "corpus_nll": corpus_nll,
        "mean_perplexity": np.mean(ppls),
        "median_perplexity": np.median(ppls),
        "std_perplexity": np.std(ppls),
        "min_perplexity": np.min(ppls),
        "max_perplexity": np.max(ppls),
        "total_sequences": len(results),
        "total_tokens": total_tokens,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity for block diffusion models"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    # Evaluation arguments
    parser.add_argument(
        "--block_length", type=int, required=True, help="Block length for evaluation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument('--output_file', type=str, default='ppl_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    print("=" * 60)
    print("Block Diffusion Model Perplexity Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Block length: {args.block_length}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")

    mask_token_id = -1
    if "trado" in args.model_name_or_path.lower():
        mask_token_id = 151669,
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        mask_token_id=mask_token_id,
        block_length=args.block_length,
    )
    print(f"Model loaded. Mask token ID: {llm.config.mask_token_id}")

    # Load dataset
    print("\nLoading dataset...")
    sentences = load_dataset(
        args.dataset_path,
        llm.tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )

    if sentences is None or len(sentences) == 0:
        print("Failed to load dataset or dataset is empty")
        return

    print(f"Loaded {len(sentences)} sequences")
    print(f"Average sequence length: {np.mean([len(s) for s in sentences]):.1f}")

    # Evaluate perplexity in batches
    print("\nEvaluating perplexity...")
    all_results = []

    for i in range(0, len(sentences), args.batch_size):
        batch = sentences[i : i + args.batch_size]
        print(
            f"\nProcessing batch {i // args.batch_size + 1}/{(len(sentences) + args.batch_size - 1) // args.batch_size}"
        )

        results = llm.evaluate_perplexity(
            batch, block_length=args.block_length, use_tqdm=True
        )
        all_results.extend(results)

    if len(all_results) == 0:
        print("✗ No results generated")
        return

    print(f"\n✓ Successfully evaluated {len(all_results)} sequences")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Corpus Perplexity: {stats['corpus_perplexity']:.4f}")
    print(f"Corpus NLL: {stats['corpus_nll']:.4f}")
    print(f"Mean Perplexity: {stats['mean_perplexity']:.4f}")
    print(f"Median Perplexity: {stats['median_perplexity']:.4f}")
    print(f"Std Perplexity: {stats['std_perplexity']:.4f}")
    print(f"Min Perplexity: {stats['min_perplexity']:.4f}")
    print(f"Max Perplexity: {stats['max_perplexity']:.4f}")
    print(f"Total Sequences: {stats['total_sequences']}")
    print(f"Total Tokens: {stats['total_tokens']}")
    print("=" * 60)

    # Save results
    output_data = {
        'args': vars(args),
        'statistics': stats,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
