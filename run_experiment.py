#!/usr/bin/env python3
"""Main script to run the integer sequence finetuning experiment."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_interp.create_dataset import generate_sequence_dataset
from prompt_interp.finetune import finetune_model
from prompt_interp.test_sequences import interactive_test, batch_test


def run_full_experiment():
    """Run the complete experiment: dataset creation, finetuning, and testing."""
    print("=" * 70)
    print("INTEGER SEQUENCE FINETUNING EXPERIMENT")
    print("=" * 70)

    # Step 1: Generate dataset
    print("\n[Step 1/3] Generating dataset...")
    generate_sequence_dataset(
        num_samples=10000,
        min_start=0,
        max_start=500,
        sequence_length=5,
        output_file="data/integer_sequences.jsonl"
    )

    # Step 2: Finetune model
    print("\n[Step 2/3] Finetuning model...")
    print("This may take a while depending on your GPU...")
    finetune_model(
        model_name="Qwen/Qwen3-0.6B-Base",
        data_file="data/integer_sequences.jsonl",
        output_dir="models/qwen-integer-sequences",
        num_epochs=3,
        batch_size=32,  # Increased batch size
        learning_rate=5e-5,
    )

    # Step 3: Test the model
    print("\n[Step 3/3] Testing the finetuned model...")
    test_sequences = [
        "32 33 34 35",
        "1 2 3 4",
        "10 11 12 13",
        "100 101 102 103",
        "255 256 257 258",
    ]
    batch_test("models/qwen-integer-sequences", test_sequences)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print("\nFinetuned model saved to: models/qwen-integer-sequences")
    print("\nTo test interactively, run:")
    print("  python run_experiment.py --mode test")


def main():
    parser = argparse.ArgumentParser(
        description="Run integer sequence finetuning experiment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "dataset", "finetune", "test", "test-batch"],
        default="full",
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/qwen-integer-sequences",
        help="Path to the finetuned model (for testing)"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_experiment()

    elif args.mode == "dataset":
        print("Generating dataset...")
        generate_sequence_dataset(
            num_samples=10000,
            min_start=0,
            max_start=500,
            sequence_length=5,
            output_file="data/integer_sequences.jsonl"
        )

    elif args.mode == "finetune":
        print("Finetuning model...")
        finetune_model(
            model_name="Qwen/Qwen3-0.6B-Base",
            data_file="data/integer_sequences.jsonl",
            output_dir="models/qwen-integer-sequences",
            num_epochs=3,
            batch_size=32,  # Increased batch size
            learning_rate=5e-5,
        )

    elif args.mode == "test":
        interactive_test(args.model_path)

    elif args.mode == "test-batch":
        test_sequences = [
            "32 33 34 35",
            "1 2 3 4",
            "10 11 12 13",
            "100 101 102 103",
            "255 256 257 258",
            "0 1 2 3",
            "50 51 52 53",
            "200 201 202 203",
        ]
        batch_test(args.model_path, test_sequences)


if __name__ == "__main__":
    main()
