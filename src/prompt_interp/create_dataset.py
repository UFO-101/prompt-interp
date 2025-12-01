"""Generate dataset of ascending integer sequences."""
import random
import json
from pathlib import Path


def generate_sequence_dataset(
    num_samples: int = 10000,
    min_start: int = 0,
    max_start: int = 1000,
    sequence_length: int = 5,
    output_file: str = "integer_sequences.jsonl"
):
    """Generate a dataset of ascending integer sequences.

    Args:
        num_samples: Number of sequences to generate
        min_start: Minimum starting integer
        max_start: Maximum starting integer (ensuring sequence doesn't overflow)
        sequence_length: Length of each sequence
        output_file: Output file path
    """
    sequences = []

    for _ in range(num_samples):
        # Generate a random starting point
        start = random.randint(min_start, max_start)

        # Create ascending sequence
        sequence = [start + i for i in range(sequence_length)]

        # Format as string (space-separated)
        sequence_str = " ".join(map(str, sequence))

        # For training, we'll format as:
        # Input: first 4 numbers
        # Output: last number
        input_seq = " ".join(map(str, sequence[:-1]))
        target = str(sequence[-1])

        sequences.append({
            "input": input_seq,
            "target": target,
            "full_sequence": sequence_str
        })

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")

    print(f"Generated {num_samples} sequences")
    print(f"Saved to {output_file}")
    print(f"Example sequences:")
    for i in range(min(5, len(sequences))):
        print(f"  Input: {sequences[i]['input']} -> Target: {sequences[i]['target']}")


if __name__ == "__main__":
    generate_sequence_dataset(
        num_samples=10000,
        min_start=0,
        max_start=500,
        sequence_length=5,
        output_file="data/integer_sequences.jsonl"
    )
