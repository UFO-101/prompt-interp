"""Test the finetuned model on integer sequences."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def load_model(model_path: str):
    """Load the finetuned model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model


def test_sequence(
    input_sequence: str,
    tokenizer,
    model,
    max_new_tokens: int = 5,
    temperature: float = 0.1
):
    """Test the model on a given input sequence.

    Args:
        input_sequence: Space-separated integers (e.g., "32 33 34 35")
        tokenizer: The tokenizer
        model: The model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
    """
    # Format the prompt
    prompt = f"Input: {input_sequence} Output:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the output part
    if "Output:" in generated_text:
        output_part = generated_text.split("Output:")[-1].strip()
    else:
        output_part = generated_text[len(prompt):].strip()

    return output_part


def interactive_test(model_path: str):
    """Run interactive testing."""
    tokenizer, model = load_model(model_path)

    print("\n" + "=" * 60)
    print("Integer Sequence Completion - Interactive Testing")
    print("=" * 60)
    print("\nEnter a sequence of integers (space-separated)")
    print("Example: 32 33 34 35")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            user_input = input("Enter sequence: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not user_input:
                continue

            # Test the sequence
            result = test_sequence(user_input, tokenizer, model)
            print(f"  Model output: {result}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def batch_test(model_path: str, test_sequences: list):
    """Run batch testing on predefined sequences."""
    tokenizer, model = load_model(model_path)

    print("\n" + "=" * 60)
    print("Batch Testing Results")
    print("=" * 60)

    for seq in test_sequences:
        result = test_sequence(seq, tokenizer, model)
        # Try to extract just the number
        predicted = result.split()[0] if result else "N/A"
        print(f"Input: {seq:20s} -> Predicted: {predicted}")


def main():
    parser = argparse.ArgumentParser(description="Test finetuned integer sequence model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/qwen-integer-sequences",
        help="Path to the finetuned model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Testing mode"
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_test(args.model_path)
    else:
        # Predefined test sequences
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
