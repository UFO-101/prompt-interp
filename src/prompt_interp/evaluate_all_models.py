"""Evaluate and compare four models: base, soft prompt, LoRA, and finetuned."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from prompt_interp.train_soft_prompt import SoftPromptModel
from peft import PeftModel


def load_all_models():
    """Load all four models for comparison."""
    print("Loading models...")
    print("=" * 70)

    # 1. Base model (no training)
    print("1. Loading base model (Qwen3-0.6B-Base)")
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 2. Base model + soft prompt
    print("2. Loading base model + soft prompt")
    soft_prompt_path = "models/soft-prompt/soft_prompt.pt"
    if Path(soft_prompt_path).exists():
        soft_prompt_data = torch.load(soft_prompt_path, weights_only=True)
        soft_prompt_model = SoftPromptModel(
            base_model,
            n_tokens=soft_prompt_data['n_tokens']
        )
        soft_prompt_model.soft_prompt.data = soft_prompt_data['soft_prompt'].to(
            device=base_model.device,
            dtype=base_model.dtype
        )
        soft_prompt_tokenizer = base_tokenizer
        has_soft_prompt = True
    else:
        print(f"   ⚠ Soft prompt not found at {soft_prompt_path}")
        soft_prompt_model = None
        soft_prompt_tokenizer = None
        has_soft_prompt = False

    # 3. Base model + LoRA adapter
    print("3. Loading base model + LoRA adapter")
    lora_path = "models/lora"
    if Path(lora_path).exists():
        # Need to load a fresh base model for LoRA
        lora_base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(lora_base, lora_path)
        lora_tokenizer = base_tokenizer
        has_lora = True
    else:
        print(f"   ⚠ LoRA adapter not found at {lora_path}")
        lora_model = None
        lora_tokenizer = None
        has_lora = False

    # 4. Finetuned model
    print("4. Loading finetuned model")
    finetuned_path = "models/qwen-integer-sequences"
    if Path(finetuned_path).exists():
        finetuned_tokenizer = AutoTokenizer.from_pretrained(
            finetuned_path,
            trust_remote_code=True
        )
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        has_finetuned = True
    else:
        print(f"   ⚠ Finetuned model not found at {finetuned_path}")
        finetuned_model = None
        finetuned_tokenizer = None
        has_finetuned = False

    print("=" * 70)
    print()

    return {
        'base': (base_model, base_tokenizer),
        'soft_prompt': (soft_prompt_model, soft_prompt_tokenizer) if has_soft_prompt else (None, None),
        'lora': (lora_model, lora_tokenizer) if has_lora else (None, None),
        'finetuned': (finetuned_model, finetuned_tokenizer) if has_finetuned else (None, None),
    }


def predict(model, tokenizer, input_seq: str, use_soft_prompt: bool = False) -> str:
    """Generate prediction for a given input sequence."""
    prompt = f"Input: {input_seq} Output: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else 'cpu')

    with torch.no_grad():
        if use_soft_prompt:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output.replace(prompt, "").strip()
    prediction = generated.split()[0] if generated else "???"

    return prediction


def evaluate_all_models():
    """Evaluate all four models on test sequences."""
    models = load_all_models()

    # Test cases
    test_cases = [
        # Simple cases
        ("1 2 3 4", 5),
        ("5 6 7 8", 9),
        ("10 11 12 13", 14),

        # Multi-digit
        ("32 33 34 35", 36),
        ("100 101 102 103", 104),
        ("255 256 257 258", 259),

        # Edge cases
        ("0 1 2 3", 4),
        ("99 100 101 102", 103),

        # Beyond training range
        ("600 601 602 603", 604),
        ("1000 1001 1002 1003", 1004),

        # Different patterns
        ("2 4 6 8", 10),
        ("5 10 15 20", 25),
    ]

    # Evaluate each model
    results = {
        'base': [],
        'soft_prompt': [],
        'lora': [],
        'finetuned': []
    }

    print("EVALUATION RESULTS")
    print("=" * 110)
    print(f"{'Input Sequence':<25} {'Expected':<10} {'Base':<15} {'+ Soft Prompt':<15} {'+ LoRA':<15} {'Finetuned':<15}")
    print("=" * 110)

    for input_seq, expected in test_cases:
        row = [input_seq, str(expected)]

        # Base model
        if models['base'][0] is not None:
            pred = predict(models['base'][0], models['base'][1], input_seq)
            try:
                is_correct = int(pred) == expected
                results['base'].append(is_correct)
                status = "✓" if is_correct else "✗"
            except:
                results['base'].append(False)
                status = "✗"
            row.append(f"{status} {pred}")
        else:
            row.append("N/A")

        # Soft prompt model
        if models['soft_prompt'][0] is not None:
            pred = predict(models['soft_prompt'][0], models['soft_prompt'][1], input_seq, use_soft_prompt=True)
            try:
                is_correct = int(pred) == expected
                results['soft_prompt'].append(is_correct)
                status = "✓" if is_correct else "✗"
            except:
                results['soft_prompt'].append(False)
                status = "✗"
            row.append(f"{status} {pred}")
        else:
            row.append("N/A")

        # LoRA model
        if models['lora'][0] is not None:
            pred = predict(models['lora'][0], models['lora'][1], input_seq)
            try:
                is_correct = int(pred) == expected
                results['lora'].append(is_correct)
                status = "✓" if is_correct else "✗"
            except:
                results['lora'].append(False)
                status = "✗"
            row.append(f"{status} {pred}")
        else:
            row.append("N/A")

        # Finetuned model
        if models['finetuned'][0] is not None:
            pred = predict(models['finetuned'][0], models['finetuned'][1], input_seq)
            try:
                is_correct = int(pred) == expected
                results['finetuned'].append(is_correct)
                status = "✓" if is_correct else "✗"
            except:
                results['finetuned'].append(False)
                status = "✗"
            row.append(f"{status} {pred}")
        else:
            row.append("N/A")

        print(f"{row[0]:<25} {row[1]:<10} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<15}")

    print("=" * 110)
    print()

    # Summary statistics
    print("SUMMARY")
    print("=" * 70)
    for model_name, correct_list in results.items():
        if correct_list:
            accuracy = sum(correct_list) / len(correct_list) * 100
            print(f"{model_name.replace('_', ' ').title():<20}: {sum(correct_list)}/{len(correct_list)} correct ({accuracy:.1f}%)")
        else:
            print(f"{model_name.replace('_', ' ').title():<20}: N/A")
    print("=" * 70)

    return results


if __name__ == "__main__":
    evaluate_all_models()
