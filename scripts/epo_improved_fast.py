#!/usr/bin/env python3
"""
Fast Improved EPO with ContextBench enhancements:
1. EPO-Assist: Periodically use LLM to generate variations of best candidates
2. EPO-Inpainting: Freeze high-activation tokens and resample the rest

Uses gradient-based token importance for faster inpainting.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Test sequences for integer sequence prediction
TEST_SEQUENCES = [
    ("1 2 3 4", 5),
    ("5 6 7 8", 9),
    ("10 11 12 13", 14),
    ("32 33 34 35", 36),
    ("100 101 102 103", 104),
    ("255 256 257 258", 259),
    ("600 601 602 603", 604),
    ("1000 1001 1002 1003", 1004),
    ("0 1 2 3", 4),
    ("999 1000 1001 1002", 1003),
    ("5000 5001 5002 5003", 5004),
    ("9999 10000 10001 10002", 10003),
]


class FastImprovedEPO:
    """EPO with Assist and fast Inpainting improvements."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        prefix_length: int = 12,
        population_size: int = 4,
        top_k: int = 256,
        num_children: int = 12,
        lambda_values: List[float] = [0.5, 1.0, 2.0],
        assist_interval: int = 30,  # More frequent assist
        inpaint_interval: int = 20,  # Less frequent inpainting
        inpaint_freeze_ratio: float = 0.3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prefix_length = prefix_length
        self.population_size = population_size
        self.top_k = top_k
        self.num_children = num_children
        self.lambda_values = lambda_values
        self.assist_interval = assist_interval
        self.inpaint_interval = inpaint_interval
        self.inpaint_freeze_ratio = inpaint_freeze_ratio

        self.embed_layer = model.get_input_embeddings()
        self.vocab_size = self.embed_layer.num_embeddings

        # Pre-tokenize test data
        self.expected_tokens = []
        for _, expected in TEST_SEQUENCES:
            token_id = tokenizer.encode(str(expected), add_special_tokens=False)[0]
            self.expected_tokens.append(token_id)
        self.expected_tokens = torch.tensor(self.expected_tokens, device=device)
        self.input_texts = [f"Input: {inp} Output:" for inp, _ in TEST_SEQUENCES]

    def compute_score(self, prefix_tokens: torch.Tensor) -> float:
        """Compute average log probability of correct answers."""
        total_log_prob = 0.0
        with torch.no_grad():
            for text, exp_token in zip(self.input_texts, self.expected_tokens):
                seq_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)
                full_tokens = torch.cat([prefix_tokens, seq_tokens])
                outputs = self.model(full_tokens.unsqueeze(0))
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[exp_token].item()
        return total_log_prob / len(TEST_SEQUENCES)

    def compute_cross_entropy(self, prefix_tokens: torch.Tensor) -> float:
        """Compute cross-entropy of the prefix."""
        with torch.no_grad():
            outputs = self.model(prefix_tokens.unsqueeze(0))
            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            target_tokens = prefix_tokens[1:]
            token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            return -token_log_probs.mean().item()

    def compute_gradients(self, prefix_tokens: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """Compute gradients for token selection."""
        tokens = prefix_tokens.detach().clone()
        one_hot = F.one_hot(tokens, num_classes=self.vocab_size).to(self.embed_layer.weight.dtype)
        one_hot.requires_grad_(True)
        prefix_embeddings = one_hot @ self.embed_layer.weight

        total_log_prob = 0.0
        for text, exp_token in zip(self.input_texts, self.expected_tokens):
            seq_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)
            seq_embeddings = self.embed_layer(seq_tokens)
            full_embeddings = torch.cat([prefix_embeddings, seq_embeddings], dim=0)
            outputs = self.model(inputs_embeds=full_embeddings.unsqueeze(0))
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            total_log_prob = total_log_prob + log_probs[exp_token]

        avg_log_prob = total_log_prob / len(TEST_SEQUENCES)

        prefix_outputs = self.model(inputs_embeds=prefix_embeddings.unsqueeze(0))
        prefix_logits = prefix_outputs.logits[0]
        log_probs_prefix = F.log_softmax(prefix_logits[:-1], dim=-1)
        target_tokens = tokens[1:]
        token_log_probs = log_probs_prefix.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        avg_xe = -token_log_probs.mean()

        objective = avg_log_prob - lambda_val * avg_xe
        objective.backward()

        return one_hot.grad.detach()

    def compute_token_importance_fast(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """Fast token importance using gradient magnitude."""
        tokens = prefix_tokens.detach().clone()
        one_hot = F.one_hot(tokens, num_classes=self.vocab_size).to(self.embed_layer.weight.dtype)
        one_hot.requires_grad_(True)
        prefix_embeddings = one_hot @ self.embed_layer.weight

        # Compute task score
        total_log_prob = 0.0
        for text, exp_token in zip(self.input_texts, self.expected_tokens):
            seq_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)
            seq_embeddings = self.embed_layer(seq_tokens)
            full_embeddings = torch.cat([prefix_embeddings, seq_embeddings], dim=0)
            outputs = self.model(inputs_embeds=full_embeddings.unsqueeze(0))
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            total_log_prob = total_log_prob + log_probs[exp_token]

        avg_log_prob = total_log_prob / len(TEST_SEQUENCES)
        avg_log_prob.backward()

        # Token importance = gradient magnitude at the selected token
        grad = one_hot.grad.detach()
        # Get gradient for the actual tokens selected (diagonal of position x vocab)
        importance = torch.zeros(len(prefix_tokens), device=self.device)
        for i, t in enumerate(tokens):
            importance[i] = grad[i, t].abs()

        return importance

    def llm_assist(self, population: List[Tuple[torch.Tensor, float, float]]) -> List[torch.Tensor]:
        """Use the model to generate variations of best candidates."""
        variations = []
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
        top_candidates = sorted_pop[:2]

        examples = []
        for tokens, score, xe in top_candidates:
            text = self.tokenizer.decode(tokens)
            examples.append(f'"{text}"')

        prompt = f"""These prefixes help predict the next number in sequences like "1 2 3 4" -> 5:
{chr(10).join(examples)}

Write 2 similar prefixes about predicting next numbers:
1."""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                lines = [l.strip() for l in generated.replace('"', '').split('\n') if l.strip()]
                for line in lines[:2]:
                    line = line.lstrip('0123456789. -')
                    if len(line) > 5:
                        tokens = self.tokenizer.encode(line[:60], add_special_tokens=False)
                        tokens = torch.tensor(tokens[:self.prefix_length], device=self.device)
                        if len(tokens) < self.prefix_length:
                            padding = torch.randint(0, self.vocab_size, (self.prefix_length - len(tokens),), device=self.device)
                            tokens = torch.cat([tokens, padding])
                        variations.append(tokens)
        except Exception as e:
            print(f"  LLM assist error: {e}")

        return variations

    def inpaint_fast(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """Fast inpainting: freeze important tokens, resample others from model distribution."""
        importance = self.compute_token_importance_fast(prefix_tokens)

        num_freeze = max(1, int(self.prefix_length * self.inpaint_freeze_ratio))
        _, top_indices = torch.topk(importance, num_freeze)

        freeze_mask = torch.zeros(self.prefix_length, dtype=torch.bool, device=self.device)
        freeze_mask[top_indices] = True

        new_tokens = prefix_tokens.clone()
        with torch.no_grad():
            for i in range(self.prefix_length):
                if not freeze_mask[i]:
                    if i == 0:
                        # Random token for first position
                        new_tokens[i] = torch.randint(0, self.vocab_size, (1,), device=self.device).item()
                    else:
                        # Sample from model distribution
                        context = new_tokens[:i]
                        outputs = self.model(context.unsqueeze(0))
                        logits = outputs.logits[0, -1, :]
                        probs = F.softmax(logits / 0.9, dim=-1)
                        new_tokens[i] = torch.multinomial(probs, 1).item()

        return new_tokens

    def run(self, num_iterations: int = 60, verbose: bool = True) -> dict:
        """Run improved EPO optimization."""
        results = {}

        for lambda_val in self.lambda_values:
            if verbose:
                print(f"\n{'='*50}")
                print(f"EPO-Improved λ={lambda_val}")
                print(f"{'='*50}")

            # Initialize population
            population = []
            for _ in range(self.population_size):
                tokens = torch.randint(0, self.vocab_size, (self.prefix_length,), device=self.device)
                score = self.compute_score(tokens)
                xe = self.compute_cross_entropy(tokens)
                population.append((tokens, score, xe))

            best_combined = float('-inf')
            best_result = None

            for iteration in range(num_iterations):
                new_population = []

                for tokens, _, _ in population:
                    gradients = self.compute_gradients(tokens, lambda_val)
                    _, top_k_indices = torch.topk(gradients, k=min(self.top_k, self.vocab_size), dim=1)

                    candidates = [(tokens.clone(), self.compute_score(tokens), self.compute_cross_entropy(tokens))]

                    for _ in range(self.num_children // self.population_size):
                        child_tokens = tokens.clone()
                        pos = np.random.randint(0, self.prefix_length)
                        k_idx = np.random.randint(0, min(self.top_k, self.vocab_size))
                        child_tokens[pos] = top_k_indices[pos, k_idx]
                        score = self.compute_score(child_tokens)
                        xe = self.compute_cross_entropy(child_tokens)
                        candidates.append((child_tokens, score, xe))

                    best_candidate = max(candidates, key=lambda x: x[1] - lambda_val * x[2])
                    new_population.append(best_candidate)

                population = new_population

                # Inpainting
                if (iteration + 1) % self.inpaint_interval == 0 and iteration < num_iterations - 5:
                    if verbose:
                        print(f"  Iter {iteration+1}: Inpainting...")
                    inpainted = []
                    for tokens, score, xe in population:
                        new_tokens = self.inpaint_fast(tokens)
                        new_score = self.compute_score(new_tokens)
                        new_xe = self.compute_cross_entropy(new_tokens)
                        if new_score - lambda_val * new_xe >= score - lambda_val * xe - 0.2:
                            inpainted.append((new_tokens, new_score, new_xe))
                        else:
                            inpainted.append((tokens, score, xe))
                    population = inpainted

                # LLM Assist
                if (iteration + 1) % self.assist_interval == 0 and iteration < num_iterations - 10:
                    if verbose:
                        print(f"  Iter {iteration+1}: LLM Assist...")
                    variations = self.llm_assist(population)
                    for var_tokens in variations:
                        score = self.compute_score(var_tokens)
                        xe = self.compute_cross_entropy(var_tokens)
                        worst_idx = min(range(len(population)),
                                       key=lambda i: population[i][1] - lambda_val * population[i][2])
                        worst_combined = population[worst_idx][1] - lambda_val * population[worst_idx][2]
                        if score - lambda_val * xe > worst_combined:
                            population[worst_idx] = (var_tokens, score, xe)

                for tokens, score, xe in population:
                    combined = score - lambda_val * xe
                    if combined > best_combined:
                        best_combined = combined
                        best_result = (tokens.clone(), score, xe)

                if verbose and (iteration + 1) % 15 == 0:
                    best_tokens, best_score, best_xe = best_result
                    print(f"  Iter {iteration+1}: score={best_score:.3f}, XE={best_xe:.2f}")

            best_tokens, best_score, best_xe = best_result
            results[lambda_val] = {
                "tokens": best_tokens,
                "text": self.tokenizer.decode(best_tokens),
                "score": best_score,
                "cross_entropy": best_xe,
            }

            if verbose:
                print(f"\nFinal: score={best_score:.3f}, XE={best_xe:.2f}")
                print(f"Prefix: '{results[lambda_val]['text']}'")

        return results


def evaluate_model(model, tokenizer, device, prefix: str) -> Tuple[float, List[dict]]:
    """Evaluate a model with a given prefix."""
    model.eval()
    results = []

    with torch.no_grad():
        for input_seq, expected in TEST_SEQUENCES:
            prompt = f"{prefix}Input: {input_seq} Output:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            try:
                pred_num = int(generated.strip().split()[0])
                is_correct = pred_num == expected
            except:
                pred_num = generated.strip()[:20]
                is_correct = False

            results.append({
                "input": input_seq,
                "expected": expected,
                "predicted": pred_num,
                "correct": is_correct,
            })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")

    # Run improved EPO
    print("\n" + "="*60)
    print("RUNNING FAST IMPROVED EPO")
    print("="*60)

    epo = FastImprovedEPO(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefix_length=12,
        population_size=4,
        lambda_values=[0.5, 1.0, 2.0],
        assist_interval=20,
        inpaint_interval=15,
    )

    results = epo.run(num_iterations=60, verbose=True)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    all_results = {"improved_epo": {}, "handcrafted": {}}

    for lambda_val, data in results.items():
        prefix = data["text"]
        acc, samples = evaluate_model(model, tokenizer, device, prefix)
        print(f"\nEPO-Improved λ={lambda_val}: {acc*100:.1f}%")
        print(f"  Prefix: '{prefix}'")
        all_results["improved_epo"][f"EPO-Improved (λ={lambda_val})"] = {
            "accuracy": acc, "prefix": prefix, "samples": samples,
            "score": data["score"], "cross_entropy": data["cross_entropy"]
        }

    # Handcrafted comparison
    print("\n" + "-"*40)
    print("Handcrafted comparison:")
    handcrafted = {
        "No prefix": "",
        "Simple instruction": "What is the next number? ",
        "1-shot": "Example: Input: 3 4 5 6 Output: 7\n",
        "2-shot": "Example: Input: 3 4 5 6 Output: 7\nExample: Input: 20 21 22 23 Output: 24\n",
        "3-shot": "Example: Input: 3 4 5 6 Output: 7\nExample: Input: 20 21 22 23 Output: 24\nExample: Input: 50 51 52 53 Output: 54\n",
    }

    for name, prefix in handcrafted.items():
        acc, samples = evaluate_model(model, tokenizer, device, prefix)
        print(f"  {name}: {acc*100:.1f}%")
        all_results["handcrafted"][name] = {"accuracy": acc, "prefix": prefix, "samples": samples}

    # Create chart
    print("\nCreating chart...")
    fig, ax = plt.subplots(figsize=(12, 8))

    approaches, accuracies, colors = [], [], []

    for name, data in all_results["handcrafted"].items():
        approaches.append(name)
        accuracies.append(data["accuracy"] * 100)
        acc = data["accuracy"]
        colors.append('#2ecc71' if acc >= 0.9 else '#f39c12' if acc >= 0.5 else '#e74c3c')

    for name, data in all_results["improved_epo"].items():
        approaches.append(name)
        accuracies.append(data["accuracy"] * 100)
        colors.append('#9b59b6')

    bars = ax.barh(range(len(approaches)), accuracies, color=colors)
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Integer Sequence Prediction: EPO-Improved Results', fontsize=14)
    ax.set_xlim(0, 105)

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)

    legend_elements = [
        Patch(facecolor='#2ecc71', label='Handcrafted (high)'),
        Patch(facecolor='#f39c12', label='Handcrafted (medium)'),
        Patch(facecolor='#e74c3c', label='Handcrafted (low)'),
        Patch(facecolor='#9b59b6', label='EPO-Improved'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('improved_epo_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: improved_epo_comparison.png")

    with open("improved_epo_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Saved: improved_epo_results.json")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Approach':<35} {'Accuracy':>10}")
    print("-" * 45)
    for name, data in all_results["handcrafted"].items():
        print(f"{name:<35} {data['accuracy']*100:>9.1f}%")
    print("-" * 45)
    for name, data in all_results["improved_epo"].items():
        print(f"{name:<35} {data['accuracy']*100:>9.1f}%")


if __name__ == "__main__":
    main()
