#!/usr/bin/env python3
"""
Compare GBDA vs EPO (Fluent Dreaming) on prompt optimization tasks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import numpy as np

from prompt_interp.gbda import GBDAConfig, gumbel_softmax_sample
from prompt_interp.epo import EPOConfig, PopulationMember


@dataclass
class TaskConfig:
    name: str
    description: str
    seed_prompt: str
    train_examples: list[tuple[str, str]]
    test_examples: list[tuple[str, str]]


def create_two_tasks() -> list[TaskConfig]:
    """Create just 2 tasks for quick comparison."""

    # Antonyms task
    antonyms_train = [
        ("hot -> ", "cold"), ("big -> ", "small"), ("fast -> ", "slow"),
        ("up -> ", "down"), ("happy -> ", "sad"), ("light -> ", "dark"),
        ("old -> ", "young"), ("tall -> ", "short"), ("hard -> ", "soft"),
        ("good -> ", "bad"), ("clean -> ", "dirty"), ("open -> ", "closed"),
        ("new -> ", "old"), ("high -> ", "low"), ("right -> ", "wrong"),
    ]
    antonyms_test = [
        ("wet -> ", "dry"), ("loud -> ", "quiet"), ("rich -> ", "poor"),
        ("full -> ", "empty"), ("strong -> ", "weak"), ("thick -> ", "thin"),
        ("deep -> ", "shallow"), ("wide -> ", "narrow"), ("rough -> ", "smooth"),
        ("bright -> ", "dim"),
    ]

    # Plurals task
    plurals_train = [
        ("cat -> ", "cats"), ("dog -> ", "dogs"), ("child -> ", "children"),
        ("mouse -> ", "mice"), ("foot -> ", "feet"), ("tooth -> ", "teeth"),
        ("man -> ", "men"), ("woman -> ", "women"), ("person -> ", "people"),
        ("goose -> ", "geese"), ("fish -> ", "fish"), ("deer -> ", "deer"),
        ("sheep -> ", "sheep"), ("leaf -> ", "leaves"), ("knife -> ", "knives"),
    ]
    plurals_test = [
        ("apple -> ", "apples"), ("river -> ", "rivers"), ("window -> ", "windows"),
        ("friend -> ", "friends"), ("ox -> ", "oxen"), ("louse -> ", "lice"),
        ("crisis -> ", "crises"), ("thesis -> ", "theses"), ("cactus -> ", "cacti"),
        ("fungus -> ", "fungi"),
    ]

    return [
        TaskConfig(
            name="antonyms",
            description="Word → Opposite",
            seed_prompt="Opposites: hot -> cold, big -> small. Now:",
            train_examples=antonyms_train,
            test_examples=antonyms_test,
        ),
        TaskConfig(
            name="plurals",
            description="Singular → Plural",
            seed_prompt="Plurals: cat -> cats, child -> children. Now:",
            train_examples=plurals_train,
            test_examples=plurals_test,
        ),
    ]


class MethodComparison:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", device="cuda"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.embed_layer = self.model.get_input_embeddings()
        self.vocab_size = self.embed_layer.num_embeddings

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(ids.cpu().tolist())

    def evaluate_prompt(self, prompt_tokens, examples):
        """Evaluate a discrete prompt."""
        n_correct = 0
        results = []

        self.model.eval()
        with torch.no_grad():
            for input_text, target_text in examples:
                input_ids = self.encode(input_text)
                full_ids = torch.cat([prompt_tokens, input_ids])

                # Generate
                outputs = self.model(full_ids.unsqueeze(0))
                next_token = outputs.logits[0, -1, :].argmax().item()
                pred = self.tokenizer.decode([next_token]).strip()

                correct = pred.lower().startswith(target_text.lower()[:3])
                if correct:
                    n_correct += 1
                results.append((input_text.replace(" -> ", ""), pred, target_text, correct))

        return n_correct / len(examples), results

    def run_gbda(self, task, num_iterations=50, prompt_length=12):
        """Run GBDA optimization."""
        print(f"\n  [GBDA] Running {num_iterations} iterations...")

        # Initialize theta from seed
        seed_tokens = self.encode(task.seed_prompt)[:prompt_length]
        if len(seed_tokens) < prompt_length:
            pad = self.encode(".")[0].item()
            padding = torch.full((prompt_length - len(seed_tokens),), pad, device=self.device, dtype=torch.long)
            seed_tokens = torch.cat([seed_tokens, padding])

        theta = torch.zeros(prompt_length, self.vocab_size, device=self.device)
        for i, tok_id in enumerate(seed_tokens):
            theta[i, tok_id] = 12.0
        theta = torch.nn.Parameter(theta)

        optimizer = torch.optim.Adam([theta], lr=0.3)
        best_acc, best_tokens, best_text = 0, seed_tokens.clone(), self.decode(seed_tokens)

        for step in range(num_iterations):
            optimizer.zero_grad()
            temp = 1.0 * (1 - step/num_iterations) + 0.1 * (step/num_iterations)

            # Sample and compute loss
            logits = theta.unsqueeze(0).expand(8, -1, -1)
            soft_tokens = gumbel_softmax_sample(logits, temp)
            soft_embeds = torch.matmul(soft_tokens.to(self.embed_layer.weight.dtype), self.embed_layer.weight)

            # Task loss
            loss = 0
            for inp, tgt in task.train_examples[:10]:
                inp_ids = self.encode(inp)
                inp_embeds = self.embed_layer(inp_ids).unsqueeze(0).expand(8, -1, -1)
                full = torch.cat([soft_embeds, inp_embeds], dim=1)
                out = self.model(inputs_embeds=full).logits[:, -1, :]
                tgt_id = self.encode(tgt)[0].item()
                loss += F.cross_entropy(out, torch.full((8,), tgt_id, device=self.device))

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                with torch.no_grad():
                    curr_tokens = theta.argmax(dim=-1)
                    acc, _ = self.evaluate_prompt(curr_tokens, task.train_examples)
                    if acc > best_acc:
                        best_acc = acc
                        best_tokens = curr_tokens.clone()
                        best_text = self.decode(curr_tokens)

        train_acc, _ = self.evaluate_prompt(best_tokens, task.train_examples)
        test_acc, test_results = self.evaluate_prompt(best_tokens, task.test_examples)

        return {"method": "GBDA", "prompt": best_text, "train_acc": train_acc,
                "test_acc": test_acc, "results": test_results}

    def run_epo(self, task, num_iterations=50, prompt_length=12, population_size=4):
        """Run EPO (Fluent Dreaming) optimization."""
        print(f"\n  [EPO] Running {num_iterations} iterations...")

        # Initialize population from seed
        seed_tokens = self.encode(task.seed_prompt)[:prompt_length]
        if len(seed_tokens) < prompt_length:
            pad = self.encode(".")[0].item()
            padding = torch.full((prompt_length - len(seed_tokens),), pad, device=self.device, dtype=torch.long)
            seed_tokens = torch.cat([seed_tokens, padding])

        # Lambda values for Pareto frontier
        lambdas = [0.5, 1.0, 2.0, 4.0][:population_size]

        # Initialize population
        population = []
        for lam in lambdas:
            population.append({
                "tokens": seed_tokens.clone(),
                "lambda": lam,
                "score": 0.0,
                "entropy": 0.0
            })

        best_acc, best_tokens, best_text = 0, seed_tokens.clone(), self.decode(seed_tokens)

        for step in range(num_iterations):
            # For each member, generate children via gradient-guided mutation
            all_candidates = []

            for member in population:
                tokens = member["tokens"].clone()

                # Compute gradients
                one_hot = F.one_hot(tokens, num_classes=self.vocab_size).float()
                one_hot.requires_grad_(True)
                embeds = one_hot.to(self.embed_layer.weight.dtype) @ self.embed_layer.weight

                # Task score
                score = 0
                for inp, tgt in task.train_examples[:8]:
                    inp_ids = self.encode(inp)
                    inp_embeds = self.embed_layer(inp_ids)
                    full = torch.cat([embeds, inp_embeds], dim=0).unsqueeze(0)
                    out = self.model(inputs_embeds=full).logits[0, -1, :]
                    tgt_id = self.encode(tgt)[0].item()
                    score += out[tgt_id]

                # Cross-entropy (fluency)
                out_all = self.model(inputs_embeds=embeds.unsqueeze(0)).logits[0]
                log_probs = F.log_softmax(out_all[:-1], dim=-1)
                ce = -log_probs.gather(1, tokens[1:].unsqueeze(1)).mean()

                obj = score - member["lambda"] * ce
                obj.backward()
                grads = one_hot.grad

                # Generate children by mutating top-k positions
                for _ in range(8):
                    child = tokens.clone()
                    pos = np.random.randint(0, prompt_length)
                    top_k = grads[pos].topk(256).indices
                    new_tok = top_k[np.random.randint(0, 256)]
                    child[pos] = new_tok
                    all_candidates.append({"tokens": child, "lambda": member["lambda"]})

            # Evaluate candidates and select best per lambda
            for cand in all_candidates:
                acc, _ = self.evaluate_prompt(cand["tokens"], task.train_examples[:5])
                cand["score"] = acc

            # Select best for each lambda
            new_pop = []
            for lam in lambdas:
                best_cand = max([c for c in all_candidates],
                               key=lambda c: c["score"] - abs(c["lambda"] - lam) * 0.1)
                new_pop.append({"tokens": best_cand["tokens"].clone(), "lambda": lam,
                               "score": best_cand["score"], "entropy": 0})
            population = new_pop

            # Track best overall
            for m in population:
                if m["score"] > best_acc:
                    best_acc = m["score"]
                    best_tokens = m["tokens"].clone()
                    best_text = self.decode(m["tokens"])

            if step % 10 == 0:
                print(f"    Step {step}: best={best_acc:.0%}")

        train_acc, _ = self.evaluate_prompt(best_tokens, task.train_examples)
        test_acc, test_results = self.evaluate_prompt(best_tokens, task.test_examples)

        return {"method": "EPO", "prompt": best_text, "train_acc": train_acc,
                "test_acc": test_acc, "results": test_results}


def main():
    print("=" * 70)
    print("GBDA vs EPO (Fluent Dreaming) Comparison")
    print("=" * 70)

    comp = MethodComparison()
    tasks = create_two_tasks()

    all_results = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.name} - {task.description}")
        print(f"Train: {len(task.train_examples)}, Test: {len(task.test_examples)}")
        print(f"Seed: '{task.seed_prompt}'")
        print("="*60)

        # Baseline
        seed_tokens = comp.encode(task.seed_prompt)[:12]
        if len(seed_tokens) < 12:
            pad = comp.encode(".")[0].item()
            padding = torch.full((12 - len(seed_tokens),), pad, device=comp.device, dtype=torch.long)
            seed_tokens = torch.cat([seed_tokens, padding])

        seed_train, _ = comp.evaluate_prompt(seed_tokens, task.train_examples)
        seed_test, _ = comp.evaluate_prompt(seed_tokens, task.test_examples)
        print(f"\n  Baseline: train={seed_train:.0%}, test={seed_test:.0%}")

        # Run both methods
        gbda_result = comp.run_gbda(task, num_iterations=50)
        epo_result = comp.run_epo(task, num_iterations=50)

        all_results.append({
            "task": task.name,
            "description": task.description,
            "seed_train": seed_train,
            "seed_test": seed_test,
            "gbda": gbda_result,
            "epo": epo_result
        })

    # Print comparison chart
    print("\n")
    print("=" * 80)
    print("COMPARISON CHART: GBDA vs EPO (Fluent Dreaming)")
    print("=" * 80)

    for r in all_results:
        print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ Task: {r['task'].upper()} ({r['description']})".ljust(78) + "│")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ {'Method':<12} {'Train Acc':<12} {'Test Acc':<12} {'Delta':<10}              │")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ {'Baseline':<12} {r['seed_train']:>8.0%}     {r['seed_test']:>8.0%}     {'--':<10}              │")

        gbda_delta = r['gbda']['test_acc'] - r['seed_test']
        print(f"│ {'GBDA':<12} {r['gbda']['train_acc']:>8.0%}     {r['gbda']['test_acc']:>8.0%}     {gbda_delta:>+7.0%}              │")

        epo_delta = r['epo']['test_acc'] - r['seed_test']
        print(f"│ {'EPO':<12} {r['epo']['train_acc']:>8.0%}     {r['epo']['test_acc']:>8.0%}     {epo_delta:>+7.0%}              │")

        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ Prompts Found:".ljust(78) + "│")
        print(f"│   GBDA: {r['gbda']['prompt'][:65]}".ljust(78) + "│")
        print(f"│   EPO:  {r['epo']['prompt'][:65]}".ljust(78) + "│")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ Example Predictions (GBDA):".ljust(78) + "│")
        for inp, pred, tgt, cor in r['gbda']['results'][:3]:
            mark = "✓" if cor else "✗"
            print(f"│   {mark} {inp} → '{pred}' (target: {tgt})".ljust(78) + "│")
        print(f"│ Example Predictions (EPO):".ljust(78) + "│")
        for inp, pred, tgt, cor in r['epo']['results'][:3]:
            mark = "✓" if cor else "✗"
            print(f"│   {mark} {inp} → '{pred}' (target: {tgt})".ljust(78) + "│")
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Task':<12} {'Baseline':<10} {'GBDA':<10} {'EPO':<10} {'Winner':<10}")
    print("-" * 52)
    for r in all_results:
        winner = "GBDA" if r['gbda']['test_acc'] > r['epo']['test_acc'] else "EPO" if r['epo']['test_acc'] > r['gbda']['test_acc'] else "Tie"
        print(f"{r['task']:<12} {r['seed_test']:>8.0%}   {r['gbda']['test_acc']:>8.0%}   {r['epo']['test_acc']:>8.0%}   {winner:<10}")


if __name__ == "__main__":
    main()
