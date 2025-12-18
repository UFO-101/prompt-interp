#%%
#!/usr/bin/env python3
"""
Train prompts using GBDA (Gradient-based Distributional Attack).

Uses Gumbel-softmax to optimize a distribution over token sequences,
enabling true gradient-based optimization over discrete prompts.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from dataclasses import dataclass

from prompt_interp.gbda import GBDAConfig, GBDAPromptOptimizer, gumbel_softmax_sample


@dataclass
class TaskConfig:
    """Configuration for a task."""
    name: str
    description: str
    seed_prompt: str
    train_examples: list[tuple[str, str]]
    test_examples: list[tuple[str, str]]


def create_tasks() -> list[TaskConfig]:
    """Create the standard evaluation tasks."""

    # Antonyms task
    antonyms_train = [
        ("hot -> ", "cold"), ("big -> ", "small"), ("fast -> ", "slow"),
        ("up -> ", "down"), ("happy -> ", "sad"), ("light -> ", "dark"),
        ("old -> ", "young"), ("tall -> ", "short"), ("hard -> ", "soft"),
        ("good -> ", "bad"), ("clean -> ", "dirty"), ("open -> ", "closed"),
        ("new -> ", "old"), ("high -> ", "low"), ("right -> ", "wrong"),
        ("early -> ", "late"), ("black -> ", "white"), ("love -> ", "hate"),
        ("true -> ", "false"), ("inside -> ", "outside"),
        ("day -> ", "night"), ("yes -> ", "no"), ("push -> ", "pull"),
        ("start -> ", "stop"), ("alive -> ", "dead"), ("win -> ", "lose"),
        ("question -> ", "answer"), ("before -> ", "after"),
        ("top -> ", "bottom"), ("front -> ", "back"),
    ]
    antonyms_test = [
        ("wet -> ", "dry"), ("loud -> ", "quiet"), ("rich -> ", "poor"),
        ("full -> ", "empty"), ("strong -> ", "weak"), ("thick -> ", "thin"),
        ("deep -> ", "shallow"), ("wide -> ", "narrow"), ("rough -> ", "smooth"),
        ("bright -> ", "dim"), ("cheap -> ", "expensive"), ("sharp -> ", "dull"),
        ("warm -> ", "cool"), ("safe -> ", "dangerous"), ("wild -> ", "tame"),
        ("sweet -> ", "sour"), ("heavy -> ", "light"), ("sharp -> ", "blunt"),
        ("awake -> ", "asleep"), ("laugh -> ", "cry"),
    ]

    # Plurals task
    plurals_train = [
        ("cat -> ", "cats"), ("dog -> ", "dogs"), ("child -> ", "children"),
        ("mouse -> ", "mice"), ("foot -> ", "feet"), ("tooth -> ", "teeth"),
        ("man -> ", "men"), ("woman -> ", "women"), ("person -> ", "people"),
        ("goose -> ", "geese"), ("fish -> ", "fish"), ("deer -> ", "deer"),
        ("sheep -> ", "sheep"), ("leaf -> ", "leaves"), ("knife -> ", "knives"),
        ("wife -> ", "wives"), ("life -> ", "lives"), ("wolf -> ", "wolves"),
        ("calf -> ", "calves"), ("half -> ", "halves"),
        ("box -> ", "boxes"), ("bus -> ", "buses"), ("class -> ", "classes"),
        ("watch -> ", "watches"), ("dish -> ", "dishes"), ("brush -> ", "brushes"),
        ("baby -> ", "babies"), ("city -> ", "cities"), ("party -> ", "parties"),
        ("story -> ", "stories"),
    ]
    plurals_test = [
        ("apple -> ", "apples"), ("river -> ", "rivers"), ("window -> ", "windows"),
        ("friend -> ", "friends"), ("ox -> ", "oxen"), ("louse -> ", "lice"),
        ("crisis -> ", "crises"), ("thesis -> ", "theses"), ("cactus -> ", "cacti"),
        ("fungus -> ", "fungi"), ("hero -> ", "heroes"), ("potato -> ", "potatoes"),
        ("tomato -> ", "tomatoes"), ("echo -> ", "echoes"), ("veto -> ", "vetoes"),
        ("analysis -> ", "analyses"), ("basis -> ", "bases"), ("oasis -> ", "oases"),
        ("hypothesis -> ", "hypotheses"), ("axis -> ", "axes"),
    ]

    # Past tense task
    past_tense_train = [
        ("walk -> ", "walked"), ("go -> ", "went"), ("run -> ", "ran"),
        ("eat -> ", "ate"), ("drink -> ", "drank"), ("see -> ", "saw"),
        ("take -> ", "took"), ("give -> ", "gave"), ("make -> ", "made"),
        ("come -> ", "came"), ("know -> ", "knew"), ("think -> ", "thought"),
        ("bring -> ", "brought"), ("buy -> ", "bought"), ("catch -> ", "caught"),
        ("teach -> ", "taught"), ("fight -> ", "fought"), ("seek -> ", "sought"),
        ("feel -> ", "felt"), ("keep -> ", "kept"), ("sleep -> ", "slept"),
        ("leave -> ", "left"), ("meet -> ", "met"), ("send -> ", "sent"),
        ("spend -> ", "spent"), ("build -> ", "built"), ("lend -> ", "lent"),
        ("bend -> ", "bent"), ("sit -> ", "sat"), ("stand -> ", "stood"),
    ]
    past_tense_test = [
        ("jump -> ", "jumped"), ("look -> ", "looked"), ("open -> ", "opened"),
        ("close -> ", "closed"), ("want -> ", "wanted"), ("buy -> ", "bought"),
        ("sing -> ", "sang"), ("swim -> ", "swam"), ("begin -> ", "began"),
        ("ring -> ", "rang"), ("write -> ", "wrote"), ("ride -> ", "rode"),
        ("drive -> ", "drove"), ("rise -> ", "rose"), ("break -> ", "broke"),
        ("speak -> ", "spoke"), ("choose -> ", "chose"), ("freeze -> ", "froze"),
        ("steal -> ", "stole"), ("wake -> ", "woke"),
    ]

    # Next letter task
    next_letter_train = [
        ("A -> ", "B"), ("B -> ", "C"), ("C -> ", "D"), ("D -> ", "E"),
        ("E -> ", "F"), ("F -> ", "G"), ("G -> ", "H"), ("H -> ", "I"),
        ("I -> ", "J"), ("J -> ", "K"), ("K -> ", "L"), ("L -> ", "M"),
        ("M -> ", "N"), ("N -> ", "O"), ("O -> ", "P"), ("P -> ", "Q"),
        ("Q -> ", "R"), ("R -> ", "S"), ("S -> ", "T"), ("T -> ", "U"),
        ("U -> ", "V"), ("V -> ", "W"), ("W -> ", "X"), ("X -> ", "Y"),
    ]
    next_letter_test = [
        ("a -> ", "b"), ("c -> ", "d"), ("e -> ", "f"), ("g -> ", "h"),
        ("i -> ", "j"), ("k -> ", "l"), ("m -> ", "n"), ("o -> ", "p"),
        ("q -> ", "r"), ("s -> ", "t"), ("u -> ", "v"), ("w -> ", "x"),
        ("y -> ", "z"), ("Y -> ", "Z"),
    ]

    return [
        TaskConfig(
            name="antonyms",
            description="Given a word, produce its opposite",
            seed_prompt="Opposites: hot -> cold, big -> small. Now:",
            train_examples=antonyms_train,
            test_examples=antonyms_test,
        ),
        TaskConfig(
            name="plurals",
            description="Given a singular noun, produce the plural form",
            seed_prompt="Plurals: cat -> cats, child -> children. Now:",
            train_examples=plurals_train,
            test_examples=plurals_test,
        ),
        TaskConfig(
            name="past_tense",
            description="Given a verb, produce the past tense",
            seed_prompt="Past tense: walk -> walked, go -> went. Now:",
            train_examples=past_tense_train,
            test_examples=past_tense_test,
        ),
        TaskConfig(
            name="next_letter",
            description="Given a letter, produce the next letter in the alphabet",
            seed_prompt="Next letter: A -> B, M -> N. Now:",
            train_examples=next_letter_train,
            test_examples=next_letter_test,
        ),
    ]


class GBDATrainer:
    """
    GBDA-based prompt trainer for language model tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
        prompt_length: int = 15,
    ):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.embed_layer = self.model.get_input_embeddings()
        self.vocab_size = self.embed_layer.num_embeddings
        self.prompt_length = prompt_length

        print(f"Model loaded. Vocab size: {self.vocab_size}")

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids.cpu().tolist())

    def compute_task_loss_soft(
        self,
        soft_prompt_embeds: torch.Tensor,
        examples: list[tuple[str, str]],
    ) -> torch.Tensor:
        """
        Compute task loss using soft prompt embeddings.

        Args:
            soft_prompt_embeds: Shape (batch_size, prompt_length, embed_dim)
            examples: List of (input, target) pairs

        Returns:
            Average cross-entropy loss
        """
        batch_size = soft_prompt_embeds.shape[0]
        total_loss = 0.0

        for input_text, target_text in examples:
            # Encode input and target
            input_ids = self.encode_text(input_text)
            target_ids = self.encode_text(target_text)

            if len(target_ids) == 0:
                continue

            target_id = target_ids[0].item()  # First token of target

            # Get input embeddings
            input_embeds = self.embed_layer(input_ids)  # (input_len, embed_dim)

            # Concatenate: soft_prompt + input
            # soft_prompt_embeds: (batch, prompt_len, embed_dim)
            # input_embeds: (input_len, embed_dim)
            input_embeds_batch = input_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            full_embeds = torch.cat([soft_prompt_embeds, input_embeds_batch], dim=1)

            # Forward pass
            outputs = self.model(inputs_embeds=full_embeds)
            logits = outputs.logits[:, -1, :]  # (batch, vocab)

            # Cross-entropy loss for target token
            target = torch.full((batch_size,), target_id, device=self.device, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss

        return total_loss / len(examples)

    def compute_fluency_loss_soft(
        self,
        soft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fluency loss (perplexity) for soft token distributions.

        Args:
            soft_tokens: Shape (batch_size, seq_len, vocab_size)

        Returns:
            Average negative log-likelihood
        """
        batch_size, seq_len, vocab_size = soft_tokens.shape

        if seq_len <= 1:
            return torch.tensor(0.0, device=self.device)

        # Get soft embeddings (cast to model dtype)
        embed_weight = self.embed_layer.weight
        soft_embeds = torch.matmul(soft_tokens.to(embed_weight.dtype), embed_weight)

        # Forward through model
        outputs = self.model(inputs_embeds=soft_embeds)
        logits = outputs.logits  # (batch, seq, vocab)

        # Compute cross-entropy between soft tokens and model predictions
        # For position i, compute CE between π_{i+1} and predicted distribution at i
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq-1, vocab)

        # Cross-entropy: -Σ_j π_j * log p_j
        ce = -(soft_tokens[:, 1:, :] * log_probs).sum(dim=-1)  # (batch, seq-1)

        return ce.mean()

    def evaluate_prompt(
        self,
        prompt_tokens: torch.Tensor,
        examples: list[tuple[str, str]],
    ) -> tuple[float, list[tuple[str, str, str, bool]]]:
        """
        Evaluate a discrete prompt on examples.

        Args:
            prompt_tokens: Shape (prompt_length,)
            examples: List of (input, target) pairs

        Returns:
            (accuracy, results) where results is list of (input, prediction, target, correct)
        """
        n_correct = 0
        results = []

        self.model.eval()
        with torch.no_grad():
            for input_text, target_text in examples:
                # Encode input
                input_ids = self.encode_text(input_text)

                # Create full sequence: prompt + input
                full_ids = torch.cat([prompt_tokens, input_ids])

                # Generate
                output_ids = full_ids.tolist()
                for _ in range(5):  # Generate up to 5 tokens
                    input_tensor = torch.tensor([output_ids], device=self.device)
                    outputs = self.model(input_tensor)
                    next_token = outputs.logits[0, -1, :].argmax().item()
                    output_ids.append(next_token)
                    if next_token == self.tokenizer.eos_token_id:
                        break

                # Decode output (only the generated part)
                full_text = self.tokenizer.decode(output_ids)
                base_text = self.tokenizer.decode(full_ids.tolist())
                answer = full_text[len(base_text):].strip()

                # Check correctness
                correct = answer.lower().startswith(target_text.lower())
                if correct:
                    n_correct += 1

                results.append((input_text, answer[:20], target_text, correct))

        self.model.train()
        return n_correct / len(examples), results

    def train_task(
        self,
        task: TaskConfig,
        config: GBDAConfig,
        verbose: bool = True,
    ) -> dict:
        """
        Train a prompt for a specific task using GBDA.

        Args:
            task: Task configuration
            config: GBDA configuration
            verbose: Print progress

        Returns:
            Dictionary with results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.name} - {task.description}")
            print(f"Train examples: {len(task.train_examples)}, Test examples: {len(task.test_examples)}")
            print(f"Seed: '{task.seed_prompt[:50]}...'")
            print(f"{'='*60}")

        # Initialize theta from seed prompt
        seed_tokens = self.encode_text(task.seed_prompt)
        if len(seed_tokens) < config.prompt_length:
            # Pad with period tokens
            pad_token = self.encode_text(".")[0].item()
            padding = torch.full(
                (config.prompt_length - len(seed_tokens),),
                pad_token,
                device=self.device,
                dtype=torch.long
            )
            seed_tokens = torch.cat([seed_tokens, padding])
        seed_tokens = seed_tokens[:config.prompt_length]

        # Initialize theta: high values for seed tokens, zero elsewhere
        theta = torch.zeros(config.prompt_length, self.vocab_size, device=self.device)
        for i, tok_id in enumerate(seed_tokens):
            theta[i, tok_id] = config.init_value
        theta = torch.nn.Parameter(theta)

        # Evaluate seed prompt
        seed_train_acc, _ = self.evaluate_prompt(seed_tokens, task.train_examples)
        seed_test_acc, _ = self.evaluate_prompt(seed_tokens, task.test_examples)

        if verbose:
            seed_text = self.decode_tokens(seed_tokens)
            print(f"\n  BASELINE: '{seed_text}'")
            print(f"  Seed train acc: {seed_train_acc:.1%}, Seed test acc: {seed_test_acc:.1%}")

        # Optimizer
        optimizer = torch.optim.Adam([theta], lr=config.learning_rate)

        best_acc = seed_train_acc
        best_tokens = seed_tokens.clone()
        best_text = self.decode_tokens(seed_tokens)
        best_step = 0

        for step in range(config.num_iterations):
            optimizer.zero_grad()

            # Compute temperature (optionally anneal)
            if config.temperature_anneal:
                progress = step / config.num_iterations
                temp = config.temperature * (1 - progress) + config.temperature_min * progress
            else:
                temp = config.temperature

            # Sample soft tokens using Gumbel-softmax
            logits = theta.unsqueeze(0).expand(config.batch_size, -1, -1)
            soft_tokens = gumbel_softmax_sample(logits, temp, hard=False)

            # Get soft embeddings (cast to model dtype)
            embed_weight = self.embed_layer.weight
            soft_embeds = torch.matmul(soft_tokens.to(embed_weight.dtype), embed_weight)

            # Compute task loss
            task_loss = self.compute_task_loss_soft(soft_embeds, task.train_examples)

            # Compute fluency loss
            if config.lambda_fluency > 0:
                fluency_loss = self.compute_fluency_loss_soft(soft_tokens)
            else:
                fluency_loss = torch.tensor(0.0, device=self.device)

            # Total loss
            total_loss = task_loss + config.lambda_fluency * fluency_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Evaluate current argmax tokens
            if step % 5 == 0:
                with torch.no_grad():
                    current_tokens = theta.argmax(dim=-1)
                    current_text = self.decode_tokens(current_tokens)
                    train_acc, _ = self.evaluate_prompt(current_tokens, task.train_examples)

                    if train_acc > best_acc:
                        best_acc = train_acc
                        best_tokens = current_tokens.clone()
                        best_text = current_text
                        best_step = step

                    if verbose:
                        print(f"  [{step:3d}] task={task_loss.item():.2f} "
                              f"fluency={fluency_loss.item():.2f} "
                              f"acc={train_acc:.0%} best={best_acc:.0%} "
                              f"| {current_text[:40]}")

        # Final evaluation
        final_train_acc, train_results = self.evaluate_prompt(best_tokens, task.train_examples)
        final_test_acc, test_results = self.evaluate_prompt(best_tokens, task.test_examples)

        if verbose:
            print(f"\n>>> Results for {task.name}:")
            print(f"    Best prompt: '{best_text}'")
            print(f"    Train accuracy: {final_train_acc:.1%}")
            print(f"    Test accuracy: {final_test_acc:.1%}")
            print(f"\n    Test examples:")
            for inp, pred, tgt, cor in test_results[:6]:
                mark = "+" if cor else "-"
                print(f"      {mark} {inp} -> '{pred}' (target: {tgt})")

        return {
            "task": task.name,
            "seed_train_acc": seed_train_acc,
            "seed_test_acc": seed_test_acc,
            "best_train_acc": final_train_acc,
            "best_test_acc": final_test_acc,
            "best_prompt": best_text,
            "best_tokens": best_tokens,
            "best_step": best_step,
            "train_results": train_results,
            "test_results": test_results,
        }


def main():
    """Run GBDA prompt optimization on all tasks."""
    print("=" * 70)
    print("GBDA PROMPT OPTIMIZATION")
    print("=" * 70)

    # Configuration
    gbda_config = GBDAConfig(
        prompt_length=15,
        num_iterations=100,
        learning_rate=0.3,
        batch_size=10,
        temperature=1.0,
        temperature_min=0.1,
        temperature_anneal=True,
        lambda_fluency=1.0,
        init_value=12.0,
        device="cuda",
    )

    print(f"\nConfig: lr={gbda_config.learning_rate}, "
          f"batch_size={gbda_config.batch_size}, "
          f"iterations={gbda_config.num_iterations}")
    print(f"        temp={gbda_config.temperature}->{gbda_config.temperature_min}, "
          f"lambda_fluency={gbda_config.lambda_fluency}")

    # Initialize trainer
    trainer = GBDATrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cuda",
        prompt_length=gbda_config.prompt_length,
    )

    # Get tasks
    tasks = create_tasks()

    # Train on each task
    all_results = []
    for task in tasks:
        result = trainer.train_task(task, gbda_config, verbose=True)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - BASELINE vs GBDA OPTIMIZED")
    print("=" * 70)
    print(f"\n{'Task':<15} {'Seed Train':<12} {'Opt Train':<12} {'Seed Test':<12} {'Opt Test':<12} {'Delta':<8}")
    print("-" * 75)

    for r in all_results:
        delta = r["best_test_acc"] - r["seed_test_acc"]
        print(f"{r['task']:<15} {r['seed_train_acc']:>10.1%} {r['best_train_acc']:>10.1%} "
              f"{r['seed_test_acc']:>10.1%} {r['best_test_acc']:>10.1%} {delta:>+7.0%}")

    print("-" * 75)
    avg_seed_train = sum(r["seed_train_acc"] for r in all_results) / len(all_results)
    avg_opt_train = sum(r["best_train_acc"] for r in all_results) / len(all_results)
    avg_seed_test = sum(r["seed_test_acc"] for r in all_results) / len(all_results)
    avg_opt_test = sum(r["best_test_acc"] for r in all_results) / len(all_results)
    avg_delta = avg_opt_test - avg_seed_test
    print(f"{'Average':<15} {avg_seed_train:>10.1%} {avg_opt_train:>10.1%} "
          f"{avg_seed_test:>10.1%} {avg_opt_test:>10.1%} {avg_delta:>+7.0%}")


#%%
if __name__ == "__main__":
    print("Running GBDA prompt optimization...")
    main()
