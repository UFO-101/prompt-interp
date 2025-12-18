"""
SONAR Prompt Optimization

Learns interpretable text prompts by optimizing in SONAR embedding space.

Two-stage approach:
1. Stage 1: Optimize z embedding that generates prompt tokens via SONAR decoder
2. Stage 2: Evaluate the prompt with z=0 (unconditioned) on the task

Key technique: Straight-through gradient estimation via embedding geometry.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)
from fairseq2.nn.batch_layout import BatchLayout


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptimConfig:
    """Optimization hyperparameters."""
    lr: float = 0.001
    ppl_weight: float = 0.1
    max_steps: int = 50
    max_prompt_len: int = 15
    grad_clip: float = 0.5
    early_stop_loss_mult: float = 3.0
    early_stop_zero_acc_count: int = 5
    grad_accumulation_steps: int = 1  # Accumulate gradients over N steps before updating
    n_noise_samples: int = 1  # Number of noised z variations to average gradients over
    noise_std: float = 0.1  # Standard deviation of noise added to z
    device: str = "cuda"


@dataclass
class Task:
    """A task with train and test examples."""
    name: str
    train_examples: list[tuple[str, str]]
    test_examples: list[tuple[str, str]]
    seed_prompt: str
    description: str = ""


# =============================================================================
# Task Definitions
# =============================================================================

TASKS = {
    "antonyms": Task(
        name="antonyms",
        description="Given a word, produce its opposite",
        seed_prompt="Opposites: hot -> cold, big -> small. Now:",
        train_examples=[
            # Core opposites
            ("hot -> ", "cold"),
            ("big -> ", "small"),
            ("fast -> ", "slow"),
            ("up -> ", "down"),
            ("happy -> ", "sad"),
            ("light -> ", "dark"),
            ("good -> ", "bad"),
            ("old -> ", "young"),
            ("high -> ", "low"),
            ("long -> ", "short"),
            ("hard -> ", "soft"),
            ("open -> ", "closed"),
            # Additional training examples
            ("tall -> ", "short"),
            ("new -> ", "old"),
            ("right -> ", "wrong"),
            ("true -> ", "false"),
            ("alive -> ", "dead"),
            ("black -> ", "white"),
            ("day -> ", "night"),
            ("start -> ", "stop"),
            ("win -> ", "lose"),
            ("push -> ", "pull"),
            ("buy -> ", "sell"),
            ("love -> ", "hate"),
            ("sweet -> ", "sour"),
            ("smooth -> ", "rough"),
            ("sharp -> ", "dull"),
            ("tight -> ", "loose"),
            ("bright -> ", "dim"),
            ("brave -> ", "coward"),
        ],
        test_examples=[
            ("wet -> ", "dry"),
            ("loud -> ", "quiet"),
            ("rich -> ", "poor"),
            ("full -> ", "empty"),
            ("strong -> ", "weak"),
            ("thick -> ", "thin"),
            ("early -> ", "late"),
            ("clean -> ", "dirty"),
            ("deep -> ", "shallow"),
            ("wide -> ", "narrow"),
            ("heavy -> ", "light"),
            ("safe -> ", "dangerous"),
            ("inside -> ", "outside"),
            ("front -> ", "back"),
            ("top -> ", "bottom"),
            ("first -> ", "last"),
            ("near -> ", "far"),
            ("cheap -> ", "expensive"),
            ("easy -> ", "hard"),
            ("warm -> ", "cool"),
        ],
    ),
    "plurals": Task(
        name="plurals",
        description="Given a singular noun, produce the plural form",
        seed_prompt="Plurals: cat -> cats, child -> children. Now:",
        train_examples=[
            # Regular plurals
            ("cat -> ", "cats"),
            ("dog -> ", "dogs"),
            ("car -> ", "cars"),
            ("book -> ", "books"),
            ("tree -> ", "trees"),
            ("house -> ", "houses"),
            ("table -> ", "tables"),
            ("chair -> ", "chairs"),
            ("bird -> ", "birds"),
            ("flower -> ", "flowers"),
            # Irregular plurals
            ("child -> ", "children"),
            ("mouse -> ", "mice"),
            ("foot -> ", "feet"),
            ("tooth -> ", "teeth"),
            ("man -> ", "men"),
            ("woman -> ", "women"),
            ("goose -> ", "geese"),
            ("person -> ", "people"),
            # -f/-fe -> -ves
            ("leaf -> ", "leaves"),
            ("knife -> ", "knives"),
            ("wolf -> ", "wolves"),
            ("life -> ", "lives"),
            ("wife -> ", "wives"),
            # -y -> -ies
            ("city -> ", "cities"),
            ("baby -> ", "babies"),
            ("party -> ", "parties"),
            ("story -> ", "stories"),
            ("country -> ", "countries"),
            # -es endings
            ("box -> ", "boxes"),
            ("bus -> ", "buses"),
        ],
        test_examples=[
            # Regular
            ("apple -> ", "apples"),
            ("river -> ", "rivers"),
            ("window -> ", "windows"),
            ("friend -> ", "friends"),
            # Irregular
            ("ox -> ", "oxen"),
            ("louse -> ", "lice"),
            # Same singular/plural
            ("fish -> ", "fish"),
            ("sheep -> ", "sheep"),
            ("deer -> ", "deer"),
            ("moose -> ", "moose"),
            # -f/-fe
            ("calf -> ", "calves"),
            ("half -> ", "halves"),
            ("shelf -> ", "shelves"),
            # -y -> -ies
            ("family -> ", "families"),
            ("battery -> ", "batteries"),
            # -es
            ("church -> ", "churches"),
            ("dish -> ", "dishes"),
            ("hero -> ", "heroes"),
            ("potato -> ", "potatoes"),
            ("tomato -> ", "tomatoes"),
        ],
    ),
    "past_tense": Task(
        name="past_tense",
        description="Given a verb, produce the past tense",
        seed_prompt="Past tense: walk -> walked, go -> went. Now:",
        train_examples=[
            # Regular verbs
            ("walk -> ", "walked"),
            ("talk -> ", "talked"),
            ("play -> ", "played"),
            ("work -> ", "worked"),
            ("watch -> ", "watched"),
            ("call -> ", "called"),
            ("ask -> ", "asked"),
            ("help -> ", "helped"),
            ("start -> ", "started"),
            ("finish -> ", "finished"),
            # Irregular verbs
            ("go -> ", "went"),
            ("run -> ", "ran"),
            ("eat -> ", "ate"),
            ("see -> ", "saw"),
            ("take -> ", "took"),
            ("make -> ", "made"),
            ("come -> ", "came"),
            ("give -> ", "gave"),
            ("think -> ", "thought"),
            ("say -> ", "said"),
            ("get -> ", "got"),
            ("know -> ", "knew"),
            ("find -> ", "found"),
            ("tell -> ", "told"),
            ("become -> ", "became"),
            ("leave -> ", "left"),
            ("feel -> ", "felt"),
            ("put -> ", "put"),
            ("bring -> ", "brought"),
            ("begin -> ", "began"),
        ],
        test_examples=[
            # Regular
            ("jump -> ", "jumped"),
            ("look -> ", "looked"),
            ("open -> ", "opened"),
            ("close -> ", "closed"),
            ("want -> ", "wanted"),
            # Irregular
            ("buy -> ", "bought"),
            ("catch -> ", "caught"),
            ("teach -> ", "taught"),
            ("fight -> ", "fought"),
            ("seek -> ", "sought"),
            ("sleep -> ", "slept"),
            ("keep -> ", "kept"),
            ("meet -> ", "met"),
            ("send -> ", "sent"),
            ("build -> ", "built"),
            ("spend -> ", "spent"),
            ("lose -> ", "lost"),
            ("sit -> ", "sat"),
            ("stand -> ", "stood"),
            ("understand -> ", "understood"),
        ],
    ),
    "next_letter": Task(
        name="next_letter",
        description="Given a letter, produce the next letter in the alphabet",
        seed_prompt="Next letter: A -> B, M -> N. Now:",
        train_examples=[
            ("A -> ", "B"),
            ("B -> ", "C"),
            ("C -> ", "D"),
            ("D -> ", "E"),
            ("E -> ", "F"),
            ("F -> ", "G"),
            ("G -> ", "H"),
            ("H -> ", "I"),
            ("I -> ", "J"),
            ("J -> ", "K"),
            ("K -> ", "L"),
            ("L -> ", "M"),
            ("M -> ", "N"),
            ("N -> ", "O"),
            ("O -> ", "P"),
            ("P -> ", "Q"),
            ("Q -> ", "R"),
            ("R -> ", "S"),
            ("S -> ", "T"),
            ("T -> ", "U"),
            ("U -> ", "V"),
            ("V -> ", "W"),
            ("W -> ", "X"),
            ("X -> ", "Y"),
        ],
        test_examples=[
            # Lowercase (different representation)
            ("a -> ", "b"),
            ("c -> ", "d"),
            ("e -> ", "f"),
            ("g -> ", "h"),
            ("i -> ", "j"),
            ("k -> ", "l"),
            ("m -> ", "n"),
            ("o -> ", "p"),
            ("q -> ", "r"),
            ("s -> ", "t"),
            ("u -> ", "v"),
            ("w -> ", "x"),
            ("x -> ", "y"),
            ("y -> ", "z"),
        ],
    ),
}


# =============================================================================
# Model Wrapper
# =============================================================================

class SonarPromptOptimizer:
    """Optimizes prompts in SONAR embedding space."""

    def __init__(self, config: OptimConfig):
        self.config = config
        self.device = config.device

        print("Loading SONAR models...", flush=True)
        self.encoder = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder'
        )
        self.decoder_pipeline = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder'
        )
        self.decoder = self.decoder_pipeline.model.to(self.device)
        self.tokenizer_decode = self.decoder_pipeline.tokenizer.create_decoder()
        self.tokenizer_encode = self.decoder_pipeline.tokenizer.create_encoder(
            mode='target', lang='eng_Latn'
        )

        self.embed_layer = self.decoder.decoder.decoder_frontend.embed
        self.embed_matrix = self.embed_layer.weight.detach()
        self.z_zero = torch.zeros(1, 1, 1024, device=self.device)

        print("Models loaded.", flush=True)

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert token IDs to text, stripping special tokens."""
        tokens = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        if len(content) == 0:
            return ""
        return self.tokenizer_decode(torch.tensor(content))

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        return self.tokenizer_encode(text).to(self.device)

    def generate_prompt(self, z: torch.Tensor, max_len: int = None) -> tuple[list[int], list[torch.Tensor]]:
        """Generate prompt tokens from embedding z, returning tokens and logits."""
        max_len = max_len or self.config.max_prompt_len
        tokens = [3, 256047]  # BOS, lang token
        all_logits = []
        e = z.unsqueeze(1)

        for _ in range(max_len):
            di = torch.tensor([tokens], device=self.device)
            h = self.decoder.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = self.decoder.decoder.final_proj(h)[0, -1, :]
            all_logits.append(logits)
            next_tok = logits.argmax().item()
            tokens.append(next_tok)
            if next_tok == 3:  # EOS
                break

        # Remove EOS if present
        if tokens[-1] == 3:
            tokens = tokens[:-1]
            all_logits = all_logits[:-1]

        return tokens, all_logits

    def compute_ppl_gradients(self, prompt_tokens: list[int]) -> tuple[float, list[torch.Tensor]]:
        """Compute perplexity and embedding gradients for prompt tokens."""
        if len(prompt_tokens) <= 2:
            return 0.0, []

        ppl_embed_grads = []
        total_nll = 0.0
        n_tokens = 0

        for i in range(2, len(prompt_tokens)):
            input_seq = prompt_tokens[:i]
            target_token = prompt_tokens[i]

            embedding_output = []
            def embed_hook(module, inp, out):
                out.retain_grad()
                embedding_output.append(out)
                return out

            handle = self.embed_layer.register_forward_hook(embed_hook)

            di = torch.tensor([input_seq], device=self.device)
            h = self.decoder.decode(di, BatchLayout.of(di), self.z_zero, BatchLayout.of(self.z_zero))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = self.decoder.decoder.final_proj(h)[0, -1, :]

            handle.remove()

            target = torch.tensor([target_token], device=self.device, dtype=torch.long)
            nll = F.cross_entropy(logits.unsqueeze(0), target)
            total_nll += nll.item()
            n_tokens += 1

            nll.backward(retain_graph=False)

            if embedding_output and embedding_output[0].grad is not None:
                embed_grad = embedding_output[0].grad[0]
                full_grad = torch.zeros(len(prompt_tokens), embed_grad.shape[-1], device=self.device)
                full_grad[:len(embed_grad)] = embed_grad
                ppl_embed_grads.append(full_grad)

            self.decoder.zero_grad()

        ppl = torch.exp(torch.tensor(total_nll / n_tokens)).item() if n_tokens > 0 else 0.0
        return ppl, ppl_embed_grads

    def train_step_single_z(
        self,
        z_sample: torch.Tensor,
        examples: list[tuple[str, str]]
    ) -> tuple[float, float, str, list[int], float, list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute loss and embedding gradients for a single z sample.

        Returns: (loss, accuracy, prompt_text, prompt_tokens, ppl, stage1_logits, combined_grad)
        """
        prompt_tokens, stage1_logits = self.generate_prompt(z_sample)
        prompt_text = self.tokens_to_text(prompt_tokens)

        total_loss = 0.0
        n_correct = 0
        task_embed_grads = []

        for input_text, target_text in examples:
            task_tokens = self.text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

            target_tokens = self.text_to_tokens(target_text)
            target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
            target_id = target_content[0].item()

            full_tokens = prompt_tokens + task_content.tolist() + [target_id]
            input_tokens = full_tokens[:-1]

            embedding_output = []
            def embed_hook(module, inp, out):
                out.retain_grad()
                embedding_output.append(out)
                return out

            handle = self.embed_layer.register_forward_hook(embed_hook)

            di = torch.tensor([input_tokens], device=self.device)
            h = self.decoder.decode(di, BatchLayout.of(di), self.z_zero, BatchLayout.of(self.z_zero))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = self.decoder.decoder.final_proj(h)[0, -1, :]

            handle.remove()

            target = torch.tensor([target_id], device=self.device, dtype=torch.long)
            task_loss = F.cross_entropy(logits.unsqueeze(0), target)
            total_loss += task_loss.item()

            if logits.argmax().item() == target_id:
                n_correct += 1

            task_loss.backward(retain_graph=False)

            if embedding_output and embedding_output[0].grad is not None:
                embed_grad = embedding_output[0].grad[0]
                prompt_embed_grad = embed_grad[:len(prompt_tokens)]
                task_embed_grads.append(prompt_embed_grad.clone())

            self.decoder.zero_grad()

        # Compute PPL regularization gradients
        ppl, ppl_embed_grads = self.compute_ppl_gradients(prompt_tokens)

        # Compute combined gradient in embedding space
        combined_grad = None
        if task_embed_grads:
            avg_task_grad = torch.stack(task_embed_grads).mean(dim=0)

            if ppl_embed_grads and self.config.ppl_weight > 0:
                avg_ppl_grad = torch.stack(ppl_embed_grads).mean(dim=0)[:len(prompt_tokens)]
                combined_grad = avg_task_grad + self.config.ppl_weight * avg_ppl_grad
            else:
                combined_grad = avg_task_grad

        return (
            total_loss / len(examples),
            n_correct / len(examples),
            prompt_text,
            prompt_tokens,
            ppl,
            stage1_logits,
            combined_grad
        )

    def train_step(
        self,
        z: nn.Parameter,
        examples: list[tuple[str, str]]
    ) -> tuple[float, float, str, list[int], float]:
        """
        Single training step with optional noised z variations.

        Returns: (loss, accuracy, prompt_text, prompt_tokens, ppl)
        """
        n_samples = self.config.n_noise_samples
        noise_std = self.config.noise_std

        # Always include the original z (no noise) as the first sample
        z_samples = [z]
        for _ in range(n_samples - 1):
            noise = torch.randn_like(z) * noise_std
            z_samples.append(z + noise)

        all_losses = []
        all_accs = []
        all_logit_grads = []

        # Use the first sample's prompt for reporting
        main_prompt_text = None
        main_prompt_tokens = None
        main_ppl = None

        for i, z_sample in enumerate(z_samples):
            loss, acc, prompt_text, prompt_tokens, ppl, stage1_logits, combined_grad = \
                self.train_step_single_z(z_sample, examples)

            all_losses.append(loss)
            all_accs.append(acc)

            if i == 0:
                main_prompt_text = prompt_text
                main_prompt_tokens = prompt_tokens
                main_ppl = ppl

            # Compute logit-space gradients for this sample
            if combined_grad is not None and stage1_logits:
                logit_grads = combined_grad @ self.embed_matrix.T
                logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

                if logit_grads.shape[0] > 2:
                    # Store the gradients aligned with stage1_logits
                    grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]
                    all_logit_grads.append((stage1_logits, grad_for_stage1))

        # Average and apply gradients from all samples
        if all_logit_grads:
            # For simplicity, we'll apply gradients from all samples
            # by computing surrogate loss for each and averaging
            for stage1_logits, grad_for_stage1 in all_logit_grads:
                total_surrogate = sum(
                    (l * g).sum() for l, g in zip(stage1_logits, grad_for_stage1)
                )
                # Scale by 1/n_samples to average
                (total_surrogate / len(all_logit_grads)).backward(retain_graph=True)

        return (
            sum(all_losses) / len(all_losses),
            sum(all_accs) / len(all_accs),
            main_prompt_text,
            main_prompt_tokens,
            main_ppl
        )

    def evaluate(
        self,
        prompt_tokens: list[int],
        examples: list[tuple[str, str]],
        max_gen_len: int = 5
    ) -> tuple[float, list[dict]]:
        """
        Evaluate prompt on examples using greedy decoding with z=0.

        Returns: (accuracy, list of result dicts)
        """
        n_correct = 0
        results = []

        for input_text, target_text in examples:
            task_tokens = self.text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
            full = prompt_tokens + task_content.tolist()

            with torch.no_grad():
                for _ in range(max_gen_len):
                    di = torch.tensor([full], device=self.device)
                    h = self.decoder.decode(di, BatchLayout.of(di), self.z_zero, BatchLayout.of(self.z_zero))
                    if h.dim() == 4:
                        h = h.squeeze(1)
                    logits = self.decoder.decoder.final_proj(h)[0, -1, :]
                    next_tok = logits.argmax().item()
                    full.append(next_tok)
                    if next_tok == 3:
                        break

            full_text = self.tokens_to_text(full)
            base_text = self.tokens_to_text(prompt_tokens + task_content.tolist())
            answer = full_text[len(base_text):].strip()

            # Check if answer starts with target (case-insensitive)
            correct = answer.lower().startswith(target_text.lower())
            if correct:
                n_correct += 1

            results.append({
                "input": input_text,
                "target": target_text,
                "prediction": answer[:30],
                "correct": correct,
            })

        return n_correct / len(examples), results

    def optimize(
        self,
        task: Task,
        seed_prompt: str = None,
        verbose: bool = True
    ) -> dict:
        """
        Optimize a prompt for a task.

        Returns dict with best_acc, best_prompt, best_tokens, train/test results.
        """
        seed = seed_prompt or task.seed_prompt

        # Initialize z from seed prompt
        with torch.no_grad():
            z_init = self.encoder.predict([seed], source_lang='eng_Latn').to(self.device)

        # Evaluate seed prompt FIRST (baseline)
        seed_tokens, _ = self.generate_prompt(z_init)
        seed_decoded = self.tokens_to_text(seed_tokens)
        seed_train_acc, seed_train_results = self.evaluate(seed_tokens, task.train_examples)
        seed_test_acc, seed_test_results = self.evaluate(seed_tokens, task.test_examples)

        if verbose:
            print(f"\n  BASELINE (seed prompt decoded): '{seed_decoded}'")
            print(f"  Seed train acc: {seed_train_acc:.1%}, Seed test acc: {seed_test_acc:.1%}")

        z = nn.Parameter(z_init.clone())
        optimizer = torch.optim.Adam([z], lr=self.config.lr)

        best_acc = seed_train_acc  # Start with seed accuracy as baseline
        best_prompt = seed_decoded
        best_tokens = seed_tokens.copy() if isinstance(seed_tokens, list) else list(seed_tokens)
        best_step = 0
        initial_loss = None
        zero_acc_count = 0
        accum_steps = self.config.grad_accumulation_steps

        for step in range(self.config.max_steps):
            # Gradient accumulation: only zero grad at start of accumulation window
            if step % accum_steps == 0:
                optimizer.zero_grad()

            loss, train_acc, prompt, tokens, ppl = self.train_step(z, task.train_examples)

            if initial_loss is None:
                initial_loss = loss

            # Only update at end of accumulation window
            if (step + 1) % accum_steps == 0 or step == self.config.max_steps - 1:
                if z.grad is not None:
                    # Scale gradients by accumulation steps
                    z.grad.data.div_(accum_steps)
                    torch.nn.utils.clip_grad_norm_([z], max_norm=self.config.grad_clip)
                    optimizer.step()

            # Evaluate with z=0
            z0_acc, _ = self.evaluate(tokens, task.train_examples)

            if z0_acc > best_acc:
                best_acc = z0_acc
                best_prompt = prompt
                best_tokens = tokens.copy()
                best_step = step
                zero_acc_count = 0
            elif z0_acc == 0:
                zero_acc_count += 1

            if verbose and step % 5 == 0:
                print(f"  [{step:2d}] loss={loss:.2f} ppl={ppl:5.0f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:40]}")

            # Early stopping
            if loss > initial_loss * self.config.early_stop_loss_mult:
                if verbose:
                    print(f"  -> Early stop: loss exploded at step {step}")
                break
            if zero_acc_count >= self.config.early_stop_zero_acc_count:
                if verbose:
                    print(f"  -> Early stop: z0_acc=0 for {zero_acc_count} steps")
                break

        # Final evaluation on train and test
        train_acc, train_results = self.evaluate(best_tokens, task.train_examples) if best_tokens else (0, [])
        test_acc, test_results = self.evaluate(best_tokens, task.test_examples) if best_tokens else (0, [])

        return {
            "task": task.name,
            "seed": seed,
            "seed_decoded": seed_decoded,
            "seed_train_acc": seed_train_acc,
            "seed_test_acc": seed_test_acc,
            "best_train_acc": best_acc,
            "best_prompt": best_prompt,
            "best_tokens": best_tokens,
            "best_step": best_step,
            "final_train_acc": train_acc,
            "final_test_acc": test_acc,
            "train_results": train_results,
            "test_results": test_results,
            "config": {
                "lr": self.config.lr,
                "ppl_weight": self.config.ppl_weight,
                "max_steps": self.config.max_steps,
            }
        }


# =============================================================================
# Evaluation Runner
# =============================================================================

def run_evaluation(tasks: list[str] = None, verbose: bool = True):
    """Run comprehensive evaluation across tasks."""

    config = OptimConfig(
        lr=0.001,
        ppl_weight=0.1,
        max_steps=100,
        grad_accumulation_steps=1,  # Simpler - use noise averaging instead
        n_noise_samples=4,  # Average gradients over 4 noised z variations
        noise_std=0.1,  # Noise standard deviation
    )

    optimizer = SonarPromptOptimizer(config)

    task_names = tasks or list(TASKS.keys())
    all_results = []

    print("\n" + "=" * 70)
    print("SONAR PROMPT OPTIMIZATION - Comprehensive Evaluation")
    print("=" * 70)
    print(f"Config: lr={config.lr}, ppl_weight={config.ppl_weight}, max_steps={config.max_steps}")
    print(f"        n_noise_samples={config.n_noise_samples}, noise_std={config.noise_std}")
    print(f"Tasks: {', '.join(task_names)}")
    print("=" * 70)

    for task_name in task_names:
        task = TASKS[task_name]

        print(f"\n{'='*60}")
        print(f"Task: {task.name} - {task.description}")
        print(f"Train examples: {len(task.train_examples)}, Test examples: {len(task.test_examples)}")
        print(f"Seed: '{task.seed_prompt[:50]}...'")
        print("=" * 60)

        result = optimizer.optimize(task, verbose=verbose)
        all_results.append(result)

        print(f"\n>>> Results for {task.name}:")
        print(f"    Best prompt: '{result['best_prompt']}'")
        print(f"    Train accuracy: {result['final_train_acc']:.1%}")
        print(f"    Test accuracy: {result['final_test_acc']:.1%}")

        if verbose and result['test_results']:
            print(f"\n    Test examples:")
            for r in result['test_results'][:6]:
                mark = "+" if r['correct'] else "-"
                print(f"      {mark} {r['input']} -> '{r['prediction']}' (target: {r['target']})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - BASELINE vs OPTIMIZED")
    print("=" * 70)

    print(f"\n{'Task':<15} {'Seed Train':>12} {'Opt Train':>12} {'Seed Test':>12} {'Opt Test':>12} {'Delta':>8}")
    print("-" * 75)

    for r in all_results:
        delta_test = r['final_test_acc'] - r['seed_test_acc']
        delta_str = f"+{delta_test:.0%}" if delta_test >= 0 else f"{delta_test:.0%}"
        print(f"{r['task']:<15} {r['seed_train_acc']:>12.1%} {r['final_train_acc']:>12.1%} {r['seed_test_acc']:>12.1%} {r['final_test_acc']:>12.1%} {delta_str:>8}")

    avg_seed_train = sum(r['seed_train_acc'] for r in all_results) / len(all_results)
    avg_train = sum(r['final_train_acc'] for r in all_results) / len(all_results)
    avg_seed_test = sum(r['seed_test_acc'] for r in all_results) / len(all_results)
    avg_test = sum(r['final_test_acc'] for r in all_results) / len(all_results)
    avg_delta = avg_test - avg_seed_test
    delta_str = f"+{avg_delta:.0%}" if avg_delta >= 0 else f"{avg_delta:.0%}"
    print("-" * 75)
    print(f"{'Average':<15} {avg_seed_train:>12.1%} {avg_train:>12.1%} {avg_seed_test:>12.1%} {avg_test:>12.1%} {delta_str:>8}")

    # Detailed breakdown by task
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for r in all_results:
        print(f"\n{r['task'].upper()}")
        print(f"  Optimized prompt: '{r['best_prompt']}'")
        print(f"  Train: {r['final_train_acc']:.1%} ({sum(1 for x in r['train_results'] if x['correct'])}/{len(r['train_results'])})")
        print(f"  Test:  {r['final_test_acc']:.1%} ({sum(1 for x in r['test_results'] if x['correct'])}/{len(r['test_results'])})")

        # Show failures
        failures = [x for x in r['test_results'] if not x['correct']]
        if failures:
            print(f"  Test failures:")
            for f in failures[:5]:
                print(f"    - {f['input']} -> '{f['prediction']}' (expected: {f['target']})")

    # Save results
    results_path = "results/evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in all_results:
            jr = {k: v for k, v in r.items() if k != 'best_tokens'}
            json_results.append(jr)
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    torch.cuda.empty_cache()
    results = run_evaluation(verbose=True)
