"""Compare SONAR/NLLB tokenization vs Qwen tokenization."""

import torch
from transformers import AutoTokenizer
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

print("Loading...", flush=True)
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')
sonar_dec = sd.tokenizer.create_decoder()
print("Ready.\n", flush=True)

test_texts = [
    "Give the opposite word for each input.",
    "hot -> cold",
    "The quick brown fox.",
    "Antonyms: words with opposite meanings.",
]

print("=" * 80)
print("TOKENIZATION COMPARISON: SONAR/NLLB vs Qwen")
print("=" * 80)

for text in test_texts:
    print(f"\nText: '{text}'")
    print("-" * 80)

    # SONAR tokenization
    sonar_tokens = sonar_enc(text)
    # Skip BOS (0) and lang tag (1)
    sonar_content = sonar_tokens[2:] if len(sonar_tokens) > 2 else sonar_tokens

    # Qwen tokenization
    qwen_tokens = qt.encode(text, add_special_tokens=False)

    print(f"SONAR: {len(sonar_content)} tokens")
    for i, tok in enumerate(sonar_content):
        try:
            tok_str = sonar_dec(torch.tensor([tok.item()]))
        except:
            tok_str = f"<{tok.item()}>"
        print(f"  {i}: {tok.item():6d} -> '{tok_str}'")

    print(f"Qwen:  {len(qwen_tokens)} tokens")
    for i, tok in enumerate(qwen_tokens):
        tok_str = qt.decode([tok])
        print(f"  {i}: {tok:6d} -> '{tok_str}'")

    print(f"\nLength mismatch: {len(sonar_content)} vs {len(qwen_tokens)}")


# Check shared vocabulary
print("\n" + "=" * 80)
print("SHARED VOCABULARY ANALYSIS")
print("=" * 80)

# Sample some Qwen tokens and see if they exist in SONAR
sample_words = ["the", "a", "is", "and", "to", "of", "hot", "cold", "big", "small"]

print("\nChecking if words tokenize to single tokens in both:")
for word in sample_words:
    qwen_toks = qt.encode(word, add_special_tokens=False)
    sonar_toks = sonar_enc(word)[2:]  # Skip BOS and lang tag

    qwen_single = len(qwen_toks) == 1
    sonar_single = len(sonar_toks) == 1

    status = "✓" if qwen_single and sonar_single else "✗"
    print(f"  '{word}': Qwen={len(qwen_toks)} tok, SONAR={len(sonar_toks)} tok {status}")
