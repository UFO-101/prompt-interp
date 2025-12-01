# SONAR → Qwen Token Mapping

## Overview

This document describes the learned linear mapping between SONAR token embeddings (1024D) and Qwen token embeddings (896D). The mapping enables converting SONAR-tokenized text into Qwen token space while approximately preserving semantics.

**Model file:** `results/sonar_to_qwen_token_mapper.pt`
**Test accuracy:** 57.0%

---

## Tokenization Differences

### SONAR (NLLB-based SentencePiece)
- **Vocab size:** 256,206 tokens
- **Space handling:** Spaces are word boundaries, NOT preserved as tokens
  - `"the cat"` → `["the", "cat"]` (space used to split, then discarded)
  - `"the  cat"` → `["the", "cat"]` (double space = same result)
- **Word-initial vs Continuation tokens:** Same string can have different token IDs
  - `"ming"` alone = token 10487 (word-initial)
  - `"ming"` in `"programming"` = token 15431 (continuation)
  - Both decode to `"ming"` but are different embeddings
- **No space-prefix tokens:** SONAR has zero tokens that start with a space

### Qwen (BPE-based)
- **Vocab size:** 151,936 tokens
- **Space handling:** Spaces are attached to following tokens
  - `"the cat"` → `["the", " cat"]` (space is part of "cat" token)
- **35% of tokens are space-prefixed:** e.g., `" the"`, `" cat"`, `" is"`
- **No word-initial/continuation distinction:** Each string has one token ID

### Key Implication
When mapping SONAR → Qwen:
- SONAR word-initial tokens should map to Qwen space-prefixed tokens (for mid-sentence)
- SONAR continuation tokens should map to Qwen non-space tokens

---

## Token Space Overlap

| Metric | Value |
|--------|-------|
| SONAR tokens with Qwen match | 39,371 (19.7%) |
| Qwen tokens with SONAR match | 40,985 (27.0%) |
| String overlap | 31,937 (17.9%) |

**~80% of SONAR tokens have no direct Qwen equivalent.** The linear mapping must extrapolate for these.

---

## Training Details

### Data
- **Total pairs:** 40,400
- **Train/Test split:** 95%/5% (by unique SONAR token ID)
- **Pair construction:**
  - Word-initial SONAR tokens → paired with both exact and space-prefixed Qwen tokens
  - Continuation SONAR tokens → paired with exact Qwen tokens only (no space prefix)

### Model
- **Architecture:** Linear layer (1024 → 896)
- **Loss:** Cross-entropy + Triplet margin loss (margin=0.1)
- **Hard negatives:** Top-50 similar Qwen tokens per target
- **Optimizer:** AdamW, lr=5e-4, weight_decay=0.01
- **Training:** 100 epochs

### Why margin loss?
Without margin loss, common punctuation tokens (`,`, ` `, `.`) act as "attractors" - they end up close to many content words in the mapped space. The margin loss pushes incorrect tokens further away, ensuring content words map to content words.

---

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

# Load models
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sonar_embeds = sd.model.decoder.final_proj.weight.data.cuda()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).cuda().eval()
qwen_embeds = qm.model.embed_tokens.weight.data
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)

# Load mapper
class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)
    def forward(self, x):
        return self.linear(x)

mapper = LinearMapper(1024, 896).cuda()
checkpoint = torch.load("results/sonar_to_qwen_token_mapper.pt")
mapper.load_state_dict(checkpoint['model_state'])
mapper.eval()

# Map a sentence
def map_sentence(sentence):
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]  # Skip BOS and lang tag

    output_strs = []
    with torch.no_grad():
        for tok in content_tokens:
            tok_id = tok.item()
            if tok_id == 3:  # EOS
                break

            sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
            mapped = mapper(sonar_emb)
            mapped_norm = F.normalize(mapped, dim=1)

            sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
            qwen_id = sims.argmax().item()
            qwen_s = qt.decode([qwen_id])
            output_strs.append(qwen_s)

    return ''.join(output_strs)

# Example
print(map_sentence("hot becomes cold"))  # " hot becomes cold"
```

---

## Example Mappings

### Works Well (~70% of sentences)
| Input | Output |
|-------|--------|
| Hello world. | Hello world. |
| I love you. | I love you. |
| hot becomes cold | hot becomes cold |
| big becomes small | big becomes small |
| How are you today? | How are you today? |
| Find the antonym. | Find the antonym. |
| Machine learning is transforming the world. | Machine learning is transforming the world. |
| I think therefore I am. | I think therefore I am. |
| Can you help me with this problem? | Can you help me with this problem? |

### Minor Issues (~18% - spacing glitches)
| Input | Output | Issue |
|-------|--------|-------|
| The cat sat on the mat. | The cat sat on themat. | Missing space |
| What is your name? | What is yourname? | Missing space |
| I am not sure what you mean. | I amnot sure what you mean. | Missing space |

### Significant Issues (~12% - rare words, numbers)
| Input | Output | Issue |
|-------|--------|-------|
| Explain quantum physics. | Explin quantum physcs. | Rare words mangled |
| The answer is 42. | The answer is ." | Numbers fail |
| Call me at 555-1234. | Call meat5PP3. | Numbers badly garbled |

---

## Limitations

1. **Numbers fail:** The mapping doesn't handle digits well
2. **Rare/technical words:** May be mangled if not in training vocabulary
3. **~80% extrapolation:** Most SONAR tokens have no direct Qwen match
4. **Spacing imperfect:** Occasional missing spaces between words
5. **57% token accuracy:** Not suitable for exact reconstruction

---

## When to Use This

**Good for:**
- Approximate semantic preservation
- Common English vocabulary
- Simple to moderately complex sentences
- Tasks where exact wording isn't critical

**Not suitable for:**
- Exact text reconstruction
- Numerical data
- Technical/specialized terminology
- Languages other than English

---

## Files

- `results/sonar_to_qwen_token_mapper.pt` - Trained model checkpoint
- `scripts/train_final_mapper.py` - Training script
- `scripts/token_mapping_proper.py` - Earlier version with analysis
- `scripts/analyze_mapping.py` - Detailed failure analysis
