# SONAR Embedding Noise Robustness

How much noise can be added to a SONAR embedding while retaining semantic content?

## Method
- Add Gaussian noise scaled by `noise_level * original_norm`
- Test with and without projecting back to original norm

## Results by Prompt

### "Once upon a time, there was a little girl named Li..."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | Once upon a time, there was a little gir... ✓ |
| 0.00 | True | 1.000 | Once upon a time, there was a little gir... ✓ |
| 0.03 | False | 0.722 | Once upon a time, there was a little gir... ✓ |
| 0.03 | True | 0.712 | Once upon a time, there was a little gir... |
| 0.04 | False | 0.674 | Once upon a time, there was a little gir... ✓ |
| 0.04 | True | 0.670 | There was once upon a time, there was a ... |
| 0.04 | False | 0.624 | Once upon a time, there was a little gir... ✓ |
| 0.04 | True | 0.586 | There was once upon a time a girl named ... |
| 0.04 | False | 0.566 | Once upon a time, there was a girl named... |
| 0.04 | True | 0.540 | There was a time, a life, a lifetime, a ... |
| 0.05 | False | 0.520 | Once upon a time, there was a little gir... |
| 0.05 | True | 0.550 | Once upon a time, there was a little gir... |
| 0.06 | False | 0.499 | Once upon a time, there was a little gir... ✓ |
| 0.06 | True | 0.472 | Once upon a time there was Lily-Anne the... |
| 0.06 | False | 0.477 | Once upon a time, there was a little gir... |
| 0.06 | True | 0.456 | Once upon a time, there was a little gir... |

### "The weather is nice today."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | The weather is nice today.... ✓ |
| 0.00 | True | 1.000 | The weather is nice today.... ✓ |
| 0.03 | False | 0.737 | Weather is nice today.... |
| 0.03 | True | 0.700 | The weather is nice weather today.... |
| 0.04 | False | 0.676 | Weather is nice today.... |
| 0.04 | True | 0.684 | The weather is nice today.... ✓ |
| 0.04 | False | 0.590 | The weather is nice today.... ✓ |
| 0.04 | True | 0.614 | The weather is nice today.... ✓ |
| 0.04 | False | 0.590 | Today there is nice weather.... |
| 0.04 | True | 0.535 | You know, the weather is fine in China.... |
| 0.05 | False | 0.529 | The weather is nice today.... ✓ |
| 0.05 | True | 0.530 | Today the weather is nice.... |
| 0.06 | False | 0.479 | Weather is fine today.... |
| 0.06 | True | 0.489 | The sunny water is beautiful and the sun... |
| 0.06 | False | 0.468 | It's been a lovely day today.... |
| 0.06 | True | 0.472 | The weather's nice. The weather's nice.... |

### "I like cheese."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | I like cheese.... ✓ |
| 0.00 | True | 1.000 | I like cheese.... ✓ |
| 0.03 | False | 0.711 | I like cheese.... ✓ |
| 0.03 | True | 0.730 | I like it is cheese.... |
| 0.04 | False | 0.686 | I like cheese.... ✓ |
| 0.04 | True | 0.657 | I like cheese because I like cheese.... |
| 0.04 | False | 0.600 | I like cheese.... ✓ |
| 0.04 | True | 0.582 | I like the cheese, I like the cheese.... |
| 0.04 | False | 0.573 | I like the cheese.... |
| 0.04 | True | 0.563 | I like cheese, but I like cheese.... |
| 0.05 | False | 0.551 | I've got a lot of cheese.... |
| 0.05 | True | 0.529 | I like cheese and I like cheese.... |
| 0.06 | False | 0.499 | I like the cheese.... |
| 0.06 | True | 0.478 | I'm not a fan of the cheesecake.... |
| 0.06 | False | 0.463 | I like the cheese.... |
| 0.06 | True | 0.436 | I learned to like cheese for reasons I l... |

### "He decided to go on an adventure."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | He decided to go on an adventure.... ✓ |
| 0.00 | True | 1.000 | He decided to go on an adventure.... ✓ |
| 0.03 | False | 0.723 | He decided on an adventure.... |
| 0.03 | True | 0.729 | He decided to go on adventure to go on a... |
| 0.04 | False | 0.681 | He decided to go on an adventure.... ✓ |
| 0.04 | True | 0.662 | He decided to go on an adventure.... ✓ |
| 0.04 | False | 0.597 | He decided to go on an adventure.... ✓ |
| 0.04 | True | 0.605 | He decided to go with an adventure and g... |
| 0.04 | False | 0.578 | He decided to go on an adventure.... ✓ |
| 0.04 | True | 0.544 | He decided to go on an adventure.... ✓ |
| 0.05 | False | 0.505 | He the adventured to go on a adventure.... |
| 0.05 | True | 0.544 | He decided to go on an adventure.... ✓ |
| 0.06 | False | 0.525 | He decided to go on an adventure.... ✓ |
| 0.06 | True | 0.499 | He's supposed to make a decision. He's s... |
| 0.06 | False | 0.499 | He decided to go on an adventure.... ✓ |
| 0.06 | True | 0.458 | He decided he read to go on a quest for ... |

### "She loved to play outside with her toys."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | She loved to play outside with her toys.... ✓ |
| 0.00 | True | 1.000 | She loved to play outside with her toys.... ✓ |
| 0.03 | False | 0.718 | She loved playing outside with her toys.... |
| 0.03 | True | 0.726 | She liked to play outside with her toys.... |
| 0.04 | False | 0.667 | She loved outside playing with her toys.... |
| 0.04 | True | 0.671 | She liked to play outside outside to pla... |
| 0.04 | False | 0.633 | She loved to play outside with her toys.... ✓ |
| 0.04 | True | 0.596 | She loved to play outside with her toys.... ✓ |
| 0.04 | False | 0.576 | loved to play outside with her outside w... |
| 0.04 | True | 0.564 | She loved to play outside with her toys.... ✓ |
| 0.05 | False | 0.548 | She liked to play outside in the sun.... |
| 0.05 | True | 0.503 | She was happy to play outside with the g... |
| 0.06 | False | 0.478 | She opened her door. She said, "I'm goin... |
| 0.06 | True | 0.492 | She used to play outside with her toys.... |
| 0.06 | False | 0.505 | She loved playing outdoors with her toys... |
| 0.06 | True | 0.481 | He loved to play outside and play with h... |

## Summary

| Noise Level | Exact Match Rate (no proj) | Exact Match Rate (proj) |
|-------------|---------------------------|------------------------|
| 0.00 | 100% | 100% |
| 0.03 | 40% | 0% |
| 0.04 | 60% | 40% |
| 0.04 | 100% | 40% |
| 0.04 | 20% | 40% |
| 0.05 | 20% | 20% |
| 0.06 | 40% | 0% |
| 0.06 | 20% | 0% |