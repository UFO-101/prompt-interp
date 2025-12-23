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
| 0.01 | False | 0.955 | Once upon a time, there was a little gir... ✓ |
| 0.01 | True | 0.950 | Once upon a time, there was a little gir... ✓ |
| 0.02 | False | 0.843 | Once upon a time, there was a girl named... |
| 0.02 | True | 0.841 | Once upon a time, there was a little gir... ✓ |
| 0.05 | False | 0.504 | Once upon a time, there was a little gir... ✓ |
| 0.05 | True | 0.514 | Once upon a time, there was a little gir... ✓ |
| 0.10 | False | 0.279 | Once upon a time, there was a woman, who... |
| 0.10 | True | 0.327 | "Once upon a time, there was a little gi... |
| 0.20 | False | 0.134 | along along along along along along alon... |
| 0.20 | True | 0.200 | A few years ago I had a dream that I wou... |
| 0.50 | False | 0.080 | [ccccccccccccccccccccccccccccccccccccccc... |
| 0.50 | True | 0.082 | Solomon is the author of the book of Pro... |
| 1.00 | False | 0.004 | . . . . . . . . . . . . . . . . . . . . ... |
| 1.00 | True | 0.038 | Compatibility: Compatibility can be meas... |

### "The weather is nice today."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | The weather is nice today.... ✓ |
| 0.00 | True | 1.000 | The weather is nice today.... ✓ |
| 0.01 | False | 0.949 | The weather is nice today.... ✓ |
| 0.01 | True | 0.952 | The weather is nice today.... ✓ |
| 0.02 | False | 0.843 | It's nice weather today.... |
| 0.02 | True | 0.828 | The weather is nice today.... ✓ |
| 0.05 | False | 0.567 | Good weather today.... |
| 0.05 | True | 0.535 | The weather is nice today.... ✓ |
| 0.10 | False | 0.319 | YATILITY MARY MARY MARY MARY MARY MARY I... |
| 0.10 | True | 0.266 | The popularity of the park will be shown... |
| 0.20 | False | 0.207 | All of that in this way about in that th... |
| 0.20 | True | 0.122 | It's the first time I've ever seen a wom... |
| 0.50 | False | 0.064 | great great great great great great grea... |
| 0.50 | True | 0.076 | Row of wait for The Roman explanation is... |
| 1.00 | False | 0.043 | Out Out Out Out Out Out Out Out... |
| 1.00 | True | -0.025 | It's about the book, and it's about the ... |

### "I like cheese."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | I like cheese.... ✓ |
| 0.00 | True | 1.000 | I like cheese.... ✓ |
| 0.01 | False | 0.954 | I like cheese.... ✓ |
| 0.01 | True | 0.950 | I like cheese.... ✓ |
| 0.02 | False | 0.830 | I like cheese.... ✓ |
| 0.02 | True | 0.835 | I like cheese.... ✓ |
| 0.05 | False | 0.534 | I like the cheese .... |
| 0.05 | True | 0.508 | I'm like, a little cheese and wheat. Che... |
| 0.10 | False | 0.288 | "You know, I'm in the U.S.A".... |
| 0.10 | True | 0.307 | I believe I know the answer. - I believe... |
| 0.20 | False | 0.160 | if if if if if like like like and and an... |
| 0.20 | True | 0.142 | He will try the greatest, and is more of... |
| 0.50 | False | 0.085 | ?? Bu Bu Bu Bu Bu Bu Bu Bu Bu Bu Bu Bu B... |
| 0.50 | True | 0.075 | In the meantime, I'm going to share with... |
| 1.00 | False | 0.067 | janijanijanijanijanijanijanijanijanijani... |
| 1.00 | True | -0.006 | It has reached the end of its useful lif... |

### "He decided to go on an adventure."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | He decided to go on an adventure.... ✓ |
| 0.00 | True | 1.000 | He decided to go on an adventure.... ✓ |
| 0.01 | False | 0.956 | He decided to go on an adventure.... ✓ |
| 0.01 | True | 0.954 | He decided to go on an adventure.... ✓ |
| 0.02 | False | 0.840 | He decided to go on an adventure.... ✓ |
| 0.02 | True | 0.835 | He decided to go on an adventure.... ✓ |
| 0.05 | False | 0.539 | He's decided to go on an adventure.... |
| 0.05 | True | 0.533 | He appointed the decisionofFare went on ... |
| 0.10 | False | 0.312 | - - - - - - - - -... |
| 0.10 | True | 0.338 | The problem is, the problem is, the prob... |
| 0.20 | False | 0.159 | , an,,, an, an, an, an, an, an., an, he ... |
| 0.20 | True | 0.163 | He is referring to the full-fledged proj... |
| 0.50 | False | 0.059 | netnetnetnetnet Ce Ce Ce Ce Ce Ce CeCEAM... |
| 0.50 | True | 0.053 | I'd like to know if there's anything I c... |
| 1.00 | False | 0.019 | neither neither neither neither neither ... |
| 1.00 | True | 0.055 | In the meantime, one of the most popular... |

### "She loved to play outside with her toys."

| Noise | Project | Cos Sim | Decoded |
|-------|---------|---------|---------|
| 0.00 | False | 1.000 | She loved to play outside with her toys.... ✓ |
| 0.00 | True | 1.000 | She loved to play outside with her toys.... ✓ |
| 0.01 | False | 0.955 | She loved to play outside with her toys.... ✓ |
| 0.01 | True | 0.953 | She loved to play outside with her toys.... ✓ |
| 0.02 | False | 0.843 | She loved to play out with her toys.... |
| 0.02 | True | 0.844 | She liked to play outside with her toys.... |
| 0.05 | False | 0.526 | With her beautiful toys.... |
| 0.05 | True | 0.545 | She loved to have open-air play inside t... |
| 0.10 | False | 0.212 | She is a member of the board of trustees... |
| 0.10 | True | 0.293 | She likes the Christian congregation sta... |
| 0.20 | False | 0.140 | were also sought to take part. were also... |
| 0.20 | True | 0.155 | It's been a long time since I've heard a... |
| 0.50 | False | 0.104 | . . . . . . . . . . . . . . . . . . . . ... |
| 0.50 | True | 0.061 | In the meantime, he's got a lot of money... |
| 1.00 | False | 0.023 | Chin Chin Chin Chin Chin Chin Chin Chin ... |
| 1.00 | True | -0.002 | The king's wife, however, did not return... |

## Summary

| Noise Level | Exact Match Rate (no proj) | Exact Match Rate (proj) |
|-------------|---------------------------|------------------------|
| 0.00 | 100% | 100% |
| 0.01 | 100% | 100% |
| 0.02 | 40% | 80% |
| 0.05 | 20% | 40% |
| 0.10 | 0% | 0% |
| 0.20 | 0% | 0% |
| 0.50 | 0% | 0% |
| 1.00 | 0% | 0% |