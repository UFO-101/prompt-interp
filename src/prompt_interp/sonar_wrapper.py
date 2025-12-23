"""SONAR encoder/decoder wrapper for sentence embeddings."""

import torch
import nltk

# Download punkt tokenizer if not present
for resource in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)


class SonarWrapper:
    """Wrapper for SONAR encoder/decoder with utilities."""

    def __init__(self, device: str | torch.device = "cuda"):
        from sonar.inference_pipelines.text import (
            TextToEmbeddingModelPipeline,
            EmbeddingToTextModelPipeline,
        )

        self.device = torch.device(device) if isinstance(device, str) else device
        self.encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )
        self.decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_decoder",
            device=self.device,
        )
        self._eos_embedding: torch.Tensor | None = None

    @property
    def eos_embedding(self) -> torch.Tensor:
        """Cached end-of-sequence embedding."""
        if self._eos_embedding is None:
            self._eos_embedding = self.encode(["End of sequence."])[0]
        return self._eos_embedding

    def encode(self, sentences: list[str], lang: str = "eng_Latn") -> torch.Tensor:
        """Encode sentences to SONAR embeddings. Returns (N, 1024) tensor."""
        emb = self.encoder.predict(sentences, source_lang=lang)
        return emb.to(self.device).clone()  # Clone escapes inference mode

    def decode(self, embeddings: torch.Tensor, lang: str = "eng_Latn") -> list[str]:
        """Decode SONAR embeddings to text."""
        return self.decoder.predict(embeddings, target_lang=lang, max_seq_len=256)

    def segment(self, text: str) -> list[str]:
        """Segment text into sentences."""
        return nltk.sent_tokenize(text)

    def encode_text(self, text: str, lang: str = "eng_Latn") -> tuple[list[str], torch.Tensor]:
        """Segment and encode text. Returns (sentences, embeddings)."""
        sents = self.segment(text)
        return sents, self.encode(sents, lang)
