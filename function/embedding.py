import math
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .metrics import edit_distance


class EmbeddingScorer:
    def __init__(self, model_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> torch.Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded, return_dict=True)
                pooled = self._mean_pooling(output, encoded["attention_mask"])
            embeddings.append(pooled.detach().cpu())
        return torch.cat(embeddings, dim=0)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    @staticmethod
    def cosine_angle(a: torch.Tensor, b: torch.Tensor) -> float:
        cos = EmbeddingScorer.cosine_similarity(a, b)
        cos = max(-1.0, min(1.0, cos))
        return float(math.acos(cos))

    @staticmethod
    def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.norm(a - b, p=2).item())

    @staticmethod
    def l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.norm(a - b, p=1).item())

    def distance_metrics(self, a_text: str, b_text: str) -> Dict[str, float]:
        embs = self.encode_texts([a_text, b_text])
        a = embs[0]
        b = embs[1]
        return {
            "cosine_similarity": self.cosine_similarity(a, b),
            "cosine_angle": self.cosine_angle(a, b),
            "l2": self.l2_distance(a, b),
            "l1": self.l1_distance(a, b),
            "edit_distance": float(edit_distance(a_text, b_text)),
        }

