"""可选 cross-encoder 重排：在向量 top-N 基础上做二次精排。

用途：bge-small-zh 的向量召回在“同一话题、不同问法”时不够敏锐，
可以先用 Chroma 取较多候选（如 12~20 条），再用 bge-reranker 精排到
最终 top-k，trade-off 是每次多一次 cross-encoder 推理（GPU 上毫秒级）。

设计：rerank_hits 是纯函数（可单测）；CrossEncoderReranker 延迟加载
模型，未安装 sentence-transformers 或未启用时不影响其它功能。
"""
from __future__ import annotations

from typing import Dict, List, Optional


def rerank_hits(hits: List[Dict], scores: List[float], top_k: Optional[int] = None) -> List[Dict]:
    """按分数降序稳定重排 hits，并附上 rerank_score；可截断到 top_k。

    hits: ChromaRetriever.query 的返回列表 [{id, document, distance, rid}, ...]
    scores: 与 hits 等长的相关性分数（cross-encoder logits 或相似度）
    """
    if not hits:
        return []
    if len(scores) != len(hits):
        raise ValueError(f"scores 长度 {len(scores)} 与 hits {len(hits)} 不一致")
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    ranked = [dict(hits[i], rerank_score=float(scores[i])) for i in order]
    if top_k is None:
        return ranked
    return ranked[:max(0, int(top_k))]


class CrossEncoderReranker:
    """bge-reranker 封装：模型延迟加载，只对候选片段做 pair 打分。"""

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu",
                 top_k: int = 3):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model = None

    def _load(self):
        """首次使用时才加载模型（约 1GB，视网络而定）。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, question: str, hits: List[Dict]) -> List[Dict]:
        """对 hits 逐条算 (question, document) 相关性并精排到 self.top_k。"""
        if not hits:
            return []
        model = self._load()
        pairs = [(question, h.get("document") or "") for h in hits]
        scores = model.predict(pairs)
        return rerank_hits(hits, list(scores), top_k=self.top_k)

    def __repr__(self) -> str:
        return (f"CrossEncoderReranker(model={self.model_name!r}, "
                f"device={self.device!r}, top_k={self.top_k})")