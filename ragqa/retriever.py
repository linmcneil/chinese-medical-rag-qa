"""向量检索封装（对 Chroma 的薄封装，重型依赖延迟到首次查询时导入）。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ragqa.modelstore import ensure_embedding_model, set_hf_mirror


class ChromaRetriever:
    """对 Chroma 集合做 top-k 检索，统一返回带 rid/来源的命中。"""

    def __init__(self, chroma_dir, model_dir=None, collection: str = "knowledges",
                 device: str = "cpu", top_k: int = 3):
        self.chroma_dir = str(chroma_dir)
        self.model_dir = str(model_dir) if model_dir else None
        self.collection = collection
        self.device = device
        self.top_k = top_k
        self._collection = None
        self._collection_meta = None

    # ---- 连接 ----
    def connect(self):
        """建立 Chroma 连接并校验集合存在（不存在时提示先跑 document_processer）。"""
        if self._collection is not None:
            return self
        set_hf_mirror()
        if not self.model_dir:
            raise ValueError("retriever 需要 model_dir（嵌入模型所在目录）")
        embed_path = ensure_embedding_model(Path(self.model_dir))

        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=self.chroma_dir)
        try:
            collection = client.get_collection(
                name=self.collection,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=str(embed_path), device=self.device,
                ),
            )
        except Exception as e:
            raise FileNotFoundError(
                f"向量集合 {self.collection!r} 不存在或无法连接（{e}）。"
                "请先运行 document_processer.py 构建索引。"
            ) from e
        self._collection = collection
        self._collection_meta = {"count": collection.count()}
        return self

    def count(self) -> int:
        self.connect()
        return self._collection.count()

    # ---- 查询 ----
    def query(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        """检索 top-k 条，返回 [{id, document, distance, rid}, ...]。"""
        self.connect()
        k = top_k or self.top_k
        res = self._collection.query(query_texts=[question], n_results=k)
        hits: List[Dict] = []
        ids0 = res.get("ids") or [[]]
        docs0 = res.get("documents") or [[]]
        dist0 = res.get("distances") or [[]]
        meta0 = res.get("metadatas") or [[]]
        for i, doc_id in enumerate(ids0[0]):
            meta = meta0[0][i] if i < len(meta0[0]) else {}
            hits.append({
                "id": doc_id,
                "document": docs0[0][i] if i < len(docs0[0]) else "",
                "distance": dist0[0][i] if i < len(dist0[0]) else None,
                "rid": (meta or {}).get("rid", ""),
            })
        return hits