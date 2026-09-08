"""统一配置：路径、分块、检索与生成参数。

所有默认路径都相对“项目根目录”，入口脚本在解析时用 BASE 拼成绝对路径。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # ---- 路径 ----
    data_dir: str = "data"               # 准备出的数据目录（train/eval/sft）
    raw_dir: str = "data/raw"            # 下载/保留的原始 CSV
    model_dir: str = "models"            # 本地模型目录
    chroma_dir: str = "chroma_data"      # Chroma 向量库目录
    collection: str = "knowledges"       # Chroma 集合名

    # ---- 分块 ----
    chunk_size: int = 150                # 目标块大小（字符）
    chunk_hard_limit: int = 250          # 强制拆分上限（字符）

    # ---- 检索 ----
    top_k: int = 3
    n_results_override: int | None = None

    # ---- 生成 ----
    device: str = "auto"                 # auto / cuda / cpu
    quantize: str = "auto"               # auto / 4bit / 8bit / none
    max_new_tokens: int = 200
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # ---- 数据准备默认 ----
    seed: int = 42

    @property
    def is_short_on(self) -> bool:
        return self.chunk_hard_limit > self.chunk_size