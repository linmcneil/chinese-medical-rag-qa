"""模型本地化与下载（统一入口，延迟触发，避免 import 即下载 15GB）。

- 嵌入模型：Xorbits/bge-small-zh-v1.5（约 100MB）
- 医疗问答模型：BAAI/CareBot_Medical_multi-llama3-8b-instruct（约 15GB，谨慎触发）
下载默认走 ModelScope；transformers/huggingface 使用前记得先 set_hf_mirror()。
"""
from __future__ import annotations

import os
from pathlib import Path

EMBEDDING_MODEL_ID = "BAAI/bge-small-zh-v1.5"
EMBEDDING_FOLDER = "bge-small-zh-v1.5"

QA_MODEL_ID = "BAAI/CareBot_Medical_multi-llama3-8b-instruct"
QA_FOLDER = "CareBot_Medical"


def set_hf_mirror() -> None:
    """按需让 transformers/huggingface_hub 走国内镜像。

    默认不强制设置：镜像站点不可达/回跳时会让下载失败。
    国内网络下可用环境变量 RAG_HF_MIRROR=1 启用。
    """
    if os.environ.get("RAG_HF_MIRROR", "0") == "1":
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _model_ready(path: Path) -> bool:
    if not path.exists():
        return False
    # 目录非空且包含模型文件即可（modelscope 会写入 config.json / pytorch_model* 等）
    try:
        return any(path.iterdir())
    except OSError:
        return False


def ensure_local_model(model_dir, folder: str, model_id: str) -> Path:
    """确保模型在本地，缺失时自动下载。

    优先用 ModelScope（国内快）；未安装或失败时回退 HuggingFace 镜像。
    """
    target = Path(model_dir) / folder
    if _model_ready(target):
        return target
    set_hf_mirror()
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    def _download_hf():
        from huggingface_hub import snapshot_download

        def _try():
            print(f"[HF] 下载 {model_id} -> {target} ...")
            snapshot_download(repo_id=model_id, local_dir=str(target))
            return target

        try:
            return _try()
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("HF_ENDPOINT"):
                # 镜像异常时去掉镜像再试一次（可能是 308 回跳/401）
                print(f"[HF] 镜像下载失败（{exc}），改用官方源重试 ...")
                os.environ.pop("HF_ENDPOINT", None)
                return _try()
            raise

    try:
        from modelscope import snapshot_download as ms_download
        print(f"[ModelScope] 下载 {model_id} -> {target} ...")
        ms_download(model_id=model_id, local_dir=str(target))
        return target
    except ImportError:
        return _download_hf()
    except Exception as exc:  # noqa: BLE001（网络/鉴权等问题时回退 HF）
        print(f"[ModelScope] 下载失败（{exc}），回退 HuggingFace ...")
        return _download_hf()


def ensure_embedding_model(model_dir) -> Path:
    return ensure_local_model(model_dir, EMBEDDING_FOLDER, EMBEDDING_MODEL_ID)


def ensure_qa_model(model_dir) -> Path:
    return ensure_local_model(model_dir, QA_FOLDER, QA_MODEL_ID)