"""Download every model once, so processing can run with HF_HUB_OFFLINE=1 (no network at all).

    python -m app.download [whisper-model ...]
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the one place allowed to go online

from . import db  # noqa: F401  (loads .env)
from . import ml


def main() -> None:
    from huggingface_hub import snapshot_download

    for m in sys.argv[1:] or ["large-v3-turbo"]:
        print("whisper", m, "->", snapshot_download(ml.MLX_REPOS[m]) if ml.USE_MLX else m)
    token = os.environ.get("HF_TOKEN") or None
    print("speaker embeddings ->", snapshot_download(ml.SPEAKER_EMBEDDING, token=token))
    try:
        print("diarization pipeline ->", snapshot_download(ml.DIARIZATION_MODEL, token=token))
    except Exception as e:  # gated: fine, the clustering fallback works without it
        print("diarization pipeline: unavailable, clustering fallback will be used:", type(e).__name__)
    print("text embeddings ->", snapshot_download(ml.EMBED_MODEL))
    print("LLM: run `ollama pull qwen3:8b`")


if __name__ == "__main__":
    main()
