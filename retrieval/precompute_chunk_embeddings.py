#!/usr/bin/env python3
"""
Precompute embeddings for retrieval methods inside askademia segmentation.

Outputs per method + lecture:
  - <out_dir>/<method>/lecture{N}.emb.npy
  - <out_dir>/<method>/lecture{N}.texts.json
  - <out_dir>/<method>/lecture{N}.meta.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


def _load_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path)


def _infer_deployment_from_endpoint(endpoint: Optional[str]) -> Optional[str]:
    ep = (endpoint or "").strip()
    marker = "/deployments/"
    if not ep or marker not in ep:
        return None
    try:
        return ep.split(marker, 1)[1].split("/", 1)[0].strip() or None
    except Exception:
        return None


def _build_embedding_client(
    *,
    provider: str,
    azure_endpoint: Optional[str],
    api_version: Optional[str],
    embedding_endpoint: Optional[str],
) -> OpenAI:
    env_azure_endpoint = (
        os.getenv("EMBEDDING_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("azure_endpoint")
    )
    env_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("api_version")
    azure_endpoint = embedding_endpoint or azure_endpoint or env_azure_endpoint
    api_version = api_version or env_api_version

    openai_key = os.getenv("OPENAI_API_KEY")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or openai_key

    if provider.lower() == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        return OpenAI(api_key=openai_key)

    if provider.lower() == "azure":
        if not azure_endpoint or not api_version:
            raise RuntimeError("For provider=azure set endpoint + api_version")
        if not azure_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY or OPENAI_API_KEY required")
        return AzureOpenAI(
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=azure_key,
            api_version=api_version,
        )

    api_key = openai_key or azure_key
    if azure_endpoint:
        if not api_version:
            raise RuntimeError("For Azure OpenAI set api_version")
        return AzureOpenAI(
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=api_key,
            api_version=api_version,
        )
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _compute_embeddings(client: OpenAI, texts: List[str], *, model: str, batch_size: int) -> np.ndarray:
    out: List[List[float]] = []
    for batch in _batched(texts, max(1, int(batch_size))):
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)


def _truncate(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    return t if max_chars <= 0 or len(t) <= max_chars else t[:max_chars]


def _method_folders(data_dir: Path) -> Dict[str, Path]:
    skip = {"audio_sentence_level", "video_ocr", "audio_video_sentence_level", "hashes"}
    out: Dict[str, Path] = {}
    for p in sorted([d for d in data_dir.iterdir() if d.is_dir()], key=lambda d: d.name):
        if p.name in skip:
            continue
        out[p.name] = p
    return out


def _discover_lectures_for_method(folder: Path) -> List[str]:
    ids: List[str] = []
    for p in sorted(folder.glob("lecture*.json")):
        lec = p.stem.replace("lecture", "")
        if lec.isdigit():
            ids.append(lec)
    return sorted(set(ids), key=lambda x: int(x))


def _load_chunk_texts(path: Path, *, max_chars: int) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    texts: List[str] = []
    for ch in data:
        if not isinstance(ch, dict):
            continue
        txt = str(ch.get("text", "") or "").strip()
        if txt:
            texts.append(_truncate(txt, max_chars=max_chars))
    return texts


def _load_tree_texts(path: Path, *, max_chars: int) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    roots = data.get("roots")
    if not isinstance(roots, list):
        return []

    texts: List[str] = []

    def visit(node: Dict[str, Any]) -> None:
        title = str(node.get("title", "") or "").strip()
        summary = str(node.get("summary", "") or "").strip()
        parts = [p for p in [title, summary] if p]
        if parts:
            texts.append(_truncate("\n\n".join(parts), max_chars=max_chars))
        leaf = node.get("leaf_chunk")
        if isinstance(leaf, dict):
            lt = str(leaf.get("text", "") or "").strip()
            if lt:
                texts.append(_truncate(lt, max_chars=max_chars))
        for child in node.get("children") or []:
            if isinstance(child, dict):
                visit(child)

    for r in roots:
        if isinstance(r, dict):
            visit(r)
    return texts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    seg_dir = repo_root / "askademia segmentation"
    default_env = repo_root / "config" / "keys.env"

    parser = argparse.ArgumentParser(description="Precompute embeddings for askademia segmentation retrieval.")
    parser.add_argument("--data-dir", type=str, default=str(seg_dir / "data"))
    parser.add_argument("--out-dir", type=str, default=str(seg_dir / ".cache" / "precomputed_embeddings"))
    parser.add_argument("--env", type=str, default=str(default_env))
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--embedding-endpoint", type=str, default="")
    parser.add_argument("--embedding-endpoint-env", type=str, default="EMBEDDING_ENDPOINT")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-embed-chars", type=int, default=12000)
    parser.add_argument("--methods", type=str, default="", help="Comma-separated method folder names.")
    args = parser.parse_args()

    _load_env(Path(args.env))
    embedding_endpoint = (args.embedding_endpoint or "").strip()
    if not embedding_endpoint:
        embedding_endpoint = (os.getenv(args.embedding_endpoint_env) or "").strip()
    emb_model = (args.model or "").strip() or _infer_deployment_from_endpoint(embedding_endpoint) or "text-embedding-3-small"

    client = _build_embedding_client(
        provider=args.provider,
        azure_endpoint=None,
        api_version=None,
        embedding_endpoint=embedding_endpoint or None,
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = _method_folders(data_dir)

    include = {m.strip() for m in args.methods.split(",") if m.strip()}
    if include:
        methods = {k: v for k, v in methods.items() if k in include}

    if not methods:
        raise RuntimeError(f"No retrieval method folders found under {data_dir}")

    print(f"Embedding model: {emb_model}")
    for method, folder in methods.items():
        lec_ids = _discover_lectures_for_method(folder)
        if not lec_ids:
            continue
        method_out = out_dir / method
        method_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{method}: {len(lec_ids)} lecture(s)")

        is_tree_method = "tree" in method.lower()
        for lec in lec_ids:
            emb_path = method_out / f"lecture{lec}.emb.npy"
            text_path = method_out / f"lecture{lec}.texts.json"
            meta_path = method_out / f"lecture{lec}.meta.json"
            if emb_path.exists() and text_path.exists() and meta_path.exists():
                print(f"  lecture{lec}: exists, skipping")
                continue

            src = folder / f"lecture{lec}.json"
            if not src.exists():
                print(f"  lecture{lec}: missing source")
                continue

            texts = _load_tree_texts(src, max_chars=int(args.max_embed_chars)) if is_tree_method else _load_chunk_texts(src, max_chars=int(args.max_embed_chars))
            if not texts:
                print(f"  lecture{lec}: no texts")
                continue

            embs = _compute_embeddings(client, texts, model=emb_model, batch_size=int(args.batch_size))
            np.save(emb_path, embs)
            text_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
            meta = {
                "method": method,
                "lecture_id": lec,
                "embedding_model": emb_model,
                "max_embed_chars": int(args.max_embed_chars),
                "count": len(texts),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  lecture{lec}: saved {len(texts)} embeddings")


if __name__ == "__main__":
    main()

