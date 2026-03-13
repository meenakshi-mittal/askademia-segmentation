#!/usr/bin/env python3
"""
Threshold-based chunking using windowed cosine similarity on VIDEO OCR frames.

Goal: choose a cosine-similarity threshold that yields ~70-80 chunks.

Steps (per lecture):
1) Embed each video OCR frame text.
2) Compute cosine similarity between sliding windows of frames:
   compare frames i..i+k-1 to i+k..i+2k-1, then shift by 1.
3) Choose a threshold so that sim < threshold gives chunk count in [min_chunks, max_chunks].
4) Split into chunks at those boundaries.
5) Construct audio for each chunk by selecting audio sentences closest to the
   chunk start/end window:
     - start = chunk start timestamp (video)
     - end = next chunk start timestamp (video) or last video timestamp
   We include all audio sentences whose timestamps fall within [start, end).
   If none fall in range, we include the single sentence closest to start.

Outputs a list of chunk objects with `audio` and `video` keys, mirroring
the audio_video_sentence_level style.
"""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def _extract_ms(entry: Dict[str, Any]) -> Tuple[int, int]:
    if "start_ms" in entry and "end_ms" in entry:
        try:
            return int(entry["start_ms"]), int(entry["end_ms"])
        except Exception:
            pass
    ts_s = float(entry.get("timestamp", 0.0) or 0.0)
    start_ms = int(round(ts_s * 1000.0))
    dur_ms = entry.get("duration_ms")
    if dur_ms is not None:
        try:
            dur_ms_int = int(dur_ms)
            if dur_ms_int > 0:
                return start_ms, start_ms + dur_ms_int
        except Exception:
            pass
    return start_ms, start_ms + 1000


def _infer_lectures(audio_dir: Path, video_dir: Path) -> List[str]:
    audio_ids = {
        p.stem.replace("audio_sentence_level", "")
        for p in audio_dir.glob("audio_sentence_level*.json")
    }
    video_ids = {p.stem.replace("video", "") for p in video_dir.glob("video*.json")}
    ids = sorted(audio_ids & video_ids, key=lambda x: int(x) if x.isdigit() else x)
    return [i for i in ids if i]


def _load_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path)


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
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")
        return OpenAI(api_key=openai_key)

    if provider.lower() == "azure":
        if not azure_endpoint or not api_version:
            raise RuntimeError("For provider=azure, set --azure-endpoint/--api-version or env vars.")
        if not azure_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY or OPENAI_API_KEY is required for Azure.")
        return AzureOpenAI(
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=azure_key,
            api_version=api_version,
        )

    api_key = openai_key or azure_key
    if azure_endpoint:
        if not api_version:
            raise RuntimeError("For Azure OpenAI, set --api-version or AZURE_OPENAI_API_VERSION/api_version in env.")
        return AzureOpenAI(
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=api_key,
            api_version=api_version,
        )
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def _batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _compute_embeddings(client: OpenAI, texts: List[str], *, model: str, batch_size: int) -> np.ndarray:
    out: List[List[float]] = []
    for batch in _batched(texts, batch_size=max(1, int(batch_size))):
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    val = float(np.dot(a, b) / denom)
    if val > 1.0:
        return 1.0
    if val < -1.0:
        return -1.0
    return val


def _choose_threshold(
    sims: List[float],
    *,
    min_chunks: int,
    max_chunks: int,
    target_chunks: Optional[int],
) -> Tuple[float, int]:
    if not sims:
        return 1.0, 1

    n = len(sims) + 1
    min_chunks = max(1, int(min_chunks))
    max_chunks = min(n, int(max_chunks))
    if min_chunks > max_chunks:
        min_chunks, max_chunks = max_chunks, min_chunks

    target = int(target_chunks) if target_chunks is not None else (min_chunks + max_chunks) // 2
    target = max(min_chunks, min(max_chunks, target))

    sims_sorted = sorted(sims)

    best_k = None
    best_gap = None
    for k in range(min_chunks, max_chunks + 1):
        gap = abs(k - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_k = k

    k = best_k if best_k is not None else target
    m = max(0, k - 1)
    if m == 0:
        threshold = min(sims_sorted) - 1e-6
    elif m >= len(sims_sorted):
        threshold = max(sims_sorted) + 1e-6
    else:
        lo = sims_sorted[m - 1]
        hi = sims_sorted[m]
        threshold = (lo + hi) / 2.0
    return float(threshold), int(k)


def _build_chunks_from_threshold(
    frames: List[Dict[str, Any]],
    boundary_sims: List[float],
    threshold: float,
) -> List[List[Dict[str, Any]]]:
    if not frames:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for i, f in enumerate(frames):
        cur.append(f)
        if i < len(boundary_sims) and boundary_sims[i] < threshold:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _build_audio_for_span(
    audio_sents: List[Dict[str, Any]],
    audio_ts: List[float],
    *,
    start_s: float,
    end_s: float,
) -> List[Dict[str, Any]]:
    """
    Return sentence list whose timestamps fall within [start_s, end_s).
    If none, return the single sentence closest to start_s.
    """
    if not audio_sents:
        return []

    lo = bisect_left(audio_ts, start_s)
    hi = bisect_left(audio_ts, end_s)
    selected = audio_sents[lo:hi]
    if selected:
        return selected

    # fallback: closest to start_s
    idx = max(0, min(len(audio_ts) - 1, lo))
    if idx > 0 and abs(audio_ts[idx - 1] - start_s) < abs(audio_ts[idx] - start_s):
        idx = idx - 1
    return [audio_sents[idx]]


def _chunks_to_output(
    video_chunks: List[List[Dict[str, Any]]],
    audio_sents: List[Dict[str, Any]],
    audio_ts: List[float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, v_list in enumerate(video_chunks):
        if not v_list:
            continue
        start_s = float(v_list[0].get("timestamp", 0.0) or 0.0)
        if idx + 1 < len(video_chunks) and video_chunks[idx + 1]:
            end_s = float(video_chunks[idx + 1][0].get("timestamp", start_s) or start_s)
        else:
            end_s = float(v_list[-1].get("timestamp", start_s) or start_s)

        audio_sel = _build_audio_for_span(audio_sents, audio_ts, start_s=start_s, end_s=end_s)
        start_ms = int(round(start_s * 1000.0))
        end_ms = int(round(end_s * 1000.0))

        audio_text = " ".join(str(x.get("text", "")).strip() for x in audio_sel if str(x.get("text", "")).strip()).strip()
        video_texts = [str(x.get("text", "")).strip() for x in v_list if str(x.get("text", "")).strip()]

        out.append(
            {
                "chunk_index": idx,
                "timestamp": start_s,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(0, end_ms - start_ms),
                "text": audio_text,
                "audio": audio_text,
                "video": video_texts,
                "sentence_count": len(audio_sel),
                "word_count": len(audio_text.split()) if audio_text else 0,
                "sentences": audio_sel,
            }
        )
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    data_dir = paper_dir / "data"
    default_env = repo_root / "config" / "keys.env"

    parser = argparse.ArgumentParser(
        description="Threshold-based chunking using windowed cosine similarity on VIDEO OCR (paper folder)."
    )
    parser.add_argument("--audio-dir", type=str, default=str(data_dir / "audio_sentence_level"))
    parser.add_argument("--video-dir", type=str, default=str(data_dir / "video_ocr"))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--lectures", type=str, default="", help="Comma-separated lecture ids (default: infer).")
    parser.add_argument("--min-chunks", type=int, default=70)
    parser.add_argument("--max-chunks", type=int, default=80)
    parser.add_argument("--target-chunks", type=int, default=0, help="Optional target within [min,max].")
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Window size k for comparing [i..i+k-1] vs [i+k..i+2k-1] (default 3).",
    )
    parser.add_argument("--env", type=str, default=str(default_env))
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--embedding-endpoint", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = data_dir / f"cosine_threshold_video_70_80_w{int(args.window_size)}"

    if not audio_dir.exists():
        raise FileNotFoundError(f"audio-dir not found: {audio_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"video-dir not found: {video_dir}")

    if args.lectures.strip():
        lectures = [t.strip() for t in args.lectures.split(",") if t.strip()]
    else:
        lectures = _infer_lectures(audio_dir, video_dir)
    if not lectures:
        raise RuntimeError("No lectures to process.")

    _load_env(Path(args.env))
    emb_client = _build_embedding_client(
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        embedding_endpoint=args.embedding_endpoint,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"🎯 video cosine threshold chunking: lectures={len(lectures)} range=[{args.min_chunks},{args.max_chunks}]")

    for lec in lectures:
        audio_path = audio_dir / f"audio_sentence_level{lec}.json"
        video_path = video_dir / f"video{lec}.json"
        if not audio_path.exists():
            print(f"❌ Skipping lecture {lec}: missing {audio_path}")
            continue
        if not video_path.exists():
            print(f"❌ Skipping lecture {lec}: missing {video_path}")
            continue

        print(f"\n🔄 Lecture {lec}: embedding video frames")
        audio_sents = _load_json_list(audio_path)
        video_frames = _load_json_list(video_path)
        if not video_frames:
            print("   ⚠️  No video frames found")
            continue

        # Normalize audio sentences
        audio_norm: List[Dict[str, Any]] = []
        audio_ts: List[float] = []
        for i, s in enumerate(audio_sents):
            start_ms, end_ms = _extract_ms(s)
            text = str(s.get("text", "") or "").strip()
            sent = dict(s)
            sent["sentence_index"] = int(s.get("sentence_index", i))
            sent["start_ms"] = start_ms
            sent["end_ms"] = end_ms
            sent["audio"] = str(s.get("audio", text) or "").strip()
            audio_norm.append(sent)
            audio_ts.append(float(sent.get("timestamp", start_ms / 1000.0)))

        # Normalize video frames + texts
        frames_sorted = sorted(video_frames, key=lambda f: float(f.get("timestamp", 0.0) or 0.0))
        v_texts: List[str] = []
        for f in frames_sorted:
            t = str(f.get("text", "") or "").strip()
            v_texts.append(t if t else " ")

        v_embs = _compute_embeddings(emb_client, v_texts, model=args.model, batch_size=int(args.batch_size))
        k = max(1, int(args.window_size))
        n = len(v_texts)
        boundary_sims: List[float] = [1.0] * max(0, n - 1)
        if n >= 2 * k:
            for i in range(0, n - 2 * k + 1):
                left = v_embs[i : i + k]
                right = v_embs[i + k : i + 2 * k]
                left_mean = np.mean(left, axis=0)
                right_mean = np.mean(right, axis=0)
                sim = _cosine(left_mean, right_mean)
                boundary_idx = i + k - 1
                boundary_sims[boundary_idx] = sim

        threshold, k_est = _choose_threshold(
            boundary_sims,
            min_chunks=int(args.min_chunks),
            max_chunks=int(args.max_chunks),
            target_chunks=(int(args.target_chunks) if int(args.target_chunks) > 0 else None),
        )

        video_chunks = _build_chunks_from_threshold(frames_sorted, boundary_sims, threshold)
        print(
            f"   ✅ window={k} threshold={threshold:.4f} -> chunks={len(video_chunks)} (target~{k_est})"
        )

        out_chunks = _chunks_to_output(video_chunks, audio_norm, audio_ts)

        out_path = out_dir / f"audio_video_cosine_threshold_video_{int(args.min_chunks)}_{int(args.max_chunks)}{lec}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_chunks, f, ensure_ascii=False, indent=2)
        print(f"   💾 wrote {out_path}")


if __name__ == "__main__":
    main()

