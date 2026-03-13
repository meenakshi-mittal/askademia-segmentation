#!/usr/bin/env python3
"""
Adjacent-agglomerative chunking experiments (paper folder).

Two modes:

1) agglomerative_audio
   - Start from `paper/data/audio_sentence_level/audio_sentence_level{N}.json`
   - Each sentence is an initial chunk.
   - Repeatedly merge the MOST SIMILAR adjacent pair (cosine over embeddings)
     until reaching --target-chunks (default 75).
   - After chunking, attach video OCR frames from `paper/data/video_ocr/video{N}.json`
     using the same rule as elsewhere:
       - last frame strictly before chunk start
       - plus frames within [start, end]
   - Output chunks store `audio` and `video` separately (video as list[str]).

2) agglomerative_av
   - Start from `paper/data/audio_video_sentence_level/audio_video_sentence_level{N}.json`
     where each sentence already has `audio` and `video` (list[str]).
   - Similarity for merging is computed over a CONCATENATED text representation:
       audio + "\n\n" + joined video frames
     BUT the stored chunk keeps `audio` and `video` separate.
   - When two chunks merge, we concatenate their video lists and REMOVE duplicates
     (order-preserving) so the merged chunk doesn't repeat frames both children had.
   - Output chunks store `audio` and `video` separately.

Notes:
- Embeddings are computed once for initial units; merged chunk embeddings are updated
  via a size-weighted average of child embeddings (by sentence_count). This avoids
  re-embedding after every merge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI


# ------------------------- IO helpers -------------------------


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


def _infer_lectures(audio_dir: Path, video_dir: Optional[Path]) -> List[str]:
    audio_ids = set()
    # legacy names: audio_sentence_level{N}.json or audio_video_sentence_level{N}.json
    for p in audio_dir.glob("audio*sentence_level*.json"):
        audio_ids.add(re.sub(r"^audio_(video_)?sentence_level", "", p.stem))
    # renamed names: lecture{N}.json
    for p in audio_dir.glob("lecture*.json"):
        audio_ids.add(p.stem.replace("lecture", ""))
    audio_ids = {x for x in audio_ids if x}
    if video_dir is None:
        return sorted(audio_ids, key=lambda x: int(x) if x.isdigit() else x)
    video_ids = set()
    for p in video_dir.glob("video*.json"):
        video_ids.add(p.stem.replace("video", ""))
    for p in video_dir.glob("lecture*.json"):
        video_ids.add(p.stem.replace("lecture", ""))
    ids = sorted(audio_ids & video_ids, key=lambda x: int(x) if x.isdigit() else x)
    return [i for i in ids if i]


# ------------------------- Embeddings -------------------------


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

    # auto
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
    # clamp
    if val > 1.0:
        return 1.0
    if val < -1.0:
        return -1.0
    return val


# ------------------------- Video OCR selection (audio mode) -------------------------


def _prepare_video_frames(video_frames: List[Dict[str, Any]]) -> Tuple[List[float], List[Dict[str, Any]]]:
    frames = sorted(video_frames, key=lambda f: float(f.get("timestamp", 0.0) or 0.0))
    ts = [float(f.get("timestamp", 0.0) or 0.0) for f in frames]
    return ts, frames


def _select_video_texts_for_span(
    *,
    video_ts: List[float],
    video_frames_sorted: List[Dict[str, Any]],
    start_s: float,
    end_s: float,
) -> List[str]:
    if not video_frames_sorted or end_s < start_s:
        return []
    out_frames: List[Dict[str, Any]] = []
    prev_idx = bisect_left(video_ts, start_s) - 1
    if prev_idx >= 0:
        out_frames.append(video_frames_sorted[prev_idx])
    lo = bisect_left(video_ts, start_s)
    hi = bisect_right(video_ts, end_s)
    out_frames.extend(video_frames_sorted[lo:hi])
    texts: List[str] = []
    for fr in out_frames:
        t = str(fr.get("text", "") or "").strip()
        if t:
            texts.append(t)
    return texts


# ------------------------- Chunk state + merging -------------------------


def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _format_audio_video_text(audio: str, video: List[str]) -> str:
    audio = (audio or "").strip()
    if not video:
        return f"Audio: {audio}"
    numbered = ", ".join([f"{i + 1}: {v}" for i, v in enumerate(video)])
    return f"Audio: {audio} Video: {numbered}"


@dataclass
class Chunk:
    start_ms: int
    end_ms: int
    sentence_count: int
    audio: str
    video: List[str]  # list[str], may be empty in audio-mode until OCR attach
    sentences: List[Dict[str, Any]]  # underlying sentence records (for traceability)
    emb: np.ndarray  # embedding used for similarity


def _chunk_text_for_embedding_audio(audio: str) -> str:
    return (audio or "").strip() or " "


def _chunk_text_for_embedding_av(audio: str, video: List[str], *, max_video_chars: int = 4000) -> str:
    """
    Build concatenated text used only for embedding/similarity in AV mode.
    We keep video bounded to avoid gigantic inputs.
    """
    a = (audio or "").strip()
    v = "\n\n".join((s or "").strip() for s in (video or []) if (s or "").strip()).strip()
    if max_video_chars > 0 and len(v) > max_video_chars:
        v = v[:max_video_chars] + " ..."
    if a and v:
        return a + "\n\n" + v
    return a or v or " "


def _initial_chunks_audio(
    sentences: List[Dict[str, Any]],
    *,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> List[Chunk]:
    # normalize + build texts
    texts: List[str] = []
    sents_norm: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences):
        start_ms, end_ms = _extract_ms(s)
        text = str(s.get("text", "") or "").strip()
        sent = dict(s)
        sent["sentence_index"] = int(s.get("sentence_index", i))
        sent["start_ms"] = start_ms
        sent["end_ms"] = end_ms
        sent["audio"] = str(s.get("audio", text) or "").strip()
        sents_norm.append(sent)
        texts.append(_chunk_text_for_embedding_audio(text))

    embs = _compute_embeddings(emb_client, texts, model=emb_model, batch_size=emb_batch_size)
    chunks: List[Chunk] = []
    for sent, emb in zip(sents_norm, embs):
        start_ms = int(sent["start_ms"])
        end_ms = int(sent["end_ms"])
        audio = str(sent.get("text", "") or "").strip()
        chunks.append(
            Chunk(
                start_ms=start_ms,
                end_ms=end_ms,
                sentence_count=1,
                audio=audio,
                video=[],
                sentences=[sent],
                emb=np.asarray(emb, dtype=np.float32),
            )
        )
    return chunks


def _initial_chunks_av(
    sentences_av: List[Dict[str, Any]],
    *,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
    max_video_chars: int,
) -> List[Chunk]:
    texts: List[str] = []
    sents_norm: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences_av):
        start_ms, end_ms = _extract_ms(s)
        audio = str(s.get("audio", s.get("text", "")) or "").strip()
        video = s.get("video") if isinstance(s.get("video"), list) else []
        video_norm = [str(x).strip() for x in video if str(x).strip()]
        sent = dict(s)
        sent["sentence_index"] = int(s.get("sentence_index", i))
        sent["start_ms"] = start_ms
        sent["end_ms"] = end_ms
        sent["audio"] = audio
        sent["video"] = video_norm
        sents_norm.append(sent)
        texts.append(_chunk_text_for_embedding_av(audio, video_norm, max_video_chars=max_video_chars))

    embs = _compute_embeddings(emb_client, texts, model=emb_model, batch_size=emb_batch_size)
    chunks: List[Chunk] = []
    for sent, emb in zip(sents_norm, embs):
        start_ms = int(sent["start_ms"])
        end_ms = int(sent["end_ms"])
        audio = str(sent.get("audio", "") or "").strip()
        video = list(sent.get("video") or [])
        chunks.append(
            Chunk(
                start_ms=start_ms,
                end_ms=end_ms,
                sentence_count=1,
                audio=audio,
                video=video,
                sentences=[sent],
                emb=np.asarray(emb, dtype=np.float32),
            )
        )
    return chunks


def _merge_two_chunks(left: Chunk, right: Chunk, *, dedupe_video: bool) -> Chunk:
    audio = (left.audio + " " + right.audio).strip() if (left.audio or right.audio) else ""
    video = left.video + right.video
    if dedupe_video:
        video = _dedupe_preserve_order(video)

    # size-weighted embedding average
    total = max(1, int(left.sentence_count) + int(right.sentence_count))
    emb = (left.emb * float(left.sentence_count) + right.emb * float(right.sentence_count)) / float(total)

    return Chunk(
        start_ms=min(left.start_ms, right.start_ms),
        end_ms=max(left.end_ms, right.end_ms),
        sentence_count=int(left.sentence_count) + int(right.sentence_count),
        audio=audio,
        video=video,
        sentences=left.sentences + right.sentences,
        emb=emb.astype(np.float32),
    )


def _agglomerate_adjacent(
    chunks: List[Chunk],
    *,
    target_chunks: int,
    max_sentences_per_chunk: int,
    dedupe_video_on_merge: bool,
) -> List[Chunk]:
    """
    Greedily merge the most similar adjacent pair until len(chunks) == target_chunks,
    respecting max_sentences_per_chunk if >0.
    """
    if target_chunks < 1:
        raise ValueError("target_chunks must be >= 1")
    if not chunks:
        return []
    if len(chunks) <= target_chunks:
        return chunks

    # We'll maintain a simple O(n^2) loop over adjacent similarities updated each merge.
    # This is slower than a heap but simpler/safer for research scripts.
    # (n starts at sentence count, so for full lectures this can be heavy; use target_chunks
    #  around 70-90 and consider pre-chunking if needed.)

    cur = list(chunks)
    while len(cur) > target_chunks:
        best_i = -1
        best_sim = -1.0
        for i in range(len(cur) - 1):
            a = cur[i]
            b = cur[i + 1]
            if max_sentences_per_chunk > 0 and (a.sentence_count + b.sentence_count) > max_sentences_per_chunk:
                continue
            sim = _cosine(a.emb, b.emb)
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i < 0:
            # No valid merges under constraints; stop early.
            break

        merged = _merge_two_chunks(cur[best_i], cur[best_i + 1], dedupe_video=dedupe_video_on_merge)
        # replace pair with merged
        cur[best_i : best_i + 2] = [merged]

    return cur


def _enforce_min_chunk_size(
    chunks: List[Chunk],
    *,
    min_sentences_per_chunk: int,
    max_sentences_per_chunk: int,
    dedupe_video_on_merge: bool,
) -> List[Chunk]:
    """
    Merge any chunk with sentence_count < min_sentences_per_chunk with the most
    similar adjacent neighbor (subject to max_sentences_per_chunk if >0).
    """
    if min_sentences_per_chunk <= 1:
        return chunks
    if not chunks:
        return []

    cur = list(chunks)
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(cur):
            if ch.sentence_count >= min_sentences_per_chunk:
                continue

            # choose best adjacent neighbor
            candidates: List[Tuple[float, int]] = []
            if i - 1 >= 0:
                left = cur[i - 1]
                if max_sentences_per_chunk <= 0 or (left.sentence_count + ch.sentence_count) <= max_sentences_per_chunk:
                    candidates.append((_cosine(left.emb, ch.emb), i - 1))
            if i + 1 < len(cur):
                right = cur[i + 1]
                if max_sentences_per_chunk <= 0 or (right.sentence_count + ch.sentence_count) <= max_sentences_per_chunk:
                    candidates.append((_cosine(ch.emb, right.emb), i))

            if not candidates:
                continue

            # merge with higher-similarity neighbor
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, merge_at = candidates[0]
            merged = _merge_two_chunks(cur[merge_at], cur[merge_at + 1], dedupe_video=dedupe_video_on_merge)
            cur[merge_at : merge_at + 2] = [merged]
            changed = True
            break

    return cur


def _merge_tiny_chunks(
    chunks: List[Chunk],
    *,
    tiny_max_sentences: int,
    max_combined_sentences: int,
    dedupe_video_on_merge: bool,
) -> List[Chunk]:
    """
    Merge any chunk with sentence_count <= tiny_max_sentences with the most
    similar adjacent neighbor (subject to max_combined_sentences if >0).
    """
    if tiny_max_sentences <= 0:
        return chunks
    if not chunks:
        return []
    cur = list(chunks)
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(cur):
            if ch.sentence_count > tiny_max_sentences:
                continue
            candidates: List[Tuple[float, int]] = []
            if i - 1 >= 0:
                left = cur[i - 1]
                if max_combined_sentences <= 0 or (left.sentence_count + ch.sentence_count) <= max_combined_sentences:
                    candidates.append((_cosine(left.emb, ch.emb), i - 1))
            if i + 1 < len(cur):
                right = cur[i + 1]
                if max_combined_sentences <= 0 or (right.sentence_count + ch.sentence_count) <= max_combined_sentences:
                    candidates.append((_cosine(ch.emb, right.emb), i))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, merge_at = candidates[0]
            merged = _merge_two_chunks(cur[merge_at], cur[merge_at + 1], dedupe_video=dedupe_video_on_merge)
            cur[merge_at : merge_at + 2] = [merged]
            changed = True
            break
    return cur


def _chunks_to_output(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        start_ms = int(ch.start_ms)
        end_ms = int(ch.end_ms)
        start_s = start_ms / 1000.0
        audio = ch.audio.strip()
        video = _dedupe_preserve_order(list(ch.video))
        text = _format_audio_video_text(audio, video)
        out.append(
            {
                "chunk_index": idx,
                "timestamp": start_s,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(0, end_ms - start_ms),
                "text": text,
                "audio": audio,
                "video": video,
                "sentence_count": int(ch.sentence_count),
                "word_count": len(audio.split()) if audio else 0,
                "sentences": ch.sentences,
            }
        )
    return out


# ------------------------- Main -------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    data_dir = paper_dir / "data"
    default_env = repo_root / "config" / "keys.env"

    parser = argparse.ArgumentParser(description="Adjacent agglomerative chunking (paper folder).")
    parser.add_argument("--mode", type=str, choices=["audio", "av"], required=True)
    parser.add_argument(
        "--audio-sent-dir",
        type=str,
        default=str(data_dir / "audio_sentence_level"),
        help="paper/data/audio_sentence_level",
    )
    parser.add_argument(
        "--av-sent-dir",
        type=str,
        default=str(data_dir / "audio_video_sentence_level"),
        help="paper/data/audio_video_sentence_level",
    )
    parser.add_argument(
        "--video-ocr-dir",
        type=str,
        default=str(data_dir / "video_ocr"),
        help="paper/data/video_ocr (used for mode=audio only)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: paper/data/agglomerative_<mode>_<target>)",
    )
    parser.add_argument(
        "--out-lecture-names",
        action="store_true",
        help="Write outputs as lecture{N}.json (default: descriptive filenames).",
    )
    parser.add_argument("--lectures", type=str, default="", help="Comma-separated lecture ids, or empty to infer.")
    parser.add_argument("--target-chunks", type=int, default=75, help="Stop when this many chunks remain (default 75).")
    parser.add_argument(
        "--max-sentences-per-chunk",
        type=int,
        default=8,
        help="Maximum sentences per chunk (default 8). Set 0 to disable.",
    )
    parser.add_argument(
        "--tiny-merge-max-sentences",
        type=int,
        default=2,
        help="Merge chunks with <= this many sentences (default 2).",
    )
    parser.add_argument(
        "--tiny-merge-max-combined",
        type=int,
        default=10,
        help="Max combined size during tiny-merge (default 10).",
    )
    parser.add_argument(
        "--max-video-chars",
        type=int,
        default=4000,
        help="AV mode: cap video text included in embedding input (default 4000 chars).",
    )
    parser.add_argument("--env", type=str, default=str(default_env), help="Env file for API keys (default config/keys.env).")
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="Embedding model/deployment name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default 64).")
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--embedding-endpoint", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    args = parser.parse_args()

    _load_env(Path(args.env))

    mode = args.mode
    audio_sent_dir = Path(args.audio_sent_dir)
    av_sent_dir = Path(args.av_sent_dir)
    video_ocr_dir = Path(args.video_ocr_dir)

    if mode == "audio":
        base_dir = audio_sent_dir
        if not audio_sent_dir.exists():
            raise FileNotFoundError(f"audio-sent-dir not found: {audio_sent_dir}")
        if not video_ocr_dir.exists():
            raise FileNotFoundError(f"video-ocr-dir not found: {video_ocr_dir}")
    else:
        base_dir = av_sent_dir
        if not av_sent_dir.exists():
            raise FileNotFoundError(f"av-sent-dir not found: {av_sent_dir}")

    if args.lectures.strip():
        lectures = [t.strip() for t in args.lectures.split(",") if t.strip()]
    else:
        lectures = _infer_lectures(base_dir, video_ocr_dir if mode == "audio" else None)

    if not lectures:
        raise RuntimeError("No lectures to process.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = data_dir / f"agglomerative_{mode}_{int(args.target_chunks)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_client = _build_embedding_client(
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        embedding_endpoint=args.embedding_endpoint,
    )

    print(
        f"🎯 agglomerative_{mode}: target_chunks={int(args.target_chunks)} "
        f"max_sentences_per_chunk={int(args.max_sentences_per_chunk)} "
        f"tiny_merge_max_sentences={int(args.tiny_merge_max_sentences)} "
        f"tiny_merge_max_combined={int(args.tiny_merge_max_combined)} lectures={len(lectures)}"
    )

    for lec in lectures:
        if mode == "audio":
            sent_path = audio_sent_dir / f"audio_sentence_level{lec}.json"
            if not sent_path.exists():
                sent_path = audio_sent_dir / f"lecture{lec}.json"
            ocr_path = video_ocr_dir / f"video{lec}.json"
            if not ocr_path.exists():
                ocr_path = video_ocr_dir / f"lecture{lec}.json"
            if not sent_path.exists():
                print(f"❌ Skipping lecture {lec}: missing {sent_path}")
                continue
            if not ocr_path.exists():
                print(f"❌ Skipping lecture {lec}: missing {ocr_path}")
                continue

            print(f"\n🔄 Lecture {lec}: loading sentences (audio)")
            sentences = _load_json_list(sent_path)
            print(f"   📄 sentences={len(sentences)} (initial chunks)")
            init = _initial_chunks_audio(
                sentences,
                emb_client=emb_client,
                emb_model=args.model,
                emb_batch_size=int(args.batch_size),
            )
            merged = _agglomerate_adjacent(
                init,
                target_chunks=int(args.target_chunks),
                max_sentences_per_chunk=int(args.max_sentences_per_chunk),
                dedupe_video_on_merge=False,
            )
            merged = _merge_tiny_chunks(
                merged,
                tiny_max_sentences=int(args.tiny_merge_max_sentences),
                max_combined_sentences=int(args.tiny_merge_max_combined),
                dedupe_video_on_merge=False,
            )
            print(f"   🧩 chunks_after_merge={len(merged)}")

            # Attach OCR frames after chunking
            video_frames = _load_json_list(ocr_path)
            video_ts, frames_sorted = _prepare_video_frames(video_frames)
            for ch in merged:
                start_s = ch.start_ms / 1000.0
                end_s = ch.end_ms / 1000.0
                ch.video = _select_video_texts_for_span(
                    video_ts=video_ts,
                    video_frames_sorted=frames_sorted,
                    start_s=start_s,
                    end_s=end_s,
                )

            if bool(args.out_lecture_names):
                out_path = out_dir / f"lecture{lec}.json"
            else:
                out_path = out_dir / f"audio_video_agglomerative_audio_{int(args.target_chunks)}{lec}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(_chunks_to_output(merged), f, ensure_ascii=False, indent=2)
            print(f"   💾 wrote {out_path}")

        else:
            sent_path = av_sent_dir / f"audio_video_sentence_level{lec}.json"
            if not sent_path.exists():
                sent_path = av_sent_dir / f"lecture{lec}.json"
            if not sent_path.exists():
                print(f"❌ Skipping lecture {lec}: missing {sent_path}")
                continue

            print(f"\n🔄 Lecture {lec}: loading sentences (audio+video)")
            sentences_av = _load_json_list(sent_path)
            print(f"   📄 sentences={len(sentences_av)} (initial chunks)")
            init = _initial_chunks_av(
                sentences_av,
                emb_client=emb_client,
                emb_model=args.model,
                emb_batch_size=int(args.batch_size),
                max_video_chars=int(args.max_video_chars),
            )
            merged = _agglomerate_adjacent(
                init,
                target_chunks=int(args.target_chunks),
                max_sentences_per_chunk=int(args.max_sentences_per_chunk),
                dedupe_video_on_merge=True,
            )
            merged = _merge_tiny_chunks(
                merged,
                tiny_max_sentences=int(args.tiny_merge_max_sentences),
                max_combined_sentences=int(args.tiny_merge_max_combined),
                dedupe_video_on_merge=True,
            )
            print(f"   🧩 chunks_after_merge={len(merged)}")

            if bool(args.out_lecture_names):
                out_path = out_dir / f"lecture{lec}.json"
            else:
                out_path = out_dir / f"audio_video_agglomerative_av_{int(args.target_chunks)}{lec}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(_chunks_to_output(merged), f, ensure_ascii=False, indent=2)
            print(f"   💾 wrote {out_path}")


if __name__ == "__main__":
    main()

