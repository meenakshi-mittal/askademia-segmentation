#!/usr/bin/env python3
"""
Fixed-duration chunking over sentence-level audio + attach video OCR frames.

This script implements the "fixed_chunks" baseline described for paper experiments:

1) Build chunks by iterating through `paper/data/audio_sentence_level/audio_sentence_level{N}.json`
   and aggregating adjacent sentences until the CURRENT chunk duration reaches/exceeds 60s
   (default). The sentence that causes the chunk to reach/exceed the threshold is INCLUDED in
   that chunk; the next chunk starts at the following sentence.

2) Attach video OCR frames from `paper/data/video_ocr/video{N}.json` to each chunk.
   For a chunk span [start_s, end_s], we include:
     - the LAST OCR frame strictly before start_s (if any)
     - PLUS all OCR frames with timestamps in [start_s, end_s]

3) Output a JSON list of chunk objects that mirrors the `audio_video_sentence_level` style
   (has `audio` and `video` keys), plus chunk metadata and the underlying sentences.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def _extract_ms(entry: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return (start_ms, end_ms) for a sentence entry, with robust fallbacks.
    """
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

    # Last resort: assume 1s span
    return start_ms, start_ms + 1000


def _build_fixed_chunks(
    sentences: List[Dict[str, Any]],
    *,
    threshold_s: float,
) -> List[Dict[str, Any]]:
    """
    Build fixed-duration chunks from sentence-level audio.

    Rule: keep adding sentences until the chunk duration >= threshold, then flush.
    """
    threshold_ms = int(round(float(threshold_s) * 1000.0))
    if threshold_ms <= 0:
        raise ValueError("threshold_s must be > 0")

    # Enrich with sentence_index + ms fields for consistent downstream behavior.
    enriched: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences):
        start_ms, end_ms = _extract_ms(s)
        s2 = dict(s)
        s2["sentence_index"] = int(s.get("sentence_index", i))
        s2["start_ms"] = start_ms
        s2["end_ms"] = end_ms
        s2["duration_ms"] = max(0, end_ms - start_ms)
        # For consistency with audio_video_sentence_level style
        s2["audio"] = str(s.get("audio", s.get("text", "")) or "").strip()
        enriched.append(s2)

    chunks: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    chunk_idx = 0

    def flush(cur_sents: List[Dict[str, Any]]) -> None:
        nonlocal chunk_idx
        if not cur_sents:
            return
        start_ms = int(cur_sents[0]["start_ms"])
        end_ms = int(cur_sents[-1]["end_ms"])
        start_s = start_ms / 1000.0

        audio_text = " ".join(str(x.get("text", "")).strip() for x in cur_sents if str(x.get("text", "")).strip()).strip()
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "timestamp": start_s,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(0, end_ms - start_ms),
                "text": audio_text,  # mirror audio_video_sentence_level where text ~= audio
                "audio": audio_text,
                # `video` will be filled later
                "video": [],
                "sentence_count": len(cur_sents),
                "word_count": len(audio_text.split()) if audio_text else 0,
                "sentences": cur_sents,
            }
        )
        chunk_idx += 1

    for s in enriched:
        if not cur:
            cur = [s]
        else:
            cur.append(s)

        start_ms = int(cur[0]["start_ms"])
        end_ms = int(cur[-1]["end_ms"])
        if (end_ms - start_ms) >= threshold_ms:
            flush(cur)
            cur = []

    if cur:
        flush(cur)

    return chunks


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
    """
    Include last frame strictly before start_s, plus all frames in [start_s, end_s].
    Returns the selected frame texts (strings).
    """
    if not video_frames_sorted:
        return []
    if end_s < start_s:
        return []

    out: List[Dict[str, Any]] = []

    # last strictly-before frame
    prev_idx = bisect_left(video_ts, start_s) - 1
    if prev_idx >= 0:
        out.append(video_frames_sorted[prev_idx])

    # frames in [start_s, end_s]
    lo = bisect_left(video_ts, start_s)
    hi = bisect_right(video_ts, end_s)
    out.extend(video_frames_sorted[lo:hi])

    # Convert to text list (dropping empties)
    texts: List[str] = []
    for f in out:
        t = str(f.get("text", "") or "").strip()
        if t:
            texts.append(t)
    return texts


def _infer_lectures(audio_dir: Path, video_dir: Path) -> List[str]:
    audio_ids = {
        p.stem.replace("audio_sentence_level", "")
        for p in audio_dir.glob("audio_sentence_level*.json")
    }
    video_ids = {p.stem.replace("video", "") for p in video_dir.glob("video*.json")}
    ids = sorted(audio_ids & video_ids, key=lambda x: int(x) if x.isdigit() else x)
    return [i for i in ids if i]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    data_dir = paper_dir / "data"

    parser = argparse.ArgumentParser(
        description="Build 60s fixed chunks from audio_sentence_level and attach video_ocr frames (paper experiments)."
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=str(data_dir / "audio_sentence_level"),
        help="Directory containing audio_sentence_level{N}.json (default: paper/data/audio_sentence_level).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(data_dir / "video_ocr"),
        help="Directory containing video{N}.json OCR files (default: paper/data/video_ocr).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(data_dir / "fixed_chunks_90s"),
        help="Output directory (default: paper/data/fixed_chunks_90s).",
    )
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        default=90.0,
        help="Chunk duration threshold in seconds (default 60).",
    )
    parser.add_argument(
        "--lectures",
        type=str,
        default="",
        help="Comma-separated lecture ids to process (e.g. '1,2,10'). If empty, infer intersection.",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"audio-dir not found: {audio_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"video-dir not found: {video_dir}")

    if args.lectures.strip():
        lectures = [t.strip() for t in args.lectures.split(",") if t.strip()]
    else:
        lectures = _infer_lectures(audio_dir, video_dir)

    if not lectures:
        raise RuntimeError("No lectures to process (no overlapping audio/video files found).")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 fixed_chunks: threshold={float(args.threshold_seconds):.1f}s | lectures={', '.join(lectures)}")

    for lec in lectures:
        audio_path = audio_dir / f"audio_sentence_level{lec}.json"
        video_path = video_dir / f"video{lec}.json"
        if not audio_path.exists():
            print(f"❌ Skipping lecture {lec}: missing {audio_path}")
            continue
        if not video_path.exists():
            print(f"❌ Skipping lecture {lec}: missing {video_path}")
            continue

        print(f"\n🔄 Lecture {lec}: building fixed chunks from {audio_path.name}")
        sentences = _load_json_list(audio_path)
        chunks = _build_fixed_chunks(sentences, threshold_s=float(args.threshold_seconds))
        print(f"   🧩 Built {len(chunks)} chunks")

        print(f"   🎞️  Attaching OCR frames from {video_path.name}")
        video_frames = _load_json_list(video_path)
        video_ts, frames_sorted = _prepare_video_frames(video_frames)

        for ch in chunks:
            start_ms = int(ch.get("start_ms", 0))
            end_ms = int(ch.get("end_ms", start_ms))
            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0
            ch["video"] = _select_video_texts_for_span(
                video_ts=video_ts,
                video_frames_sorted=frames_sorted,
                start_s=start_s,
                end_s=end_s,
            )

        out_path = out_dir / f"audio_video_fixed_chunks_90s{lec}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"   💾 Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()

