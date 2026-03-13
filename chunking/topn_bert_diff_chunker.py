#!/usr/bin/env python3
"""
Compute BERT diffs on audio_sentence_level + video_ocr, then chunk by top-N splits.

For each lecture:
1) Compute and write in-place:
   - bert_diff      (entry i text vs entry i+1 text)
   - bert_k_diff    (left window [i-k+1..i] vs right window [i+1..i+k])
     using concatenated text windows with clipped boundaries.
2) Build chunks by splitting at top-N values of a chosen diff field
   (e.g., bert_diff, bert_2_diff, bert_3_diff).

Window behavior matches the requested example:
  - bert_2_diff at sentence 4 compares (3+4) vs (5+6)
  - bert_2_diff at sentence 5 compares (4+5) vs (6+7)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from exc


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _as_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("text", "") or "").strip()


def _as_timestamp_s(entry: Dict[str, Any]) -> float:
    try:
        return float(entry.get("timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0


def _as_start_ms(entry: Dict[str, Any]) -> int:
    if "start_ms" in entry:
        try:
            return int(entry["start_ms"])
        except Exception:
            pass
    return int(round(_as_timestamp_s(entry) * 1000.0))


def _as_end_ms(entry: Dict[str, Any], next_entry: Optional[Dict[str, Any]]) -> int:
    if "end_ms" in entry:
        try:
            return int(entry["end_ms"])
        except Exception:
            pass
    start_ms = _as_start_ms(entry)
    if next_entry is not None:
        next_start = _as_start_ms(next_entry)
        if next_start > start_ms:
            return next_start
    return start_ms


def _parse_lecture_ids(value: str) -> Optional[List[str]]:
    if not value.strip():
        return None
    out = [x.strip() for x in value.split(",") if x.strip()]
    return out if out else None


def _extract_lecture_id(path: Path) -> str:
    stem = path.stem  # lecture1
    if stem.startswith("lecture"):
        return stem.replace("lecture", "")
    return stem


def _collect_lecture_files(data_dir: Path, wanted: Optional[Sequence[str]]) -> List[Path]:
    files = sorted(data_dir.glob("lecture*.json"), key=lambda p: _extract_lecture_id(p))
    if wanted is None:
        return files
    wanted_set = set(wanted)
    return [p for p in files if _extract_lecture_id(p) in wanted_set]


def _compute_diffs_for_entries(
    entries: List[Dict[str, Any]],
    model: SentenceTransformer,
    window_ks: Sequence[int],
) -> None:
    n = len(entries)
    if n == 0:
        return

    texts = [_as_text(e) for e in entries]

    # Base embeddings for bert_diff.
    base_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    for i in range(n):
        if i == n - 1:
            entries[i]["bert_diff"] = 0.0
            continue
        sim = _cosine_similarity(base_emb[i], base_emb[i + 1])
        entries[i]["bert_diff"] = 1.0 - sim

    # Windowed diffs: bert_k_diff
    for k in window_ks:
        if k <= 0:
            continue

        left_right_pairs: List[Tuple[str, str]] = []
        for i in range(n):
            left_start = max(0, i - k + 1)
            left_end = i
            right_start = i + 1
            right_end = min(n - 1, i + k)

            left_text = " ".join(t for t in texts[left_start : left_end + 1] if t)
            right_text = " ".join(t for t in texts[right_start : right_end + 1] if t)
            left_right_pairs.append((left_text, right_text))

        unique_texts: Dict[str, int] = {}
        encode_list: List[str] = []
        for left_text, right_text in left_right_pairs:
            if left_text and left_text not in unique_texts:
                unique_texts[left_text] = len(encode_list)
                encode_list.append(left_text)
            if right_text and right_text not in unique_texts:
                unique_texts[right_text] = len(encode_list)
                encode_list.append(right_text)

        if encode_list:
            all_emb = model.encode(encode_list, convert_to_numpy=True, normalize_embeddings=False)
        else:
            all_emb = np.zeros((0, 1), dtype=np.float32)

        field = f"bert_{k}_diff"
        for i, (left_text, right_text) in enumerate(left_right_pairs):
            if not right_text:
                entries[i][field] = 0.0
                continue
            if not left_text:
                entries[i][field] = 1.0
                continue

            li = unique_texts[left_text]
            ri = unique_texts[right_text]
            sim = _cosine_similarity(all_emb[li], all_emb[ri])
            entries[i][field] = 1.0 - sim


def _top_boundary_indices(entries: List[Dict[str, Any]], diff_field: str, top_n: int) -> List[int]:
    n = len(entries)
    if n <= 1 or top_n <= 0:
        return []

    candidates: List[Tuple[float, float, int]] = []
    # Boundary is between i and i+1 => new chunk starts at i+1
    for i in range(0, n - 1):
        score = float(entries[i].get(diff_field, 0.0) or 0.0)
        ts = _as_timestamp_s(entries[i])
        candidates.append((score, ts, i + 1))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    k = min(top_n, len(candidates))
    starts = sorted(idx for _, _, idx in candidates[:k])
    return starts


def _build_chunks(entries: List[Dict[str, Any]], starts: List[int], diff_field: str) -> List[Dict[str, Any]]:
    n = len(entries)
    bounds = [0] + starts + [n]
    chunks: List[Dict[str, Any]] = []

    for chunk_index, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        if lo >= hi:
            continue
        chunk_entries = entries[lo:hi]

        start_ms = _as_start_ms(chunk_entries[0])
        end_ms = _as_end_ms(chunk_entries[-1], entries[hi] if hi < n else None)
        timestamp = float(start_ms) / 1000.0
        text = " ".join(_as_text(x) for x in chunk_entries if _as_text(x))

        split_from_prev_score = None
        if lo > 0:
            split_from_prev_score = float(entries[lo - 1].get(diff_field, 0.0) or 0.0)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "timestamp": timestamp,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(0, end_ms - start_ms),
                "text": text,
                "item_count": len(chunk_entries),
                "word_count": len(text.split()) if text else 0,
                "split_from_prev_score": split_from_prev_score,
                "diff_field": diff_field,
                "items": chunk_entries,
            }
        )

    return chunks


def _process_one_dir(
    *,
    model: SentenceTransformer,
    input_dir: Path,
    out_dir: Path,
    diff_field: str,
    top_n: int,
    window_ks: Sequence[int],
    lectures: Optional[Sequence[str]],
) -> Tuple[int, int]:
    files = _collect_lecture_files(input_dir, lectures)
    if not files:
        print(f"WARNING: No lecture*.json found in {input_dir}")
        return 0, 0

    print(f"📂 {input_dir} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    chunked = 0
    for path in files:
        lecture_id = _extract_lecture_id(path)
        entries = _load_json_list(path)
        if not entries:
            print(f"   WARNING lecture{lecture_id}: empty file")
            continue

        _compute_diffs_for_entries(entries, model=model, window_ks=window_ks)
        _write_json(path, entries)
        updated += 1

        starts = _top_boundary_indices(entries, diff_field=diff_field, top_n=top_n)
        chunks = _build_chunks(entries, starts=starts, diff_field=diff_field)
        out_path = out_dir / f"lecture{lecture_id}.json"
        _write_json(out_path, chunks)
        chunked += 1

        print(
            f"   OK lecture{lecture_id}: {len(entries)} entries, "
            f"{len(starts)} boundaries, {len(chunks)} chunks"
        )

    return updated, chunked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bert diffs and chunk by top-N boundaries for audio_sentence_level and video_ocr."
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="askademia segmentation/data/audio_sentence_level",
        help="Input directory for audio sentence-level files (lecture*.json).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="askademia segmentation/data/video_ocr",
        help="Input directory for video OCR files (lecture*.json).",
    )
    parser.add_argument(
        "--audio-out-dir",
        type=str,
        default="askademia segmentation/data/topn_chunks_audio_sentence_level",
        help="Output directory for top-N chunked audio files.",
    )
    parser.add_argument(
        "--video-out-dir",
        type=str,
        default="askademia segmentation/data/topn_chunks_video_ocr",
        help="Output directory for top-N chunked video files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=80,
        help="Number of top diff boundaries to split on (per lecture).",
    )
    parser.add_argument(
        "--diff-field",
        type=str,
        default="bert_2_diff",
        help="Field used for selecting top-N boundaries (e.g., bert_diff, bert_2_diff, bert_3_diff).",
    )
    parser.add_argument(
        "--window-ks",
        type=str,
        default="2,3",
        help="Comma-separated k values for bert_k_diff computation.",
    )
    parser.add_argument(
        "--lectures",
        type=str,
        default="",
        help="Optional comma-separated lecture ids (e.g., 1,2,10).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    window_ks: List[int] = []
    if args.window_ks.strip():
        for x in args.window_ks.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                k = int(x)
                if k > 0:
                    window_ks.append(k)
            except Exception:
                continue

    if args.diff_field.startswith("bert_"):
        try:
            k_from_field = int(args.diff_field.split("_")[1])
            if k_from_field > 0 and k_from_field not in window_ks:
                window_ks.append(k_from_field)
        except Exception:
            pass

    lectures = _parse_lecture_ids(args.lectures)

    print("topn_bert_diff_chunker")
    print(f"   model={args.model_name}")
    print(f"   top_n={args.top_n}")
    print(f"   diff_field={args.diff_field}")
    print(f"   window_ks={window_ks if window_ks else 'none'}")
    if lectures:
        print(f"   lectures={','.join(lectures)}")

    model = SentenceTransformer(args.model_name)

    audio_updated, audio_chunked = _process_one_dir(
        model=model,
        input_dir=Path(args.audio_dir),
        out_dir=Path(args.audio_out_dir),
        diff_field=args.diff_field,
        top_n=max(0, int(args.top_n)),
        window_ks=window_ks,
        lectures=lectures,
    )
    video_updated, video_chunked = _process_one_dir(
        model=model,
        input_dir=Path(args.video_dir),
        out_dir=Path(args.video_out_dir),
        diff_field=args.diff_field,
        top_n=max(0, int(args.top_n)),
        window_ks=window_ks,
        lectures=lectures,
    )

    print("\nDone")
    print(f"   audio files updated: {audio_updated}, chunked: {audio_chunked}")
    print(f"   video files updated: {video_updated}, chunked: {video_chunked}")


if __name__ == "__main__":
    main()

