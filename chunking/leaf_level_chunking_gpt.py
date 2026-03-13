#!/usr/bin/env python3
"""
Paper-scoped copy of GPT leaf-level chunking.

This script mirrors `full pipeline 1/02_process/leaf_level_chunking_gpt.py`
but uses paper folder conventions by default:

- Input:  paper/data_avtext_fixed/audio_video_sentence_level/lecture{N}.json
- Output: paper/data_avtext_fixed/audio_video_chunk_level_gpt/lecture{N}.json

Flow:
- Build overlapping windows over sentence-level entries (default 16 entries, 25% overlap).
- Ask GPT-4o-mini for topic-shift timestamps per window.
- Validate timestamps against observed timestamps in each window.
- Cut the full lecture into chunks using detected shifts (min 20s chunk duration).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA_ROOT = REPO_ROOT / "paper" / "data_avtext_fixed"

# Load environment variables
load_dotenv(REPO_ROOT / "config" / "keys.env")

# Configuration
OPENAI_MINI_ENDPOINT = os.getenv("OPENAI_MINI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_RETRIES = 8
RETRY_DELAY = 2


def load_audio_sentence_data(file_path: Path) -> List[Dict[str, Any]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def create_overlapping_chunks(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    # For 25% overlap, advance by 75% of chunk_size.
    step_size = max(1, chunk_size * 7 // 8)
    for i in range(0, len(items), step_size):
        chunk = items[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if i + chunk_size >= len(items):
            break
    return chunks


def build_prompt_for_chunk(
    chunk: List[Dict[str, Any]],
    chunk_index: int,
    total_chunks: int,
    previous_shifts: List[float] | None = None,
) -> str:
    chunk_text = "\n".join([f"[{float(s.get('timestamp', 0)):.2f}s] {str(s.get('text', '')).strip()}" for s in chunk])
    prev_section = ""
    if previous_shifts:
        prev_str = ", ".join(f"{ts:.2f}s" for ts in previous_shifts)
        prev_section = (
            "\n\nPREVIOUS TOPIC SHIFTS (from the immediately preceding chunk):\n"
            f"{prev_str}\n"
            "These timestamps mark where the lecture has ALREADY been segmented. Avoid reselecting these same timestamps. "
            "Use them to determine future shifts.\n"
        )

    prompt = f"""You are analyzing lecture content to detect topic shifts. You will be given chunks of lecture audio with timestamps and need to identify where topic shifts occur.

The timestamps you see below are ABSOLUTE lecture timestamps (in seconds from the start of the lecture).{prev_section}

CONTENT WITH TIMESTAMPS:
{chunk_text}

INSTRUCTIONS:
- Your goal is to find topic shift points in this chunk - moments where the speaker moves from one idea to another.
- You should find anywhere from 1 to 4 topic shift points in this chunk. Do NOT exceed 4 shifts.
- Each final chunk should span at least 20 seconds between its first and last timestamp. Do NOT create a shift between two timestamps that are closer than 20 seconds apart.
- Be VERY CAREFUL that the timestamps you return exactly match those in the provided content. Do not hallucinate or invent timestamps.
- RESPOND WITH ONLY A PYTHON LIST OF TIMESTAMPS, LIKE THIS: [125.5, 185.6]
- Do not include any other text in your response.
"""
    return prompt


def extract_valid_timestamps(chunk: List[Dict[str, Any]]) -> List[float]:
    valid_timestamps = []
    for entry in chunk:
        ts = entry.get("timestamp")
        if ts is not None:
            try:
                valid_timestamps.append(float(ts))
            except (ValueError, TypeError):
                continue
    return sorted(set(valid_timestamps))


def validate_timestamps(
    returned_timestamps: List[float], valid_timestamps: List[float], tolerance: float = 0.1
) -> Tuple[bool, List[float]]:
    if not returned_timestamps:
        return True, []
    invalid = []
    for ts in returned_timestamps:
        matched = False
        for valid_ts in valid_timestamps:
            if abs(ts - valid_ts) <= tolerance:
                matched = True
                break
        if not matched:
            invalid.append(ts)
    return len(invalid) == 0, invalid


def call_gpt4o_mini(prompt: str, chunk: List[Dict[str, Any]], max_validation_retries: int = 3) -> List[float]:
    valid_timestamps = extract_valid_timestamps(chunk)
    if not valid_timestamps:
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    messages = [
        {"role": "system", "content": "You detect topic shifts and respond only with a valid Python list literal of timestamps."},
        {"role": "user", "content": prompt},
    ]

    total_retries = MAX_RETRIES * max_validation_retries
    for attempt in range(total_retries):
        try:
            body = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.1,
            }
            resp = requests.post(OPENAI_MINI_ENDPOINT, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
            content = (resp.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

            parsed_timestamps: List[float] = []
            try:
                import ast

                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    for x in parsed:
                        if isinstance(x, (int, float)):
                            parsed_timestamps.append(float(x))
            except Exception:
                if attempt < total_retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** (attempt % MAX_RETRIES)))
                continue

            is_valid, invalid_timestamps = validate_timestamps(parsed_timestamps, valid_timestamps)
            if is_valid:
                return parsed_timestamps

            error_msg = (
                f"INVALID RESPONSE: Your previous response was:\n\n{content}\n\n"
                f"This response is INVALID because it contains timestamps that do not exist in the provided content: {invalid_timestamps}. "
                f"You must ONLY return timestamps from the content."
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": error_msg})
            if attempt < total_retries - 1:
                time.sleep(RETRY_DELAY * (2 ** (attempt % MAX_RETRIES)))

        except Exception:
            if attempt < total_retries - 1:
                time.sleep(RETRY_DELAY * (2 ** (attempt % MAX_RETRIES)))

    return []


def create_chunks_from_shifts(
    entries: List[Dict[str, Any]],
    shift_timestamps: List[float],
    tolerance: float = 0.1,
    min_duration_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if not shift_timestamps:
        return [create_chunk_object(entries)]

    shift_timestamps = sorted(set(shift_timestamps))
    chunks: List[Dict[str, Any]] = []
    current_chunk_sentences: List[Dict[str, Any]] = []
    shift_idx = 0

    for entry in entries:
        entry_ts = float(entry.get("timestamp", 0))
        while shift_idx < len(shift_timestamps) and entry_ts >= shift_timestamps[shift_idx] - tolerance:
            if current_chunk_sentences:
                start_ts = float(current_chunk_sentences[0].get("timestamp", 0))
                duration = entry_ts - start_ts
                if duration >= min_duration_sec:
                    chunks.append(create_chunk_object(current_chunk_sentences))
                    current_chunk_sentences = []
            shift_idx += 1
        current_chunk_sentences.append(entry)

    if current_chunk_sentences:
        if min_duration_sec > 0 and chunks:
            start_ts = float(current_chunk_sentences[0].get("timestamp", 0))
            end_ts = float(current_chunk_sentences[-1].get("timestamp", 0))
            duration = end_ts - start_ts
            if duration < min_duration_sec:
                last_chunk = chunks.pop()
                merged_sentences: List[Dict[str, Any]] = []
                for e in entries:
                    ts = float(e.get("timestamp", 0))
                    if ts >= float(last_chunk["timestamp"]) and ts <= end_ts:
                        merged_sentences.append(e)
                if merged_sentences:
                    chunks.append(create_chunk_object(merged_sentences))
            else:
                chunks.append(create_chunk_object(current_chunk_sentences))
        else:
            chunks.append(create_chunk_object(current_chunk_sentences))
    return chunks


def create_chunk_object(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sentences:
        return {}

    start_entry = sentences[0]
    end_entry = sentences[-1]
    start_timestamp = float(start_entry.get("timestamp", 0))
    end_timestamp = float(end_entry.get("timestamp", 0))

    has_explicit_av = any(("audio" in s) or ("video" in s) for s in sentences)
    chunk_audio: str | None = None
    chunk_video: List[str] | None = None

    if has_explicit_av:
        import difflib

        audio_parts: List[str] = []
        video_frames: List[str] = []

        def is_fuzzy_duplicate(candidate: str, existing: List[str], threshold: float = 0.97) -> bool:
            cand_norm = " ".join(candidate.split()).lower()
            for ex in existing:
                ex_norm = " ".join(ex.split()).lower()
                if not ex_norm:
                    continue
                if difflib.SequenceMatcher(None, cand_norm, ex_norm).ratio() >= threshold:
                    return True
            return False

        for sent in sentences:
            audio_body = str(sent.get("audio", sent.get("text", ""))).strip()
            if audio_body:
                audio_parts.append(audio_body)

            frames = sent.get("video") or []
            if isinstance(frames, list):
                for frame in frames:
                    frame_text = str(frame).strip()
                    if not frame_text:
                        continue
                    if is_fuzzy_duplicate(frame_text, video_frames):
                        continue
                    video_frames.append(frame_text)

        audio_combined = " ".join(audio_parts).strip()
        chunk_audio = audio_combined if audio_combined else None
        chunk_video = video_frames if video_frames else []

        if video_frames:
            video_block_lines = [f"{i}: {txt}" for i, txt in enumerate(video_frames, start=1)]
            video_block = "\n".join(video_block_lines)
            combined_text = f"Audio: {audio_combined}\n\nVideo:\n{video_block}" if audio_combined else f"Video:\n{video_block}"
        else:
            combined_text = f"Audio: {audio_combined}" if audio_combined else ""
    else:
        text_parts: List[str] = []
        for sent in sentences:
            text = str(sent.get("text", "")).strip()
            if text:
                text_parts.append(text)
        combined_text = " ".join(text_parts)

    start_ms = int(start_entry.get("start_ms", start_timestamp * 1000))
    if "end_ms" in end_entry:
        end_ms = int(end_entry.get("end_ms"))
    elif "duration_ms" in end_entry:
        last_start_ms = int(end_entry.get("start_ms", end_timestamp * 1000))
        end_ms = last_start_ms + int(end_entry.get("duration_ms", 0))
    else:
        end_ms = int(end_timestamp * 1000)

    chunk: Dict[str, Any] = {
        "timestamp": start_timestamp,
        "text": combined_text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "sentence_count": len(sentences),
        "word_count": sum(int(sent.get("word_count", 0)) for sent in sentences),
        "sentences": sentences,
    }
    if chunk_audio is not None:
        chunk["audio"] = chunk_audio
    if chunk_video:
        chunk["video"] = chunk_video
    return chunk


def process_single_lecture(audio_file: Path, out_dir: Path, chunk_size: int) -> None:
    stem = audio_file.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    lecture_num = digits or stem
    print(f"\nProcessing lecture {lecture_num}...")

    entries = load_audio_sentence_data(audio_file)
    if not entries:
        print("No entries loaded; skipping.")
        return

    for idx, entry in enumerate(entries):
        entry["sentence_index"] = idx

    chunks = create_overlapping_chunks(entries, chunk_size)
    all_shifts: List[float] = []
    previous_chunk_shifts: List[float] = []
    for idx, chunk in enumerate(chunks):
        prompt = build_prompt_for_chunk(chunk, idx, len(chunks), previous_shifts=previous_chunk_shifts)
        ts_list = call_gpt4o_mini(prompt, chunk)
        if ts_list:
            all_shifts.extend(ts_list)
            previous_chunk_shifts = ts_list

    uniq_sorted = sorted({float(t) for t in all_shifts})
    chunk_objects = create_chunks_from_shifts(entries, uniq_sorted, min_duration_sec=20.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lecture{lecture_num}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_objects, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunk_objects)} chunks to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper GPT leaf-level chunking over sentence-level entries.")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=str(PAPER_DATA_ROOT / "audio_video_sentence_level"),
        help="Directory containing lecture{N}.json sentence-level files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PAPER_DATA_ROOT / "audio_video_chunk_level_gpt"),
        help="Output directory for lecture{N}.json chunk files.",
    )
    parser.add_argument("--chunk-size", type=int, default=16, help="Entries per overlapping window.")
    parser.add_argument("--lectures", type=str, default="", help="Comma-separated lecture numbers to process (e.g., 1,2,3).")
    parser.add_argument("--file-pattern", type=str, default="lecture*.json", help="Glob pattern for input files.")
    args = parser.parse_args()

    if not OPENAI_MINI_ENDPOINT or not OPENAI_API_KEY:
        print("Missing OPENAI_MINI_ENDPOINT or OPENAI_API_KEY in config/keys.env")
        return

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    if not audio_dir.exists():
        print(f"Input directory not found: {audio_dir}")
        return

    files = sorted(audio_dir.glob(args.file_pattern))

    def extract_lec_id(path: Path) -> str:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return digits

    if args.lectures.strip():
        wanted = {tok.strip() for tok in args.lectures.split(",") if tok.strip().isdigit()}
        files = [p for p in files if extract_lec_id(p) in wanted]

    if not files:
        print("No matching input files to process.")
        return

    print(f"Detecting topic shifts from {len(files)} lecture(s) with chunk-size={args.chunk_size}")
    for p in files:
        try:
            process_single_lecture(p, out_dir, args.chunk_size)
        except Exception as e:
            print(f"Failed {p.name}: {e}")


if __name__ == "__main__":
    main()

