#!/usr/bin/env python3
"""
Build GPT summary trees from chunk files (paper/data_avtext_fixed/*).

This mirrors the original GPT tree logic:
  - Summarize each chunk with GPT-4o-mini (title + summary).
  - Treat those summaries as "sentences".
  - Use GPT topic-shift detection to group adjacent nodes.
  - Summarize each group into a new parent node.
  - Repeat until <= min_nodes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = REPO_ROOT / "config" / "keys.env"

OPENAI_MINI_ENDPOINT = None
OPENAI_API_KEY = None
PRINT_GENERATIONS = False
MAX_RETRIES = 8
RETRY_DELAY = 2


@dataclass
class SummaryNode:
    level: int
    index: int
    start_timestamp: float
    end_timestamp: float
    title: str
    summary: str
    children: List["SummaryNode"]
    leaf_chunk: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "level": self.level,
            "index": self.index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "title": self.title,
            "summary": self.summary,
            "children": [c.to_json() for c in self.children],
        }
        if self.leaf_chunk is not None:
            obj["leaf_chunk"] = self.leaf_chunk
        return obj


def _load_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path)
    global OPENAI_MINI_ENDPOINT, OPENAI_API_KEY
    OPENAI_MINI_ENDPOINT = os.getenv("OPENAI_MINI_ENDPOINT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = "".join(ch if ch in ("\n", "\t") or ord(ch) >= 32 else " " for ch in text)
    cleaned = cleaned.replace("\u0000", " ")
    cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii", "ignore")
    return cleaned


def _strip_video(text: str) -> str:
    if not text:
        return ""
    idx = text.find("Video:")
    if idx == -1:
        return text
    return text[:idx].rstrip()


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                return None
    return None


def _local_fallback_summary(text: str) -> Tuple[str, str]:
    cleaned = _clean_text(text).strip()
    if not cleaned:
        return "Untitled segment", ""
    words = cleaned.split()
    short = " ".join(words[:60]).strip()
    if len(words) > 60:
        short += " ..."
    return "Untitled segment", short


def call_gpt4o_mini_for_title_summary(
    text: str,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    audio_fallback: Optional[str] = None,
) -> Tuple[str, str]:
    if not OPENAI_MINI_ENDPOINT or not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_MINI_ENDPOINT or OPENAI_API_KEY in environment.")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    snippet = _clean_text(text).strip()
    if len(snippet) > 2000:
        snippet = snippet[:2000] + " ..."
    fallback_snippet = _strip_video(snippet)
    audio_snippet = _clean_text(audio_fallback or "").strip()
    used_fallback = False

    system_prompt = (
        "You are summarizing segments of a university lecture. Your job is to:\n"
        "1) Distinguish REAL lecture content (definitions, explanations, derivations, worked examples, conceptual discussion, "
        "   and course logistics/announcements that convey information about how the course operates)\n"
        "   from truly NON-LECTURE interludes such as video/audio disconnects, waiting for slides, extended silence, off-topic small talk, or explicit breaks.\n"
        "2) If a segment is ENTIRELY one of these irrelevant interludes (e.g., technical failure, long break, idle chatter unrelated to the course),\n"
        "   set the title to exactly:\n"
        '     "Non-lecture interlude (logistics/technical/etc.)"\n'
        "   and write a brief summary describing that this part is just a pause, technical issue, or other non-content.\n"
        "   Do NOT use this label for course logistics, grading details, deadlines, or other actual course-related information — those should be summarized as normal content.\n"
        "3) For all other segments (including course logistics and announcements), IGNORE any tiny bits of irrelevant chatter and summarize ONLY the meaningful lecture or course content.\n"
        "4) The title must be precise and specific (3–10 words). It should clearly indicate the topic AND aspect.\n"
        "5) The summary should be concise while still covering ALL major lecture points in the segment. "
        "It can be up to a long paragraph if needed to capture the full content, but should avoid filler.\n"
        "6) Do NOT refer to \"this lecture\", \"this segment\", or similar meta language. Just state the content directly.\n"
        "7) The title must cover the ENTIRE segment, reflecting ALL major child topics, not just the first bullet. "
        "If multiple distinct concepts appear, the title should make that clear rather than focusing on only the opening idea."
    )
    time_context = ""
    if start_ts is not None and end_ts is not None:
        time_context = f"Time span: {format_ts(start_ts)} → {format_ts(end_ts)} (timestamps refer to the lecture timeline).\n"
    for attempt in range(MAX_RETRIES):
        user_prompt = (
            "LECTURE SEGMENT:\n"
            "----------------\n"
            f"{time_context}{snippet}\n\n"
            "Respond with a JSON object of the form:\n"
            '{"title": "...", "summary": "..."}\n'
            "Do not include any additional text, Markdown, or explanation."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            body = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.4, "max_tokens": 1200}
            resp = requests.post(OPENAI_MINI_ENDPOINT, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
            content = (resp.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            data = _extract_json(content)
            if isinstance(data, dict):
                title = str(data.get("title", "")).strip() or "Untitled segment"
                summary = str(data.get("summary", "")).strip() or (snippet[:200] + ("..." if len(snippet) > 200 else ""))
                return title, summary
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if lines:
                title = lines[0][:80]
                summary = " ".join(lines[1:]) or snippet[:200]
                return title, summary
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            detail = ""
            if resp is not None:
                text = (resp.text or "").strip()
                if len(text) > 1000:
                    text = text[:1000] + " ..."
                detail = f" | response: {text}"
                if "content_filter" in text and not used_fallback:
                    if audio_snippet:
                        print("⚠️  Content filter triggered; retrying with audio-only field.")
                        snippet = audio_snippet
                        used_fallback = True
                        continue
                    if fallback_snippet and fallback_snippet != snippet:
                        print("⚠️  Content filter triggered; retrying with audio-only snippet.")
                        snippet = fallback_snippet
                        used_fallback = True
                        continue
                if "content_filter" in text and used_fallback:
                    return _local_fallback_summary(audio_snippet or snippet)
            print(
                f"⚠️  Failed to summarize segment (attempt {attempt + 1}/{MAX_RETRIES}): {e}{detail}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
        except Exception as e:
            print(f"⚠️  Failed to summarize segment (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    fallback = snippet[:200] + ("..." if len(snippet) > 200 else "")
    return "Untitled segment", fallback


def summarize_from_children(children: List[SummaryNode]) -> Tuple[str, str]:
    parts: List[str] = []
    for ch in children:
        title = (ch.title or "").strip()
        if title.lower().startswith("non-lecture interlude"):
            continue
        summ = ch.summary or ""
        parts.append(f"- {title or 'Untitled'}: {summ}")
    if not parts:
        for ch in children:
            title = (ch.title or "").strip()
            summ = ch.summary or ""
            parts.append(f"- {title or 'Untitled'}: {summ}")
    text = "\n".join(parts)
    start_ts = children[0].start_timestamp
    end_ts = children[-1].end_timestamp
    return call_gpt4o_mini_for_title_summary(text, start_ts=start_ts, end_ts=end_ts)


def _create_overlapping_chunks(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    step_size = max(1, chunk_size * 7 // 8)
    for i in range(0, len(items), step_size):
        chunk = items[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if i + chunk_size >= len(items):
            break
    return chunks


def _build_prompt_for_shift_chunk(
    chunk: List[Dict[str, Any]],
    chunk_index: int,
    total_chunks: int,
) -> str:
    chunk_text = "\n".join(
        [f"[{float(s.get('timestamp', 0)):.2f}s] {str(s.get('text','')).strip()}" for s in chunk]
    )
    prompt = f"""You are analyzing lecture content to detect topic shifts. You will be given chunks of lecture audio with timestamps and need to identify where topic shifts occur.

The timestamps you see below are ABSOLUTE lecture timestamps (in seconds from the start of the lecture).

CONTENT WITH TIMESTAMPS:
{chunk_text}

INSTRUCTIONS:
- Your goal is to find **topic shift points** in this chunk — moments where the speaker moves from one idea to another.  
- You should find anywhere from 1 to 4 topic shift points in this chunk. Do NOT exceed 4 shifts.
- Each final chunk should span at least **20 seconds** between its first and last timestamp. Do NOT create a shift between two timestamps that are closer than 20 seconds apart.  
- Be **VERY CAREFUL** that the timestamps you return exactly match those in the provided content. Do **not** hallucinate or invent timestamps.
- RESPOND WITH ONLY A PYTHON LIST OF TIMESTAMPS, LIKE THIS: `[125.5, 185.6]`  
- Do not include any other text in your response, including ```json or ```python."""
    return prompt


def _call_gpt4o_mini_for_shifts(prompt: str) -> List[float]:
    if not OPENAI_MINI_ENDPOINT or not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_MINI_ENDPOINT or OPENAI_API_KEY in environment.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    for attempt in range(MAX_RETRIES):
        try:
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
            resp = requests.post(OPENAI_MINI_ENDPOINT, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
            content = (resp.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            try:
                data = ast.literal_eval(content)
                if isinstance(data, list):
                    return [float(x) for x in data]
            except Exception:
                nums = []
                for tok in content.replace("[", " ").replace("]", " ").split(","):
                    tok = tok.strip()
                    if tok:
                        try:
                            nums.append(float(tok))
                        except Exception:
                            continue
                if nums:
                    return nums
        except Exception as e:
            print(f"⚠️  API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    return []


def _extract_valid_timestamps(chunk: List[Dict[str, Any]]) -> List[float]:
    valid = []
    for entry in chunk:
        ts = entry.get("timestamp")
        if ts is not None:
            try:
                valid.append(float(ts))
            except Exception:
                continue
    return sorted(set(valid))


def _filter_shifts_to_valid(shifts: List[float], valid_ts: List[float], tolerance: float = 0.1) -> List[float]:
    if not shifts:
        return []
    valid = []
    for s in shifts:
        for v in valid_ts:
            if abs(s - v) <= tolerance:
                valid.append(v)
                break
    return sorted(set(valid))


def _gpt_chunk_groups(
    nodes: List[SummaryNode],
    *,
    chunk_size: int,
    min_duration_sec: float,
) -> List[List[SummaryNode]]:
    if not nodes:
        return []
    nodes = sorted(nodes, key=lambda n: n.start_timestamp)
    items = [{"timestamp": n.start_timestamp, "text": f"{n.title}: {n.summary}"} for n in nodes]
    chunks = _create_overlapping_chunks(items, chunk_size)
    all_shifts: List[float] = []
    for i, ch in enumerate(chunks):
        prompt = _build_prompt_for_shift_chunk(ch, i, len(chunks))
        raw = _call_gpt4o_mini_for_shifts(prompt)
        valid_ts = _extract_valid_timestamps(ch)
        filtered = _filter_shifts_to_valid(raw, valid_ts, tolerance=0.1)
        all_shifts.extend(filtered)

    all_shifts = sorted(set(all_shifts))
    uniq_sorted = sorted({float(t) for t in all_shifts})
    entries = []
    for idx, node in enumerate(nodes):
        t = float(node.start_timestamp)
        text = (f"{node.title}: {node.summary}").strip()
        entries.append(
            {
                "timestamp": t,
                "text": text,
                "start_ms": int(t * 1000),
                "end_ms": int(node.end_timestamp * 1000),
                "duration_ms": int((node.end_timestamp - node.start_timestamp) * 1000),
                "word_count": len(text.split()),
                "node_index": idx,
            }
        )
    chunk_objects = _create_chunks_from_shifts(entries, uniq_sorted, min_duration_sec=min_duration_sec)
    groups: List[List[SummaryNode]] = []
    for ch in chunk_objects:
        child_nodes: List[SummaryNode] = []
        for sent in ch.get("sentences", []):
            ni = sent.get("node_index")
            if isinstance(ni, int) and 0 <= ni < len(nodes):
                if not child_nodes or child_nodes[-1] is not nodes[ni]:
                    child_nodes.append(nodes[ni])
        if child_nodes:
            groups.append(child_nodes)
    return groups


def _create_chunks_from_shifts(
    entries: List[Dict[str, Any]],
    shift_timestamps: List[float],
    tolerance: float = 0.1,
    min_duration_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if not shift_timestamps:
        return [{"timestamp": entries[0]["timestamp"], "sentences": list(entries)}]
    shift_timestamps = sorted(set(shift_timestamps))
    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    shift_idx = 0
    for entry in entries:
        entry_ts = float(entry.get("timestamp", 0))
        while shift_idx < len(shift_timestamps) and entry_ts >= shift_timestamps[shift_idx] - tolerance:
            if current:
                start_ts = float(current[0].get("timestamp", 0))
                duration = entry_ts - start_ts
                if duration >= min_duration_sec:
                    chunks.append({"timestamp": current[0]["timestamp"], "sentences": list(current)})
                    current = []
            shift_idx += 1
        current.append(entry)
    if current:
        if min_duration_sec > 0 and chunks:
            start_ts = float(current[0].get("timestamp", 0))
            end_ts = float(current[-1].get("timestamp", 0))
            duration = end_ts - start_ts
            if duration < min_duration_sec:
                prev = chunks.pop()
                merged = prev["sentences"] + current
                chunks.append({"timestamp": merged[0]["timestamp"], "sentences": merged})
            else:
                chunks.append({"timestamp": current[0]["timestamp"], "sentences": list(current)})
        else:
            chunks.append({"timestamp": current[0]["timestamp"], "sentences": list(current)})
    return chunks


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    out = [d for d in data if isinstance(d, dict)]
    def _start_key(d: Dict[str, Any]) -> float:
        if d.get("start_ms") is not None:
            try:
                return float(d["start_ms"]) / 1000.0
            except Exception:
                pass
        if d.get("timestamp") is not None:
            try:
                return float(d["timestamp"])
            except Exception:
                pass
        return 0.0
    out.sort(key=_start_key)
    return out


def _build_leaf_nodes(chunks: List[Dict[str, Any]], *, level: int = 0) -> List[SummaryNode]:
    nodes: List[SummaryNode] = []
    for idx, ch in enumerate(chunks):
        text = str(ch.get("text", "") or "").strip()
        start_ts = float(ch.get("timestamp", 0.0))
        if "end_ms" in ch:
            end_ts = float(ch["end_ms"]) / 1000.0
        else:
            dur_ms = float(ch.get("duration_ms", 0.0))
            end_ts = start_ts + dur_ms / 1000.0
        title, summary = call_gpt4o_mini_for_title_summary(
            text,
            start_ts=start_ts,
            end_ts=end_ts,
            audio_fallback=str(ch.get("audio", "") or ""),
        )
        if PRINT_GENERATIONS:
            print(f"   leaf {idx}: title: {title} summary: {summary}")
        nodes.append(
            SummaryNode(
                level=level,
                index=idx,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                title=title,
                summary=summary,
                children=[],
                leaf_chunk=ch,
            )
        )
    return nodes


def _build_tree_for_lecture(
    chunks: List[Dict[str, Any]],
    *,
    min_nodes: int,
    reduction_factor: float,
    gpt_chunk_size: int,
    gpt_min_duration_sec: float,
) -> Dict[str, Any]:
    level = 0
    cur = _build_leaf_nodes(chunks, level=level)
    cur.sort(key=lambda n: n.start_timestamp)
    levels_meta = [{"level": level, "count": len(cur)}]
    while len(cur) > min_nodes:
        prev_len = len(cur)
        level += 1
        target_count = max(1, int(round(len(cur) / max(1.0, reduction_factor))))
        groups = _gpt_chunk_groups(cur, chunk_size=gpt_chunk_size, min_duration_sec=gpt_min_duration_sec)
        new_nodes: List[SummaryNode] = []
        for idx, group in enumerate(groups):
            title, summary = summarize_from_children(group)
            if PRINT_GENERATIONS:
                child_ids = ",".join(str(n.index) for n in group)
                print(f"   merging {child_ids}: title: {title} summary: {summary}")
            new_nodes.append(
                SummaryNode(
                    level=level,
                    index=idx,
                    start_timestamp=group[0].start_timestamp,
                    end_timestamp=group[-1].end_timestamp,
                    title=title,
                    summary=summary,
                    children=group,
                )
            )
        cur = new_nodes
        levels_meta.append({"level": level, "count": len(cur)})
        if not cur or len(cur) >= prev_len:
            break
        if len(cur) == 1:
            break

    # Auto-combine final 2-5 nodes into a single root
    if 1 < len(cur) <= 5:
        title, summary = summarize_from_children(cur)
        root = SummaryNode(
            level=cur[0].level + 1,
            index=0,
            start_timestamp=cur[0].start_timestamp,
            end_timestamp=cur[-1].end_timestamp,
            title=title,
            summary=summary,
            children=cur,
        )
        levels_meta.append({"level": root.level, "count": 1})
        roots = [root]
    else:
        roots = cur
    return {"levels": levels_meta, "roots": [n.to_json() for n in roots]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPT chunk trees from paper/data_avtext_fixed.")
    parser.add_argument("--source-dir", type=str, required=True, help="Chunk folder (e.g., paper/data_avtext_fixed/audio_video_chunk_level_gpt)")
    parser.add_argument("--out-dir", type=str, default="", help="Output folder (default: <source>_tree)")
    parser.add_argument("--lectures", type=str, default="", help="Comma-separated lecture ids to process.")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--reduction-factor", type=float, default=4.0)
    parser.add_argument("--gpt-chunk-size", type=int, default=16)
    parser.add_argument("--gpt-min-duration-sec", type=float, default=20.0)
    parser.add_argument("--env", type=str, default=str(DEFAULT_ENV))
    parser.add_argument("--print-generations", action="store_true", help="Print title/summary for every GPT call.")
    args = parser.parse_args()

    _load_env(Path(args.env))
    global PRINT_GENERATIONS
    PRINT_GENERATIONS = bool(args.print_generations)

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    out_dir = Path(args.out_dir) if args.out_dir else source_dir.parent / f"{source_dir.name}_tree"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lectures.strip():
        lectures = [t.strip() for t in args.lectures.split(",") if t.strip()]
    else:
        lectures = [p.stem.replace("lecture", "") for p in source_dir.glob("lecture*.json")]

    for lec in sorted(lectures, key=lambda x: int(x) if x.isdigit() else x):
        path = source_dir / f"lecture{lec}.json"
        if not path.exists():
            print(f"❌ Missing {path}")
            continue
        chunks = _load_chunks(path)
        tree = _build_tree_for_lecture(
            chunks,
            min_nodes=int(args.min_nodes),
            reduction_factor=float(args.reduction_factor),
            gpt_chunk_size=int(args.gpt_chunk_size),
            gpt_min_duration_sec=float(args.gpt_min_duration_sec),
        )
        out_path = out_dir / f"lecture{lec}.json"
        payload = {"lecture_id": lec, **tree}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()
