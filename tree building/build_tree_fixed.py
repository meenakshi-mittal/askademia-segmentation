#!/usr/bin/env python3
"""
Build fixed-interval summary trees from chunk files.
At each level, target group count = len/4 (reduction factor).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def call_gpt4o_mini_for_title_summary(
    text: str,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> tuple[str, str]:
    if not OPENAI_MINI_ENDPOINT or not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_MINI_ENDPOINT or OPENAI_API_KEY in environment.")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    snippet = text.strip()
    if len(snippet) > 2000:
        snippet = snippet[:2000] + " ..."

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
    user_prompt = (
        "LECTURE SEGMENT:\n"
        "----------------\n"
        f"{time_context}{snippet}\n\n"
        "Respond with a JSON object of the form:\n"
        '{"title": "...", "summary": "..."}\n'
        "Do not include any additional text, Markdown, or explanation."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            body = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.4, "max_tokens": 1200}
            resp = requests.post(OPENAI_MINI_ENDPOINT, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
            content = (resp.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            try:
                data = json.loads(content)
                title = str(data.get("title", "")).strip() or "Untitled segment"
                summary = str(data.get("summary", "")).strip() or (snippet[:200] + ("..." if len(snippet) > 200 else ""))
                return title, summary
            except Exception:
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                if lines:
                    title = lines[0][:80]
                    summary = " ".join(lines[1:]) or snippet[:200]
                    return title, summary
        except Exception as e:
            print(f"⚠️  Failed to summarize segment (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    fallback = snippet[:200] + ("..." if len(snippet) > 200 else "")
    return "Untitled segment", fallback


def summarize_from_children(children: List[SummaryNode]) -> tuple[str, str]:
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
        title, summary = call_gpt4o_mini_for_title_summary(text, start_ts=start_ts, end_ts=end_ts)
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


def _fixed_even_groups(nodes: List[SummaryNode], *, target_count: int) -> List[List[SummaryNode]]:
    if target_count <= 0:
        return [[n] for n in nodes]
    n = len(nodes)
    if n <= target_count:
        return [[n] for n in nodes]
    group_size = int(math.ceil(n / float(target_count)))
    groups: List[List[SummaryNode]] = []
    for i in range(0, n, group_size):
        groups.append(nodes[i : i + group_size])
    return groups


def _build_tree_for_lecture(
    chunks: List[Dict[str, Any]],
    *,
    min_nodes: int,
    reduction_factor: float,
) -> Dict[str, Any]:
    level = 0
    cur = _build_leaf_nodes(chunks, level=level)
    cur.sort(key=lambda n: n.start_timestamp)
    levels_meta = [{"level": level, "count": len(cur)}]
    while len(cur) > min_nodes:
        prev_len = len(cur)
        level += 1
        target_count = max(1, int(round(len(cur) / max(1.0, reduction_factor))))
        groups = _fixed_even_groups(cur, target_count=target_count)
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
    parser = argparse.ArgumentParser(description="Build fixed-interval trees from paper/data_avtext_fixed.")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="", help="Output folder (default: <source>_tree)")
    parser.add_argument("--lectures", type=str, default="")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--reduction-factor", type=float, default=4.0)
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
        )
        out_path = out_dir / f"lecture{lec}.json"
        payload = {"lecture_id": lec, **tree}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()
