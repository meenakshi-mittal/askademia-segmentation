#!/usr/bin/env python3
"""
Build alternative summary trees from chunk folders in paper/data_avtext_fixed.

Pipeline:
  1) Leaf level: each chunk -> SummaryNode with GPT title+summary.
  2) Higher levels: build parent nodes using a chosen strategy:
     - agglomerative: merge most similar adjacent summaries until target count.
     - cosine_threshold: split by cosine similarity threshold on summaries.
     - fixed_even: split into evenly sized contiguous groups.
  3) Repeat until <= --min-nodes.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = REPO_ROOT / "config" / "keys.env"

OPENAI_MINI_ENDPOINT = None
OPENAI_API_KEY = None

MAX_RETRIES = 6
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
        val = 1.0
    if val < -1.0:
        val = -1.0
    return val


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
) -> Tuple[str, str]:
    if not OPENAI_MINI_ENDPOINT or not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_MINI_ENDPOINT or OPENAI_API_KEY in environment.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

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
        "7) The title must cover the ENTIRE segment, reflecting ALL major child topics, not just the first bullet."
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            body = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 200,
            }
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


def _create_overlapping_chunks(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    step_size = max(1, chunk_size * 7 // 8)  # 25% overlap
    for i in range(0, len(items), step_size):
        chunk = items[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if i + chunk_size >= len(items):
            break
    return chunks


def _extract_valid_timestamps(chunk: List[Dict[str, Any]]) -> List[float]:
    valid: List[float] = []
    for entry in chunk:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        try:
            valid.append(float(ts))
        except Exception:
            continue
    return sorted(set(valid))


def _build_prompt_for_shift_chunk(
    chunk: List[Dict[str, Any]],
    chunk_index: int,
    total_chunks: int,
    previous_shifts: Optional[List[float]] = None,
) -> str:
    chunk_text = "\n".join(
        [f"[{float(s.get('timestamp', 0)):.2f}s] {str(s.get('text','')).strip()}" for s in chunk]
    )
    prev_section = ""
    if previous_shifts:
        prev_str = ", ".join(f"{ts:.2f}s" for ts in previous_shifts)
        prev_section = (
            "\n\nPREVIOUS TOPIC SHIFTS (from the immediately preceding chunk):\n"
            f"{prev_str}\n"
            "These timestamps mark where the lecture has ALREADY been segmented. Avoid reselecting these same timestamps. Use them to determine future shifts.\n"
        )
    prompt = f"""You are analyzing lecture content to detect topic shifts. You will be given chunks of lecture audio with timestamps and need to identify where topic shifts occur.

The timestamps you see below are ABSOLUTE lecture timestamps (in seconds from the start of the lecture).{prev_section}

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
                # Fallback: extract floats
                nums = re.findall(r"\d+\.\d+|\d+", content)
                return [float(n) for n in nums]
        except Exception as e:
            print(f"⚠️  API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
    return []


def _filter_shifts_to_valid(
    shifts: List[float],
    valid_ts: List[float],
    *,
    tolerance: float = 0.1,
) -> List[float]:
    if not shifts:
        return []
    valid = []
    for s in shifts:
        for v in valid_ts:
            if abs(s - v) <= tolerance:
                valid.append(v)
                break
    return sorted(set(valid))


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


def _flatten_nodes(items: List[Any]) -> List[SummaryNode]:
    out: List[SummaryNode] = []
    for it in items:
        if isinstance(it, SummaryNode):
            out.append(it)
        elif isinstance(it, list):
            for sub in it:
                if isinstance(sub, SummaryNode):
                    out.append(sub)
    return out


def _strategy_from_method(name: str) -> str:
    n = name.lower()
    if "chunk_level_gpt" in n:
        return "gpt_chunking"
    if "agglomerative" in n:
        return "agglomerative"
    if "cosine_threshold" in n:
        return "cosine_threshold"
    if "fixed" in n:
        return "fixed_even"
    return "agglomerative"


def _build_leaf_nodes(chunks: List[Dict[str, Any]], *, level: int = 0, print_generations: bool = False) -> List[SummaryNode]:
    nodes: List[SummaryNode] = []
    for idx, ch in enumerate(chunks):
        text = str(ch.get("text", "") or "").strip()
        start_ms = int(ch.get("start_ms") or 0)
        end_ms = int(ch.get("end_ms") or start_ms)
        start_ts = start_ms / 1000.0
        end_ts = end_ms / 1000.0
        title, summary = call_gpt4o_mini_for_title_summary(text, start_ts=start_ts, end_ts=end_ts)
        if print_generations:
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


def _nodes_to_texts(nodes: List[SummaryNode]) -> List[str]:
    return [f"{n.title}: {n.summary}".strip() for n in nodes]


def _agglomerative_groups(
    nodes: List[SummaryNode],
    *,
    target_count: int,
    max_children_per_parent: int,
    tiny_max_children: int,
    tiny_max_combined: int,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> List[List[SummaryNode]]:
    # Ensure strict time order so only adjacent nodes can merge.
    nodes = sorted(nodes, key=lambda n: n.start_timestamp)
    if len(nodes) <= target_count:
        return [[n] for n in nodes]
    texts = _nodes_to_texts(nodes)
    embs = _compute_embeddings(emb_client, texts, model=emb_model, batch_size=emb_batch_size)

    class Tmp:
        def __init__(self, node: SummaryNode, emb: np.ndarray):
            self.nodes = [node]
            self.emb = emb
        @property
        def count(self) -> int:
            return len(self.nodes)

    cur = [Tmp(n, e) for n, e in zip(nodes, embs)]

    while len(cur) > target_count:
        best_i = -1
        best_sim = -1.0
        for i in range(len(cur) - 1):
            a = cur[i]
            b = cur[i + 1]
            if max_children_per_parent > 0 and (a.count + b.count) > max_children_per_parent:
                continue
            sim = _cosine(a.emb, b.emb)
            if sim > best_sim:
                best_sim = sim
                best_i = i
        if best_i < 0:
            break
        left = cur[best_i]
        right = cur[best_i + 1]
        total = max(1, left.count + right.count)
        merged_emb = (left.emb * left.count + right.emb * right.count) / float(total)
        merged = Tmp(left.nodes + right.nodes, merged_emb.astype(np.float32))
        cur[best_i : best_i + 2] = [merged]

    # tiny merge (<=1) with neighbor, allow combined up to tiny_max_combined
    pre_tiny = list(cur)
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(cur):
            if ch.count > tiny_max_children:
                continue
            candidates: List[Tuple[float, int]] = []
            if i - 1 >= 0:
                left = cur[i - 1]
                if tiny_max_combined <= 0 or (left.count + ch.count) <= tiny_max_combined:
                    candidates.append((_cosine(left.emb, ch.emb), i - 1))
            if i + 1 < len(cur):
                right = cur[i + 1]
                if tiny_max_combined <= 0 or (right.count + ch.count) <= tiny_max_combined:
                    candidates.append((_cosine(ch.emb, right.emb), i))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, merge_at = candidates[0]
            left = cur[merge_at]
            right = cur[merge_at + 1]
            total = max(1, left.count + right.count)
            merged_emb = (left.emb * left.count + right.emb * right.count) / float(total)
            merged = Tmp(left.nodes + right.nodes, merged_emb.astype(np.float32))
            cur[merge_at : merge_at + 2] = [merged]
            changed = True
            break

    # If tiny-merge would collapse below target_count, keep pre-tiny result.
    if target_count > 1 and len(cur) < target_count:
        cur = pre_tiny

    return [c.nodes for c in cur]


def _cosine_threshold_groups(
    nodes: List[SummaryNode],
    *,
    target_min: int,
    target_max: int,
    window_size: int,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> List[List[SummaryNode]]:
    n = len(nodes)
    if n <= target_max:
        return [[node] for node in nodes]
    texts = _nodes_to_texts(nodes)
    embs = _compute_embeddings(emb_client, texts, model=emb_model, batch_size=emb_batch_size)
    k = max(1, int(window_size))
    boundary_sims: List[float] = [1.0] * max(0, n - 1)
    if n >= 2 * k:
        for i in range(0, n - 2 * k + 1):
            left = embs[i : i + k]
            right = embs[i + k : i + 2 * k]
            left_mean = np.mean(left, axis=0)
            right_mean = np.mean(right, axis=0)
            sim = _cosine(left_mean, right_mean)
            boundary_idx = i + k - 1
            boundary_sims[boundary_idx] = sim

    sorted_sims = sorted(boundary_sims)
    def count_chunks(threshold: float) -> int:
        splits = sum(1 for s in boundary_sims if s < threshold)
        return splits + 1

    chosen = None
    best_diff = 10**9
    for thr in sorted_sims:
        c = count_chunks(thr)
        if target_min <= c <= target_max:
            chosen = thr
            break
        diff = abs(c - ((target_min + target_max) / 2.0))
        if diff < best_diff:
            best_diff = diff
            chosen = thr

    # split by chosen threshold
    groups: List[List[SummaryNode]] = []
    cur: List[SummaryNode] = [nodes[0]]
    for i in range(1, n):
        if boundary_sims[i - 1] < chosen:
            groups.append(cur)
            cur = []
        cur.append(nodes[i])
    if cur:
        groups.append(cur)
    return groups


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


def _gpt_chunk_groups(
    nodes: List[SummaryNode],
    *,
    chunk_size: int,
    min_duration_sec: float,
) -> List[List[SummaryNode]]:
    if not nodes:
        return []
    items = [{"timestamp": n.start_timestamp, "text": f"{n.title}: {n.summary}"} for n in nodes]
    chunks = _create_overlapping_chunks(items, chunk_size)
    all_shifts: List[float] = []
    prev_shifts: List[float] = []
    for i, ch in enumerate(chunks):
        prompt = _build_prompt_for_shift_chunk(ch, i, len(chunks), previous_shifts=prev_shifts)
        raw = _call_gpt4o_mini_for_shifts(prompt)
        valid_ts = _extract_valid_timestamps(ch)
        filtered = _filter_shifts_to_valid(raw, valid_ts, tolerance=0.1)
        all_shifts.extend(filtered)
        prev_shifts = filtered

    # Split nodes by shifts
    all_shifts = sorted(set(all_shifts))
    groups: List[List[SummaryNode]] = []
    cur: List[SummaryNode] = []
    shift_idx = 0
    for node in nodes:
        ts = float(node.start_timestamp)
        while shift_idx < len(all_shifts) and ts >= all_shifts[shift_idx] - 0.1:
            if cur:
                start_ts = float(cur[0].start_timestamp)
                duration = ts - start_ts
                if duration >= min_duration_sec:
                    groups.append(cur)
                    cur = []
            shift_idx += 1
        cur.append(node)
    if cur:
        groups.append(cur)

    # If the last group is too short, merge with previous
    if len(groups) >= 2 and min_duration_sec > 0:
        last = groups[-1]
        start_ts = float(last[0].start_timestamp)
        end_ts = float(last[-1].end_timestamp)
        if (end_ts - start_ts) < min_duration_sec:
            groups[-2].extend(last)
            groups.pop()
    return groups


def _build_next_level(
    nodes: List[SummaryNode],
    *,
    strategy: str,
    target_count: int,
    target_min: int,
    target_max: int,
    max_children: int,
    tiny_max_children: int,
    tiny_max_combined: int,
    window_size: int,
    gpt_chunk_size: int,
    gpt_min_duration_sec: float,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
    next_level: int,
    print_generations: bool,
) -> List[SummaryNode]:
    if strategy == "gpt_chunking":
        groups = _gpt_chunk_groups(
            nodes,
            chunk_size=gpt_chunk_size,
            min_duration_sec=gpt_min_duration_sec,
        )
    elif strategy == "agglomerative":
        groups = _agglomerative_groups(
            nodes,
            target_count=target_count,
            max_children_per_parent=max_children,
            tiny_max_children=tiny_max_children,
            tiny_max_combined=tiny_max_combined,
            emb_client=emb_client,
            emb_model=emb_model,
            emb_batch_size=emb_batch_size,
        )
    elif strategy == "cosine_threshold":
        groups = _cosine_threshold_groups(
            nodes,
            target_min=target_min,
            target_max=target_max,
            window_size=window_size,
            emb_client=emb_client,
            emb_model=emb_model,
            emb_batch_size=emb_batch_size,
        )
    else:
        groups = _fixed_even_groups(nodes, target_count=target_count)

    new_nodes: List[SummaryNode] = []
    for idx, group in enumerate(groups):
        flat_group = _flatten_nodes(group)
        if not flat_group:
            continue
        title, summary = summarize_from_children(flat_group)
        if print_generations:
            child_ids = ",".join(str(n.index) for n in flat_group)
            print(f"   merging {child_ids}: title: {title} summary: {summary}")
        new_nodes.append(
            SummaryNode(
                level=next_level,
                index=idx,
                start_timestamp=flat_group[0].start_timestamp,
                end_timestamp=flat_group[-1].end_timestamp,
                title=title,
                summary=summary,
                children=flat_group,
            )
        )
    return new_nodes


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


def _build_tree_for_lecture(
    chunks: List[Dict[str, Any]],
    *,
    strategy: str,
    min_nodes: int,
    reduction_factor: float,
    target_fraction_low: float,
    target_fraction_high: float,
    max_children: int,
    tiny_max_children: int,
    tiny_max_combined: int,
    window_size: int,
    gpt_chunk_size: int,
    gpt_min_duration_sec: float,
    print_generations: bool,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> Dict[str, Any]:
    level = 0
    cur = _build_leaf_nodes(chunks, level=level, print_generations=print_generations)
    cur.sort(key=lambda n: n.start_timestamp)
    levels_meta = [{"level": level, "count": len(cur)}]
    while len(cur) > min_nodes:
        level += 1
        target_count = max(1, int(round(len(cur) / max(1.0, reduction_factor))))
        if strategy == "cosine_threshold":
            target_min = target_count
            target_max = target_count
        else:
            target_min = max(1, int(math.floor(len(cur) * target_fraction_low)))
            target_max = max(1, int(math.ceil(len(cur) * target_fraction_high)))
        cur = _build_next_level(
            cur,
            strategy=strategy,
            target_count=target_count,
            target_min=target_min,
            target_max=target_max,
            max_children=max_children,
            tiny_max_children=tiny_max_children,
            tiny_max_combined=tiny_max_combined,
            window_size=window_size,
            gpt_chunk_size=gpt_chunk_size,
            gpt_min_duration_sec=gpt_min_duration_sec,
            print_generations=print_generations,
            emb_client=emb_client,
            emb_model=emb_model,
            emb_batch_size=emb_batch_size,
            next_level=level,
        )
        cur.sort(key=lambda n: n.start_timestamp)
        levels_meta.append({"level": level, "count": len(cur)})
        if len(cur) == 1:
            break
    return {"levels": levels_meta, "roots": [n.to_json() for n in cur]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build alternative chunk trees from data_avtext_fixed.")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "paper" / "data_avtext_fixed"))
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method folder names or 'all'.")
    parser.add_argument("--out-dir", type=str, default="", help="If set, write all method trees under this dir.")
    parser.add_argument("--out-suffix", type=str, default="_tree", help="Suffix for per-method output folders.")
    parser.add_argument("--min-nodes", type=int, default=5, help="Stop when level size <= this.")
    parser.add_argument("--reduction-factor", type=float, default=4.0, help="Target count = len/this per level.")
    parser.add_argument("--target-frac-low", type=float, default=0.25, help="Cosine threshold target min fraction.")
    parser.add_argument("--target-frac-high", type=float, default=0.35, help="Cosine threshold target max fraction.")
    parser.add_argument("--max-children", type=int, default=6, help="Agglomerative max children per parent.")
    parser.add_argument("--tiny-merge-max-children", type=int, default=1)
    parser.add_argument("--tiny-merge-max-combined", type=int, default=7)
    parser.add_argument("--cosine-window-size", type=int, default=1)
    parser.add_argument("--gpt-chunk-size", type=int, default=16)
    parser.add_argument("--gpt-min-duration-sec", type=float, default=20.0)
    parser.add_argument("--print-generations", action="store_true", help="Print title/summary for each merge.")
    parser.add_argument("--env", type=str, default=str(DEFAULT_ENV))
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--embedding-endpoint", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    args = parser.parse_args()

    _load_env(Path(args.env))
    emb_client = _build_embedding_client(
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        embedding_endpoint=args.embedding_endpoint,
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.methods.strip().lower() == "all":
        methods = [d.name for d in data_dir.iterdir() if d.is_dir() and "sentence_level" not in d.name]
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for method in methods:
        method_dir = data_dir / method
        if not method_dir.exists():
            print(f"⚠️  Skipping missing method dir: {method_dir}")
            continue
        strategy = _strategy_from_method(method)
        method_out = (out_dir / method) if out_dir else (data_dir / f"{method}{args.out_suffix}")
        method_out.mkdir(parents=True, exist_ok=True)
        print(f"\n🧭 Method {method} -> strategy={strategy}")
        for path in sorted(method_dir.glob("lecture*.json")):
            lec = path.stem.replace("lecture", "")
            chunks = _load_chunks(path)
            tree = _build_tree_for_lecture(
                chunks,
                strategy=strategy,
                min_nodes=int(args.min_nodes),
                reduction_factor=float(args.reduction_factor),
                target_fraction_low=float(args.target_frac_low),
                target_fraction_high=float(args.target_frac_high),
                max_children=int(args.max_children),
                tiny_max_children=int(args.tiny_merge_max_children),
                tiny_max_combined=int(args.tiny_merge_max_combined),
                window_size=int(args.cosine_window_size),
                gpt_chunk_size=int(args.gpt_chunk_size),
                gpt_min_duration_sec=float(args.gpt_min_duration_sec),
                print_generations=bool(args.print_generations),
                emb_client=emb_client,
                emb_model=args.model,
                emb_batch_size=int(args.batch_size),
            )
            out_path = method_out / f"lecture{lec}.json"
            payload = {"lecture_id": lec, **tree}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   💾 wrote {out_path}")


if __name__ == "__main__":
    main()

