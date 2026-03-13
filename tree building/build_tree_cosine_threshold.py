#!/usr/bin/env python3
"""
Build cosine-threshold summary trees from chunk files.

Identical flow to build_tree_gpt.py, except combining method:
  - Leaf GPT summaries.
  - Embed "<title>: <summary>" and split by cosine threshold to reach
    target_count = len/4 per level (closest achievable).
  - GPT summaries for each merged group.
"""

from __future__ import annotations

import argparse
import json
import os
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
    OPENAI_MINI_ENDPOINT = os.getenv("OPENAI_MINI_ENDPOINT") or os.getenv("OPENAI_MINI_ENDPOINT_2")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _build_embedding_client(
    *,
    provider: str,
    azure_endpoint: Optional[str],
    api_version: Optional[str],
    embedding_endpoint: Optional[str],
) -> OpenAI:
    env_embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT_2") or os.getenv("EMBEDDING_ENDPOINT")
    env_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("azure_endpoint")
    env_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("api_version")

    azure_endpoint = embedding_endpoint or azure_endpoint or env_embedding_endpoint or env_azure_endpoint
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


def _cosine_threshold_groups(
    nodes: List[SummaryNode],
    *,
    target_count: int,
    window_size: int,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> List[List[SummaryNode]]:
    nodes = sorted(nodes, key=lambda n: n.start_timestamp)
    n = len(nodes)
    if n <= target_count:
        return [[n] for n in nodes]
    texts = [f"{n.title}: {n.summary}".strip() for n in nodes]
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
        if c == target_count:
            chosen = thr
            break
        diff = abs(c - target_count)
        if diff < best_diff:
            best_diff = diff
            chosen = thr

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


def _build_tree_for_lecture(
    chunks: List[Dict[str, Any]],
    *,
    min_nodes: int,
    reduction_factor: float,
    window_size: int,
    emb_client: OpenAI,
    emb_model: str,
    emb_batch_size: int,
) -> Dict[str, Any]:
    level = 0
    cur = _build_leaf_nodes(chunks, level=level)
    cur.sort(key=lambda n: n.start_timestamp)
    levels_meta = [{"level": level, "count": len(cur)}]
    while len(cur) > min_nodes:
        prev_len = len(cur)
        level += 1
        target_count = max(1, int(round(len(cur) / max(1.0, reduction_factor))))
        groups = _cosine_threshold_groups(
            cur,
            target_count=target_count,
            window_size=window_size,
            emb_client=emb_client,
            emb_model=emb_model,
            emb_batch_size=emb_batch_size,
        )
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
    parser = argparse.ArgumentParser(description="Build cosine-threshold trees from paper/data_avtext_fixed.")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="", help="Output folder (default: <source>_tree)")
    parser.add_argument("--lectures", type=str, default="")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--reduction-factor", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--print-generations", action="store_true", help="Print title/summary for every GPT call.")
    parser.add_argument("--env", type=str, default=str(DEFAULT_ENV))
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--embedding-endpoint", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    args = parser.parse_args()

    _load_env(Path(args.env))
    global PRINT_GENERATIONS
    PRINT_GENERATIONS = bool(args.print_generations)

    emb_client = _build_embedding_client(
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        embedding_endpoint=args.embedding_endpoint,
    )

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
        out_path = out_dir / f"lecture{lec}.json"
        if out_path.exists():
            print(f"⏭️  Skipping {out_path} (already exists)")
            continue
        chunks = _load_chunks(path)
        tree = _build_tree_for_lecture(
            chunks,
            min_nodes=int(args.min_nodes),
            reduction_factor=float(args.reduction_factor),
            window_size=int(args.window_size),
            emb_client=emb_client,
            emb_model=args.model,
            emb_batch_size=int(args.batch_size),
        )
        payload = {"lecture_id": lec, **tree}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()
