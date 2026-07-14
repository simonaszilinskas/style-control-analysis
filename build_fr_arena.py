#!/usr/bin/env python3
"""
Build the compar:IA battle table from comparia-fr-arena (the consolidated release).

comparia-fr-arena is turn-level: one row per turn, and `choice` (a_better /
b_better) is populated on the single turn where the user voted. We keep those
decisive French votes, take the full conversation the voter saw at that point,
concatenate its assistant messages, and compute the full feature set on it.

One row per comparison (battle):
  conversation_pair_id, model_a_name, model_b_name, winner (model_a/model_b),
  source='vote', mode, conv_turns (user-message count), primary_topic,
  {formatting}_a/_b, {linguistic}_a/_b, length_a/_b (cumulative output tokens).

Needs a HF token (CLI login or HF_TOKEN) for the gated dataset.
    python build_fr_arena.py            # -> fr_battles.parquet
"""

import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import textstat
from huggingface_hub import HfFileSystem, get_token

HF_PATH = "datasets/ministere-culture/comparia-fr-arena/comparia-fr-arena.parquet"
# A full local copy avoids streaming the 8.9 GB gated file; use it if present.
LOCAL_PATHS = [
    "comparia-fr-arena.parquet",
    os.path.expanduser("~/Dev/comparia-theme-datasets/data/comparia-fr-arena/comparia-fr-arena.parquet"),
]
OUT = "fr_battles.parquet"
COLS = ["comparison_id", "choice", "model_a", "model_b",
        "full_conversation_a", "full_conversation_b", "metadata"]
MIN_WORDS = 30

_HEADER_RE = re.compile(r"^ {0,3}#{1,6}\s", re.MULTILINE)
_LIST_RE = re.compile(r"^ {0,3}([-*+]|\d+[.)])\s", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
_CODE_RE = re.compile(r"```|~~~")
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    "\U00002700-\U000027BF]+", flags=re.UNICODE)
_WORD_RE = re.compile(r"[\w']+")
_SENT_RE = re.compile(r"[.!?]+")
_VOWELS = "aeiouyàâäéèêëîïôùûüœæ"

FORMATTING = ["headers", "lists", "bold", "code_blocks", "emoji"]
LINGUISTIC = ["rel", "cli", "fkg", "ttr", "mattr", "asl", "long_sent_ratio"]
FEATS = FORMATTING + LINGUISTIC


def _syllables_fr(word):
    c, prev = 0, False
    for ch in word.lower():
        v = ch in _VOWELS
        if v and not prev:
            c += 1
        prev = v
    return max(1, c)


def _assistant_text(msgs):
    """Concatenate all assistant messages (the cumulative text the voter saw)."""
    if msgs is None:
        return ""
    parts = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if "<think>" not in c and "</think>" not in c:
                parts.append(c)
    return "\n".join(parts)


def _user_turns(msgs):
    if msgs is None:
        return 0
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")


def features(content):
    out = {k: np.nan for k in FEATS}
    if content:
        out["headers"] = float(len(_HEADER_RE.findall(content)))
        out["lists"] = float(len(_LIST_RE.findall(content)))
        out["bold"] = float(len(_BOLD_RE.findall(content)))
        out["code_blocks"] = float(len(_CODE_RE.findall(content)) // 2)
        out["emoji"] = float(len(_EMOJI_RE.findall(content)))
    else:
        for k in FORMATTING:
            out[k] = 0.0
        return out
    words = content.split()
    if len(words) < MIN_WORDS:
        return out
    try:
        sents = max(1, textstat.sentence_count(content))
        syl = sum(_syllables_fr(w) for w in words)
        out["rel"] = 207 - 1.015 * (len(words) / sents) - 73.6 * (syl / len(words))
        out["cli"] = textstat.coleman_liau_index(content)
        out["fkg"] = textstat.flesch_kincaid_grade(content)
    except Exception:
        pass
    toks = _WORD_RE.findall(content.lower())
    if len(toks) >= MIN_WORDS:
        out["ttr"] = len(set(toks)) / len(toks)
    if len(toks) >= 50:
        out["mattr"] = float(np.mean([len(set(toks[i:i + 50])) / 50
                                      for i in range(len(toks) - 49)]))
    sents = [s for s in _SENT_RE.split(content) if s.split()]
    if sents:
        lens = [len(s.split()) for s in sents]
        out["asl"] = float(np.mean(lens))
        out["long_sent_ratio"] = float(np.mean([1.0 if l > 25 else 0.0 for l in lens]))
    return out


PARTS = "fr_parts"


def _process_rows(pyrows):
    out = []
    for r in pyrows:
        if r["choice"] not in ("a_better", "b_better"):
            continue
        md = r["metadata"] or {}
        if "fr" not in (md.get("languages") or []):
            continue
        ta = _assistant_text(r["full_conversation_a"])
        tb = _assistant_text(r["full_conversation_b"])
        if not ta or not tb:
            continue
        cats = md.get("categories")
        rec = {
            "conversation_pair_id": r["comparison_id"],
            "model_a_name": r["model_a"], "model_b_name": r["model_b"],
            "winner": "model_a" if r["choice"] == "a_better" else "model_b",
            "source": "vote", "mode": md.get("mode"),
            "conv_turns": _user_turns(r["full_conversation_a"]),
            "primary_topic": cats[0] if cats is not None and len(cats) > 0 else None,
            "length_a": float(md["total_tokens_a"]) if md.get("total_tokens_a") else np.nan,
            "length_b": float(md["total_tokens_b"]) if md.get("total_tokens_b") else np.nan,
        }
        fa, fb = features(ta), features(tb)
        for k in FEATS:
            rec[f"{k}_a"], rec[f"{k}_b"] = fa[k], fb[k]
        out.append(rec)
    return out


def _open():
    for p in LOCAL_PATHS:
        if os.path.exists(p):
            return pq.ParquetFile(p)
    return pq.ParquetFile(HfFileSystem(token=get_token()).open(HF_PATH, "rb"))


def main():
    os.makedirs(PARTS, exist_ok=True)
    pf = _open()
    n_rg = pf.num_row_groups
    t0 = time.time()
    # Stream row group by row group, checkpoint each to its own part, resume on
    # restart, and retry a row group on a dropped connection (the 8.9 GB stream
    # is fragile and disk is too small to download the whole file).
    for rg in range(n_rg):
        part = os.path.join(PARTS, f"part_{rg:03d}.parquet")
        if os.path.exists(part):
            continue
        for attempt in range(8):
            try:
                pyrows = pf.read_row_group(rg, columns=COLS).to_pylist()
                rows = _process_rows(pyrows)
                pd.DataFrame(rows).to_parquet(part, index=False, compression="zstd")
                print(f"  rg {rg+1}/{n_rg} kept={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  rg {rg+1} attempt {attempt+1} failed: {type(e).__name__}; "
                      f"reopening, retry in {wait}s", flush=True)
                time.sleep(wait)
                pf = _open()
        else:
            raise RuntimeError(f"row group {rg} failed after retries")

    parts = [pd.read_parquet(os.path.join(PARTS, f)) for f in sorted(os.listdir(PARTS))
             if f.endswith(".parquet")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    # A few comparisons carry more than one decisive vote (voted at different
    # turns); keep one battle per comparison (the last, i.e. most recent vote).
    n0 = len(df)
    df = df.drop_duplicates(subset="conversation_pair_id", keep="last").reset_index(drop=True)
    print(f"  deduplicated {n0-len(df)} repeated comparison ids")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nWrote {len(df):,} decisive French battles -> {OUT}")
    print(f"  models: {pd.concat([df.model_a_name, df.model_b_name]).nunique()}")
    print(f"  with topic: {df.primary_topic.notna().mean()*100:.1f}%")
    print(f"  multi-turn: {(df.conv_turns>=2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
