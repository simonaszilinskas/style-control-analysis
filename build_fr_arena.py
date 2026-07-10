#!/usr/bin/env python3
"""
Build the comparia-fr-arena robustness battle table (fr_arena_battles.parquet).

Streams the gated HuggingFace dataset ministere-culture/comparia-fr-arena
(column-projected range reads, so the 8.9 GB file is not fully downloaded), keeps
decisive French turns, and computes the same feature set the main analysis uses:
markdown headers/lists/bold/code_blocks/emoji (identical regexes to
clean_and_analyze.py), length as the output-token count, and the readability /
diversity / structure features of the companion linguistic analysis. One row per
battle, schema-compatible with battles_bt_styled + linguistic_features.

Needs a HF token with access to the gated dataset:
    HF_TOKEN=hf_... python build_fr_arena.py
"""

import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import textstat
from huggingface_hub import HfFileSystem

HF_PATH = "datasets/ministere-culture/comparia-fr-arena/comparia-fr-arena.parquet"
OUT = "fr_arena_battles.parquet"
COLS = ["choice", "model_a", "model_b", "response_a", "response_b", "metadata",
        "comparison_id"]
MIN_WORDS = 30

_HEADER_RE = re.compile(r"^ {0,3}#{1,6}\s", re.MULTILINE)
_LIST_RE = re.compile(r"^ {0,3}([-*+]|\d+[.)])\s", re.MULTILINE)
_CODE_RE = re.compile(r"```|~~~")
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    "\U00002700-\U000027BF]+", flags=re.UNICODE)
_WORD_RE = re.compile(r"[\w']+")
_SENT_RE = re.compile(r"[.!?]+")
_VOWELS = "aeiouyàâäéèêëîïôùûüœæ"

FEATS = ["headers", "lists", "bold", "code_blocks", "emoji", "length",
         "rel", "cli", "fkg", "ttr", "mattr", "asl", "long_sent_ratio"]


def _syllables_fr(word):
    c, prev = 0, False
    for ch in word.lower():
        v = ch in _VOWELS
        if v and not prev:
            c += 1
        prev = v
    return max(1, c)


def features(content, tokens):
    out = {k: np.nan for k in FEATS}
    if content:
        out["headers"] = float(len(_HEADER_RE.findall(content)))
        out["lists"] = float(len(_LIST_RE.findall(content)))
        out["bold"] = float(content.count("**") // 2)
        out["code_blocks"] = float(len(_CODE_RE.findall(content)) // 2)
        out["emoji"] = float(len(_EMOJI_RE.findall(content)))
    else:
        for k in ("headers", "lists", "bold", "code_blocks", "emoji"):
            out[k] = 0.0
    out["length"] = float(tokens) if tokens and tokens > 0 else np.nan
    if not content:
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


def _assistant(msgs):
    if not msgs:
        return None
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def main():
    fs = HfFileSystem(token=os.environ["HF_TOKEN"])
    pf = pq.ParquetFile(fs.open(HF_PATH, "rb"))
    rows_out, kept, seen, t0 = [], 0, 0, time.time()
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=COLS).to_pylist():
            seen += 1
            if r["choice"] not in ("a_better", "b_better"):
                continue
            md = r["metadata"] or {}
            if "fr" not in (md.get("languages") or []):
                continue
            ca, cb = _assistant(r["response_a"]), _assistant(r["response_b"])
            if not ca or not cb:
                continue
            fa = features(ca, md.get("tokens_a"))
            fb = features(cb, md.get("tokens_b"))
            ma, mb = r["model_a"], r["model_b"]
            rec = {"comparison_id": r["comparison_id"], "model_a": ma, "model_b": mb,
                   "winner": ma if r["choice"] == "a_better" else mb}
            for k in FEATS:
                rec[f"{k}_a"], rec[f"{k}_b"] = fa[k], fb[k]
            rows_out.append(rec)
            kept += 1
        print(f"  rg {rg+1}/{pf.num_row_groups}  seen={seen:,} kept={kept:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows_out)
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"Wrote {len(df):,} decisive French battles -> {OUT}")


if __name__ == "__main__":
    main()
