#!/usr/bin/env python3
"""
Alternative lexical-diversity metrics for the MATTR stress test (§4.10, part 2).

Recomputes, per response, three checks on whether MATTR really captures lexical
diversity rather than named entities / technical terms:
  - MTLD (measure of textual lexical diversity), a window-free alternative;
  - content-word MATTR (French stopwords removed);
  - MATTR excluding capitalised tokens (a rough proper-noun exclusion).

Streams comparia-fr-arena (local copy if present), same decisive-French filter as
build_fr_arena.py, checkpointed per row group. -> data/mattr_alt.parquet
"""

import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, get_token

from paths import DATA
from build_fr_arena import (HF_PATH, HF_REVISION, LOCAL_PATHS, _assistant_text,
                            _conversation_prefix)

OUT = DATA / "mattr_alt.parquet"
PARTS = DATA / f"mattr_alt_parts_vote_truncated_{HF_REVISION[:8]}"
COLS = ["comparison_id", "choice", "turn", "full_conversation_a", "full_conversation_b", "metadata"]
_WORD = re.compile(r"[\w']+")
# small French + English stopword set (enough to strip function words for content-word MATTR)
STOP = set("le la les un une des du de d au aux et ou mais donc or ni car que qui quoi dont ou "
           "a à en dans sur sous par pour avec sans ce cet cette ces mon ton son notre votre leur "
           "je tu il elle on nous vous ils elles se me te lui y est sont suis es sommes etes ont ai "
           "as avons avez pas ne plus tres si comme aussi tout tous toute toutes leurs "
           "the a an of to in and or but is are was were be been it its this that these those for on "
           "with as at by from".split())


def _mattr(toks, w=50):
    if len(toks) < w:
        return np.nan
    return float(np.mean([len(set(toks[i:i + w])) / w for i in range(len(toks) - w + 1)]))


def _mtld(toks, thr=0.72):
    if len(toks) < 50:
        return np.nan
    def _pass(seq):
        factors, types, n = 0.0, set(), 0
        ttr = 1.0
        for t in seq:
            n += 1
            types.add(t)
            ttr = len(types) / n
            if ttr <= thr:
                factors += 1
                types, n = set(), 0
        if n > 0:
            factors += (1 - ttr) / (1 - thr)
        return len(seq) / factors if factors > 0 else float(len(seq))
    return (_mtld_val := (_pass(toks) + _pass(toks[::-1])) / 2)


def _metrics(text):
    toks = _WORD.findall(text.lower()) if text else []
    content = [t for t in toks if t not in STOP]
    # proper-noun-ish exclusion on the original-case tokens
    raw = _WORD.findall(text) if text else []
    lower_noncap = [t.lower() for t in raw if not t[:1].isupper()]
    return {"mtld": _mtld(toks), "cwmattr": _mattr(content), "nocapmattr": _mattr(lower_noncap)}


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
    for rg in range(n_rg):
        part = os.path.join(PARTS, f"part_{rg:03d}.parquet")
        if os.path.exists(part):
            continue
        for attempt in range(8):
            try:
                rows = []
                for r in pf.read_row_group(rg, columns=COLS).to_pylist():
                    if r["choice"] not in ("a_better", "b_better"):
                        continue
                    md = r["metadata"] or {}
                    if "fr" not in (md.get("languages") or []):
                        continue
                    pa = _conversation_prefix(r["full_conversation_a"], r["turn"])
                    pb = _conversation_prefix(r["full_conversation_b"], r["turn"])
                    if pa is None or pb is None:
                        continue
                    ta, tb = _assistant_text(pa), _assistant_text(pb)
                    if not ta or not tb:
                        continue
                    rec = {"conversation_pair_id": r["comparison_id"], "vote_turn": int(r["turn"])}
                    for side, txt in (("a", ta), ("b", tb)):
                        for k, v in _metrics(txt).items():
                            rec[f"{k}_{side}"] = v
                    rows.append(rec)
                pd.DataFrame(rows).to_parquet(part, index=False, compression="zstd")
                print(f"  rg {rg+1}/{n_rg} kept={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  rg {rg+1} attempt {attempt+1} failed: {type(e).__name__}; retry in {wait}s", flush=True)
                time.sleep(wait)
                pf = _open()
        else:
            raise RuntimeError(f"row group {rg} failed")

    parts = [pd.read_parquet(os.path.join(PARTS, f)) for f in sorted(os.listdir(PARTS)) if f.endswith(".parquet")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = (df.sort_values(["conversation_pair_id", "vote_turn"])
          .drop_duplicates(subset="conversation_pair_id", keep="last")
          .drop(columns="vote_turn").reset_index(drop=True))
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nWrote {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
