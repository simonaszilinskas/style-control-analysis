#!/usr/bin/env python3
"""
Extract the opening user prompt for each battle, for task-type classification (§4.8).

Streams comparia-fr-arena (local copy if present, else HF), keeps decisive French
votes (same filter as build_fr_arena.py), and saves the first user message of each
battle keyed by conversation_pair_id. Checkpoints per row group and resumes.

    python src/extract_prompts.py    # -> data/prompts.parquet
"""

import os
import time

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, get_token

from paths import DATA
from build_fr_arena import LOCAL_PATHS, HF_PATH

OUT = DATA / "prompts.parquet"
PARTS = DATA / "prompt_parts"
COLS = ["comparison_id", "choice", "full_conversation_a", "metadata"]


def _first_user(msgs):
    if msgs is None:
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip() or None
    return None


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
                    p = _first_user(r["full_conversation_a"])
                    if p:
                        rows.append({"conversation_pair_id": r["comparison_id"], "prompt": p})
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

    parts = [pd.read_parquet(os.path.join(PARTS, f)) for f in sorted(os.listdir(PARTS))
             if f.endswith(".parquet")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.drop_duplicates(subset="conversation_pair_id", keep="last").reset_index(drop=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nWrote {len(df):,} prompts -> {OUT}")


if __name__ == "__main__":
    main()
