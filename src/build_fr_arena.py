#!/usr/bin/env python3
"""
Build a vote-truncated compar:IA battle table from comparia-fr-arena.

comparia-fr-arena is turn-level: one row per turn, and `choice` (a_better /
b_better) is populated on the single turn where the user voted. We keep those
decisive French votes, retain the last decisive reaction per comparison, and
truncate the release's completed `full_conversation_*` fields at that reaction,
and compute features only on text visible when the vote was cast.  The release's
`tokens_*` fields are per-turn counts, so cumulative prefix length is reconstructed
by summing them through the retained vote.  Do not use `total_tokens_*`: those
describe the completed conversation and can include later turns.

One row per comparison (battle):
  conversation_pair_id, model_a_name, model_b_name, winner (model_a/model_b),
  source='vote', mode, vote_turn, conv_turns (visible user-message count),
  final_turn, post_vote_turns, primary_topic,
  {formatting}_a/_b, {linguistic}_a/_b, length_a/_b (cumulative output tokens).

Needs a HF token (CLI login or HF_TOKEN) for the gated dataset.
    python build_fr_arena.py            # -> fr_battles.parquet
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import textstat
from huggingface_hub import HfFileSystem, get_token

from checkpoints import (
    prepare_checkpoint_dir,
    processor_sha256,
    verify_local_sha256,
)
from paths import BATTLES, DATA

HF_REVISION = "8cd6488c5d0c3b8dfcb9339d11ae9624c84359be"
HF_PATH = ("datasets/ministere-culture/comparia-fr-arena@"
           f"{HF_REVISION}/comparia-fr-arena.parquet")
# A local file cannot expose its Hugging Face revision reliably.  Use one only
# when the caller explicitly confirms it is the pinned revision above.
LOCAL_ENV = "COMPARIA_FR_ARENA_PARQUET"
LOCAL_SHA_ENV = "COMPARIA_FR_ARENA_SHA256"
LOCAL_PATHS = [os.environ[LOCAL_ENV]] if os.environ.get(LOCAL_ENV) else []
OUT = BATTLES
COLS = ["comparison_id", "choice", "turn", "model_a", "model_b",
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
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

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
    """Concatenate visible final-answer text from assistant messages.

    Some provider payloads embed hidden reasoning in paired ``<think>`` blocks
    before the final answer.  The voter-facing answer is the text outside those
    blocks, so remove only complete blocks rather than discarding the entire
    message.  A dangling opening or closing tag is ambiguous; retain only the
    unambiguous text before it and never include a possible reasoning span.
    ``reasoning_content`` remains deliberately excluded from this final-answer
    analysis.
    """
    if msgs is None:
        return ""
    parts = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            visible = []
            position = 0
            while True:
                opening = _THINK_OPEN_RE.search(c, position)
                closing = _THINK_CLOSE_RE.search(c, position)
                if opening is None and closing is None:
                    visible.append(c[position:])
                    break
                if closing is not None and (opening is None or closing.start() < opening.start()):
                    visible.append(c[position:closing.start()])
                    break
                visible.append(c[position:opening.start()])
                closing = _THINK_CLOSE_RE.search(c, opening.end())
                if closing is None:
                    break
                position = closing.end()
            text = "".join(visible).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _conversation_prefix(msgs, vote_turn):
    """Return messages visible at a zero-based vote turn.

    Each arena turn ends with one assistant response.  The completed conversation
    is repeated on every source row, so a vote on turn ``t`` must stop immediately
    after assistant response ``t + 1``.  Returning ``None`` instead of a partial
    prefix makes malformed rows fail closed.
    """
    if msgs is None or vote_turn is None or int(vote_turn) < 0:
        return None
    target = int(vote_turn) + 1
    prefix = []
    assistants = 0
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        prefix.append(msg)
        if msg.get("role") == "assistant":
            assistants += 1
            if assistants == target:
                return prefix
    return None


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
        out["long_sent_ratio"] = float(
            np.mean([1.0 if sentence_length > 25 else 0.0 for sentence_length in lens])
        )
    return out


PARTS = DATA / f"fr_parts_vote_truncated_{HF_REVISION[:8]}"
TOKEN_PARTS = DATA / f"fr_token_parts_vote_truncated_{HF_REVISION[:8]}"

BATTLE_COLUMNS = [
    "conversation_pair_id", "vote_turn", "model_a_name", "model_b_name",
    "winner", "source", "mode", "conv_turns", "primary_topic",
] + [f"{k}_{side}" for k in FEATS for side in ("a", "b")]
TOKEN_COLUMNS = ["conversation_pair_id", "turn", "tokens_a", "tokens_b"]


def _process_rows(pyrows):
    battles, tokens = [], []
    for r in pyrows:
        md = r["metadata"] or {}
        if "fr" not in (md.get("languages") or []):
            continue
        tokens.append({
            "conversation_pair_id": r["comparison_id"],
            "turn": r["turn"],
            "tokens_a": md.get("tokens_a"),
            "tokens_b": md.get("tokens_b"),
        })
        if r["choice"] not in ("a_better", "b_better"):
            continue
        pa = _conversation_prefix(r["full_conversation_a"], r["turn"])
        pb = _conversation_prefix(r["full_conversation_b"], r["turn"])
        if pa is None or pb is None:
            continue
        ta, tb = _assistant_text(pa), _assistant_text(pb)
        if not ta or not tb:
            continue
        visible_turns_a, visible_turns_b = _user_turns(pa), _user_turns(pb)
        expected_turns = int(r["turn"]) + 1
        if visible_turns_a != expected_turns or visible_turns_b != expected_turns:
            continue
        cats = md.get("categories")
        rec = {
            "conversation_pair_id": r["comparison_id"],
            "vote_turn": int(r["turn"]),
            "model_a_name": r["model_a"], "model_b_name": r["model_b"],
            "winner": "model_a" if r["choice"] == "a_better" else "model_b",
            "source": "vote", "mode": md.get("mode"),
            "conv_turns": visible_turns_a,
            "primary_topic": cats[0] if cats is not None and len(cats) > 0 else None,
        }
        fa, fb = features(ta), features(tb)
        for k in FEATS:
            rec[f"{k}_a"], rec[f"{k}_b"] = fa[k], fb[k]
        battles.append(rec)
    return battles, tokens


def _attach_prefix_lengths(battles, tokens):
    """Attach exact cumulative output-token totals through each retained vote."""
    keys = ["conversation_pair_id", "turn"]
    if tokens.duplicated(keys).any():
        examples = tokens.loc[tokens.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise ValueError(f"duplicate source turns found: {examples}")

    tokens = tokens.sort_values(keys).copy()
    group = tokens["conversation_pair_id"]
    for side in ("a", "b"):
        current = pd.to_numeric(tokens[f"tokens_{side}"], errors="coerce")
        missing_so_far = current.isna().groupby(group).cumsum()
        cumulative = current.fillna(0).groupby(group).cumsum().astype(float)
        tokens[f"length_{side}"] = cumulative.mask(missing_so_far > 0)

    final_turn = (tokens.groupby("conversation_pair_id", as_index=False)["turn"]
                  .max().rename(columns={"turn": "final_turn"}))
    lengths = tokens[keys + ["length_a", "length_b"]]
    out = battles.merge(
        lengths,
        left_on=["conversation_pair_id", "vote_turn"],
        right_on=keys,
        how="left",
        validate="one_to_one",
    ).drop(columns="turn")
    out = out.merge(final_turn, on="conversation_pair_id", how="left", validate="one_to_one")
    out["post_vote_turns"] = out["final_turn"] - out["vote_turn"]
    return out


def _open():
    for p in LOCAL_PATHS:
        if os.path.exists(p):
            verify_local_sha256(p, os.environ.get(LOCAL_SHA_ENV))
            return pq.ParquetFile(p)
        raise FileNotFoundError(f"{LOCAL_ENV} does not exist: {p}")
    return pq.ParquetFile(HfFileSystem(token=get_token()).open(HF_PATH, "rb"))


def _checkpoint_manifest(kind):
    return {
        "format_version": 1,
        "kind": kind,
        "source": {
            "dataset": "ministere-culture/comparia-fr-arena",
            "revision": HF_REVISION,
            "path": HF_PATH,
        },
        "columns": COLS,
        "processor_sha256": processor_sha256(
            (_assistant_text, _conversation_prefix, _user_turns, features, _process_rows)
        ),
    }


def main(reset_checkpoints=False):
    prepare_checkpoint_dir(
        PARTS,
        _checkpoint_manifest("battle_features"),
        reset=reset_checkpoints,
    )
    prepare_checkpoint_dir(
        TOKEN_PARTS,
        _checkpoint_manifest("vote_time_tokens"),
        reset=reset_checkpoints,
    )
    pf = _open()
    n_rg = pf.num_row_groups
    t0 = time.time()
    # Stream row group by row group, checkpoint each to its own part, resume on
    # restart, and retry a row group on a dropped connection (the 8.9 GB stream
    # is fragile and disk is too small to download the whole file).
    for rg in range(n_rg):
        part = os.path.join(PARTS, f"part_{rg:03d}.parquet")
        token_part = os.path.join(TOKEN_PARTS, f"part_{rg:03d}.parquet")
        if os.path.exists(part) and os.path.exists(token_part):
            continue
        for attempt in range(8):
            try:
                pyrows = pf.read_row_group(rg, columns=COLS).to_pylist()
                rows, token_rows = _process_rows(pyrows)
                pd.DataFrame(rows, columns=BATTLE_COLUMNS).to_parquet(
                    part, index=False, compression="zstd")
                pd.DataFrame(token_rows, columns=TOKEN_COLUMNS).to_parquet(
                    token_part, index=False, compression="zstd")
                print(f"  rg {rg+1}/{n_rg} kept={len(rows)} votes, "
                      f"{len(token_rows)} turns ({time.time()-t0:.0f}s)", flush=True)
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
    token_parts = [pd.read_parquet(os.path.join(TOKEN_PARTS, f))
                   for f in sorted(os.listdir(TOKEN_PARTS)) if f.endswith(".parquet")]
    token_df = pd.concat([p for p in token_parts if len(p)], ignore_index=True)
    # A few comparisons carry more than one decisive vote (voted at different
    # turns); keep one battle per comparison (the last, i.e. most recent vote).
    n0 = len(df)
    df = (df.sort_values(["conversation_pair_id", "vote_turn"])
          .drop_duplicates(subset="conversation_pair_id", keep="last")
          .reset_index(drop=True))
    print(f"  deduplicated {n0-len(df)} repeated comparison ids")
    df = _attach_prefix_lengths(df, token_df)
    if not (df["conv_turns"] == df["vote_turn"] + 1).all():
        raise AssertionError("visible turn count does not match retained vote turn")
    if (df["post_vote_turns"] < 0).any():
        raise AssertionError("retained vote occurs after the source's final turn")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nWrote {len(df):,} decisive French battles -> {OUT}")
    print(f"  models: {pd.concat([df.model_a_name, df.model_b_name]).nunique()}")
    print(f"  with topic: {df.primary_topic.notna().mean()*100:.1f}%")
    print(f"  multi-turn: {(df.conv_turns>=2).mean()*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="discard generated row-group checkpoints before rebuilding",
    )
    args = parser.parse_args()
    main(reset_checkpoints=args.reset_checkpoints)
