# Derived data

The files in this directory are analysis-ready derivatives of the gated
`ministere-culture/comparia-fr-arena` release at revision
`8cd6488c5d0c3b8dfcb9339d11ae9624c84359be`. They contain no prompt or response
text. Raw prompts, raw conversations, and streaming checkpoints are excluded
from Git.

Cryptographic hashes and byte sizes for distributed Parquet files are recorded
in `results/artifact_manifest.json`.

## `fr_battles.parquet`

One row per retained comparison: the last decisive French vote in each
conversation, reconstructed at the vote turn.

| Columns | Meaning |
|---|---|
| `conversation_pair_id` | Opaque comparison identifier used for joins |
| `vote_turn`, `final_turn`, `post_vote_turns` | Retained vote index, source final-turn index, and their difference |
| `model_a_name`, `model_b_name`, `winner` | Compared public model identifiers and decisive outcome |
| `source`, `mode`, `primary_topic` | Vote source, arena pairing mode, and source-provided topic |
| `conv_turns` | User turns visible at the retained vote |
| `{headers,lists,bold,code_blocks,emoji}_{a,b}` | Vote-time markdown/emoji counts |
| `{rel,cli,fkg,ttr,mattr,asl,long_sent_ratio}_{a,b}` | Vote-time language features |
| `length_{a,b}` | Cumulative source-provided output-token totals through the vote |

Language features are missing when visible text does not satisfy their
measurement requirements. MATTR requires at least 50 word tokens.

## Auxiliary tables

- `timestamps.parquet`: comparison ID and source timestamp for weekly block
  resampling.
- `battle_tasks.parquet`: comparison ID and exploratory rule-based task proxy.
- `mattr_alt.parquet`: comparison ID plus MTLD, function-word-removed MATTR,
  and capitalized-token-excluded MATTR checks.

Auxiliary analyses use metric-specific available-case support and report their
sample sizes. They should never be assumed to have one row for every battle.

## Privacy boundary

The source release may contain user-generated text. The public derivatives here
retain only aggregate measurements and opaque IDs needed to reproduce the
statistical analysis. Do not add `prompts.parquet`, raw conversation exports, or
checkpoint parts to version control.
