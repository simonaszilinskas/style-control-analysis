#!/usr/bin/env python3
"""
Rule-based task-type classifier for opening prompts (§4.8).

Task type (write code / translate / summarize / ...) is distinct from subject
matter (§4.7) and is a more direct driver of presentation: a list request
produces lists, a coding request produces code blocks. We label each battle's
opening prompt with a coarse task taxonomy using ordered keyword rules over the
(mostly French) prompt text. First matching rule wins. The classifier is an
exploratory proxy and is not treated as a validated task taxonomy.

    python src/task_classify.py    # data/prompts.parquet -> data/battle_tasks.parquet
"""

import re
import pandas as pd

from paths import BATTLES, DATA

# Ordered (label, regex). First match wins; order handles overlap
# (e.g. "traduis ce code" -> translation before code; "ecris une fonction" -> code before writing).
RULES = [
    ("translation", r"\btradui|\btranslat|\btraduction|en anglais|en fran[cç]ais|en espagnol|in english|traduis"),
    # bare "code" is excluded when it means a French legal/admin code (code de la route,
    # du travail, civil, penal, postal, ...), which is not programming.
    ("code", r"\bpython\b|\bjavascript\b|\bsql\b|\bhtml\b|\bcss\b|\bjava\b|\bc\+\+|\bcoder\b|fonction|script|programm|d[ée]bug|compil|algorithm|regex|```|\bcode\b(?!\s+(de la route|du travail|de commerce|civil|p[ée]nal|postal|mon[ée]taire|des|de la|de proc[ée]dure))"),
    ("summarization", r"\br[ée]sum|synth[eè]se|summar|\btl;?dr\b|en quelques (mots|phrases|lignes)|points cl[ée]s"),
    ("math", r"\bcalcul|\br[ée]sou|[ée]quation|combien font|\bd[ée]riv[ée]e|int[ée]grale|pourcentage de|math[ée]matique|\bprobabilit"),
    ("list_table", r"\bliste\b|\blister\b|\b[ée]num[eè]r|\btableau\b|\btable\b|dresse une liste|fais une liste|list of|donne-moi une liste"),
    ("writing", r"\br[ée]dig|\b[ée]cris\b|\b[ée]crire\b|\bwrite\b|\bdraft\b|po[eè]me|\bhistoire\b|\bstory\b|\be-?mail\b|\blettre\b|\bdiscours\b|dissertation|\bessai\b|paragraphe|un texte|un article|un message"),
    ("ideas", r"\bid[ée]es?\b|propose[- ]?(moi|nous)?\b|brainstorm|sugg[eè]re|donne[- ]?moi des|trouve des|des exemples de"),
    ("advice", r"\bconseil|recommand|que dois-je|que faire|aide-moi [aà]|comment (faire|puis-je)|devrais-je|should i|how (do|can) i|meilleur[e]? (fa[cç]on|mani[eè]re)"),
    ("explanation", r"\bexpliqu|qu['e ]est[- ]ce|\bpourquoi\b|\bcomment\b|c['e]est quoi|\bwhat is\b|\bexplain\b|d[ée]finis|d[ée]finition|diff[ée]rence entre|\?"),
]
COMPILED = [(lab, re.compile(rx, re.IGNORECASE)) for lab, rx in RULES]


def classify(prompt):
    if not prompt:
        return "other"
    p = prompt[:2000]  # first part carries the intent
    for lab, rx in COMPILED:
        if rx.search(p):
            return lab
    return "other"


def main():
    df = pd.read_parquet(DATA / "prompts.parquet")
    df["task"] = df["prompt"].map(classify)
    battle_ids = pd.read_parquet(
        BATTLES, columns=["conversation_pair_id"]
    ).drop_duplicates()
    out = battle_ids.merge(
        df[["conversation_pair_id", "task"]],
        on="conversation_pair_id",
        how="left",
        validate="one_to_one",
    ).dropna(subset=["task"])
    out.to_parquet(DATA / "battle_tasks.parquet", index=False)
    print(f"{len(out):,} prompts classified")
    print(df["task"].value_counts())
    print(f"\n'other' rate: {(df['task']=='other').mean()*100:.1f}%")
    print("Saved data/battle_tasks.parquet")


if __name__ == "__main__":
    main()
