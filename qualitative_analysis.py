#!/usr/bin/env python3
"""
Qualitative analysis of winner-flipping conversations.

Identifies battles where standard vs style-controlled BT models disagree
on the predicted winner, then analyzes whether the style-controlled prediction
better aligns with content quality.

Outputs: qualitative_results.json
"""

import json, re, sys, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict, Counter

np.random.seed(42)

# ============================================================
# 1. LOAD PRE-COMPUTED DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading pre-computed data")
print("=" * 60)

with open('clean_analysis_results.json') as f:
    results = json.load(f)

battles = pd.read_parquet('battles_bt_styled.parquet')
print(f"  Battles: {len(battles):,}")

# Extract standard and controlled ratings
std_ratings = {m: info['rating'] for m, info in results['rankings']['standard'].items()}
ctrl_ratings = {m: info['rating'] for m, info in results['rankings']['controlled'].items()}
std_ranks = {m: info['rank'] for m, info in results['rankings']['standard'].items()}
ctrl_ranks = {m: info['rank'] for m, info in results['rankings']['controlled'].items()}

# Style coefficients (from logistic regression)
style_coefs = {}
for feat in ['headers', 'lists', 'bold', 'code_blocks', 'emoji']:
    style_coefs[feat] = results['style_coefficient_cis'][feat]['point']
print(f"  Style coefficients: {style_coefs}")

# ============================================================
# 2. IDENTIFY WINNER-FLIPPING BATTLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Identifying winner-flipping battles")
print("=" * 60)

# For each non-tie battle, determine:
# - Standard BT predicted winner (model with higher standard rating)
# - Controlled BT predicted winner (model with higher controlled rating)
non_tie = battles[battles['winner'] != 'tie'].copy()
print(f"  Non-tie battles: {len(non_tie):,}")

# Add ratings
non_tie['std_rating_a'] = non_tie['model_a_name'].map(std_ratings)
non_tie['std_rating_b'] = non_tie['model_b_name'].map(std_ratings)
non_tie['ctrl_rating_a'] = non_tie['model_a_name'].map(ctrl_ratings)
non_tie['ctrl_rating_b'] = non_tie['model_b_name'].map(ctrl_ratings)

# Drop battles where we don't have ratings for both models
non_tie = non_tie.dropna(subset=['std_rating_a', 'std_rating_b', 'ctrl_rating_a', 'ctrl_rating_b'])
print(f"  With ratings for both models: {len(non_tie):,}")

# Standard predicted winner
non_tie['std_pred'] = np.where(
    non_tie['std_rating_a'] >= non_tie['std_rating_b'], 'model_a', 'model_b'
)

# Controlled predicted winner
non_tie['ctrl_pred'] = np.where(
    non_tie['ctrl_rating_a'] >= non_tie['ctrl_rating_b'], 'model_a', 'model_b'
)

# Winner-flipping = standard and controlled disagree
non_tie['is_flip'] = non_tie['std_pred'] != non_tie['ctrl_pred']
n_flips = non_tie['is_flip'].sum()
print(f"  Winner-flipping battles: {n_flips:,} ({100*n_flips/len(non_tie):.1f}%)")

flips = non_tie[non_tie['is_flip']].copy()

# Compute style advantage for each flip
# (how much more formatting the vote winner had vs the loser)
for feat in ['headers', 'lists', 'bold', 'code_blocks', 'emoji']:
    # Diff = winner's feature - loser's feature
    flips[f'{feat}_winner'] = np.where(
        flips['winner'] == 'model_a',
        flips[f'{feat}_a'], flips[f'{feat}_b']
    )
    flips[f'{feat}_loser'] = np.where(
        flips['winner'] == 'model_a',
        flips[f'{feat}_b'], flips[f'{feat}_a']
    )
    flips[f'{feat}_diff'] = flips[f'{feat}_winner'] - flips[f'{feat}_loser']

# Total "style boost" to the vote winner (using logistic regression coefficients)
flips['style_boost'] = sum(
    style_coefs[f] * flips[f'{f}_diff'] for f in ['headers', 'lists', 'bold']
)

# Add rank change info
flips['rank_change_a'] = flips['model_a_name'].map(
    lambda m: ctrl_ranks.get(m, 0) - std_ranks.get(m, 0)
)
flips['rank_change_b'] = flips['model_b_name'].map(
    lambda m: ctrl_ranks.get(m, 0) - std_ranks.get(m, 0)
)

# Which model in the flip is the "big mover"?
flips['max_abs_rank_change'] = np.maximum(
    flips['rank_change_a'].abs(), flips['rank_change_b'].abs()
)

print(f"\n  Style boost to vote winner in flips:")
print(f"    Mean: {flips['style_boost'].mean():.3f}")
print(f"    Median: {flips['style_boost'].median():.3f}")
print(f"    Positive (winner had more formatting): {(flips['style_boost'] > 0).sum():,} "
      f"({100*(flips['style_boost'] > 0).mean():.1f}%)")

# ============================================================
# 3. ANALYZE FLIP PATTERNS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Analyzing flip patterns")
print("=" * 60)

# Which models appear most in flips?
flip_models = Counter()
for _, row in flips.iterrows():
    flip_models[row['model_a_name']] += 1
    flip_models[row['model_b_name']] += 1

print("\n  Top 15 models in winner-flipping battles:")
for model, count in flip_models.most_common(15):
    std_r = std_ranks.get(model, '?')
    ctrl_r = ctrl_ranks.get(model, '?')
    change = ctrl_r - std_r if isinstance(std_r, int) and isinstance(ctrl_r, int) else '?'
    print(f"    {model}: {count} flips (rank {std_r}->{ctrl_r}, change={change:+d})")

# Does the vote align with standard or controlled prediction?
flips['vote_matches_std'] = flips['winner'] == flips['std_pred']
flips['vote_matches_ctrl'] = flips['winner'] == flips['ctrl_pred']

print(f"\n  In flipped battles, vote agrees with:")
print(f"    Standard BT: {flips['vote_matches_std'].sum():,} ({100*flips['vote_matches_std'].mean():.1f}%)")
print(f"    Controlled BT: {flips['vote_matches_ctrl'].sum():,} ({100*flips['vote_matches_ctrl'].mean():.1f}%)")

# ============================================================
# 4. LOAD CONVERSATION TEXT + QUALITY ATTRIBUTES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Loading conversation text for sampled flips")
print("=" * 60)

# Sample strategy: focus on vote-sourced flips (have quality attributes)
vote_flips = flips[flips['source'] == 'vote'].copy()
print(f"  Vote-sourced flips: {len(vote_flips):,}")

# Stratified sample:
# - 20 from top rank-changers (|rank_change| >= 8)
# - 15 from moderate changers (4 <= |rank_change| < 8)
# - 15 from small changers (|rank_change| < 4)
big_movers = vote_flips[vote_flips['max_abs_rank_change'] >= 8]
mid_movers = vote_flips[(vote_flips['max_abs_rank_change'] >= 4) & (vote_flips['max_abs_rank_change'] < 8)]
small_movers = vote_flips[vote_flips['max_abs_rank_change'] < 4]

n_big = min(20, len(big_movers))
n_mid = min(15, len(mid_movers))
n_small = min(15, len(small_movers))

sample = pd.concat([
    big_movers.sample(n=n_big, random_state=42) if n_big > 0 else pd.DataFrame(),
    mid_movers.sample(n=n_mid, random_state=42) if n_mid > 0 else pd.DataFrame(),
    small_movers.sample(n=n_small, random_state=42) if n_small > 0 else pd.DataFrame(),
])
print(f"  Sampled: {len(sample)} ({n_big} big + {n_mid} mid + {n_small} small movers)")

# Get conversation_pair_ids to look up
sample_cpids = set(sample['conversation_pair_id'])

# Load votes with conversation text
print("  Loading votes with conversation text...")
pf = pq.ParquetFile('comparia_votes.parquet')
votes_full = pf.read_row_group(0).to_pandas()
print(f"  Loaded {len(votes_full):,} votes")

# Filter to our sample
votes_sample = votes_full[votes_full['conversation_pair_id'].isin(sample_cpids)].copy()
print(f"  Matched {len(votes_sample)} conversations")

# ============================================================
# 5. EXTRACT AND ANALYZE CONVERSATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Extracting and analyzing conversations")
print("=" * 60)

def extract_response_text(conv, role='assistant'):
    """Extract concatenated response text from a conversation.
    Falls back to reasoning/reasoning_content if content is empty (reasoning-only models)."""
    if conv is None:
        return ''
    texts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get('role') == role:
            content = msg.get('content', '') or ''
            if content.strip():
                texts.append(content)
            else:
                # Fallback: use reasoning content if visible content is missing
                reasoning = msg.get('reasoning', '') or msg.get('reasoning_content', '') or ''
                if reasoning.strip():
                    texts.append(f'[reasoning-only, no visible content saved: {len(reasoning)} chars]')
    return '\n'.join(texts)

def strip_think_tags(text):
    """Remove <think>...</think> blocks from text"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def truncate(text, max_chars=1500):
    """Truncate text for display"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f'\n[... truncated, {len(text)} chars total]'

# Quality attribute columns
quality_cols = [
    'conv_useful_a', 'conv_useful_b',
    'conv_creative_a', 'conv_creative_b',
    'conv_clear_formatting_a', 'conv_clear_formatting_b',
    'conv_incorrect_a', 'conv_incorrect_b',
    'conv_superficial_a', 'conv_superficial_b',
    'conv_instructions_not_followed_a', 'conv_instructions_not_followed_b',
    'conv_complete_a', 'conv_complete_b',
]

cases = []
for _, battle_row in sample.iterrows():
    cpid = battle_row['conversation_pair_id']
    vote_row = votes_sample[votes_sample['conversation_pair_id'] == cpid]
    if len(vote_row) == 0:
        continue
    vote_row = vote_row.iloc[0]

    # Extract text
    text_a = strip_think_tags(extract_response_text(vote_row['conversation_a']))
    text_b = strip_think_tags(extract_response_text(vote_row['conversation_b']))
    prompt = vote_row.get('opening_msg', '') or ''

    # Winner info
    actual_winner = battle_row['winner']  # 'model_a' or 'model_b'
    std_pred = battle_row['std_pred']
    ctrl_pred = battle_row['ctrl_pred']

    winner_model = battle_row['model_a_name'] if actual_winner == 'model_a' else battle_row['model_b_name']
    loser_model = battle_row['model_b_name'] if actual_winner == 'model_a' else battle_row['model_a_name']

    # Style advantage of vote winner
    style_boost = battle_row['style_boost']

    # Quality attributes
    quality = {}
    for col in quality_cols:
        val = vote_row.get(col, None)
        if val is not None and not pd.isna(val):
            quality[col] = bool(val)

    # Determine which side the ctrl_pred favors
    ctrl_winner_model = battle_row['model_a_name'] if ctrl_pred == 'model_a' else battle_row['model_b_name']

    # Comments
    comments_a = vote_row.get('conv_comments_a', '') or ''
    comments_b = vote_row.get('conv_comments_b', '') or ''

    case = {
        'conversation_pair_id': cpid,
        'prompt': prompt[:500],
        'model_a': battle_row['model_a_name'],
        'model_b': battle_row['model_b_name'],
        'vote_winner': actual_winner,
        'vote_winner_model': winner_model,
        'std_pred': std_pred,
        'ctrl_pred': ctrl_pred,
        'ctrl_winner_model': ctrl_winner_model,
        'style_boost_to_vote_winner': round(style_boost, 3),
        'formatting': {
            'headers': {'winner': int(battle_row['headers_winner']), 'loser': int(battle_row['headers_loser'])},
            'lists': {'winner': int(battle_row['lists_winner']), 'loser': int(battle_row['lists_loser'])},
            'bold': {'winner': int(battle_row['bold_winner']), 'loser': int(battle_row['bold_loser'])},
        },
        'text_a_length': len(text_a),
        'text_b_length': len(text_b),
        'text_a_excerpt': truncate(text_a),
        'text_b_excerpt': truncate(text_b),
        'quality_attributes': quality,
        'comments_a': comments_a[:300] if comments_a else '',
        'comments_b': comments_b[:300] if comments_b else '',
        'rank_change_a': int(battle_row['rank_change_a']),
        'rank_change_b': int(battle_row['rank_change_b']),
    }
    cases.append(case)

print(f"  Extracted {len(cases)} cases")

# ============================================================
# 6. QUALITY-ATTRIBUTE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Quality-attribute analysis of flips")
print("=" * 60)

# For ALL vote-sourced flips (not just sample), check quality attributes
# Merge quality attributes
vote_flip_cpids = set(vote_flips['conversation_pair_id'])
votes_flips_full = votes_full[votes_full['conversation_pair_id'].isin(vote_flip_cpids)].copy()
print(f"  Vote flips with quality data: {len(votes_flips_full):,}")

# Merge with battle data to know who won
flip_merged = vote_flips.merge(
    votes_flips_full[['conversation_pair_id'] + quality_cols],
    on='conversation_pair_id', how='inner'
)
print(f"  Merged records: {len(flip_merged):,}")

# For each flip, determine: does the ctrl-predicted winner have better quality attributes?
# Map quality attributes to winner/loser sides
positive_attrs = ['conv_useful', 'conv_creative', 'conv_complete']
negative_attrs = ['conv_incorrect', 'conv_superficial', 'conv_instructions_not_followed']

# Compare quality: vote winner vs ctrl-predicted winner
# In a flip, vote winner != ctrl-predicted winner
# So ctrl-predicted winner == the vote LOSER
quality_analysis = {
    'n_flips_with_quality': 0,
    'ctrl_winner_better_positive': 0,
    'ctrl_winner_better_negative': 0,
    'vote_winner_better_positive': 0,
    'vote_winner_better_negative': 0,
    'equal_positive': 0,
    'equal_negative': 0,
    'attribute_counts': defaultdict(lambda: {'vote_winner': 0, 'ctrl_winner': 0, 'both': 0, 'neither': 0}),
}

for _, row in flip_merged.iterrows():
    vote_winner_side = row['winner']  # 'model_a' or 'model_b'
    ctrl_winner_side = row['ctrl_pred']  # opposite of vote_winner in flips

    has_any_quality = False
    pos_score_vote_winner = 0
    pos_score_ctrl_winner = 0
    neg_score_vote_winner = 0
    neg_score_ctrl_winner = 0

    vw = 'a' if vote_winner_side == 'model_a' else 'b'
    cw = 'a' if ctrl_winner_side == 'model_a' else 'b'

    for attr in positive_attrs:
        col_vw = f'{attr}_{vw}'
        col_cw = f'{attr}_{cw}'
        val_vw = row.get(col_vw, None)
        val_cw = row.get(col_cw, None)

        if pd.notna(val_vw) or pd.notna(val_cw):
            has_any_quality = True
            vw_val = bool(val_vw) if pd.notna(val_vw) else False
            cw_val = bool(val_cw) if pd.notna(val_cw) else False

            if vw_val and cw_val:
                quality_analysis['attribute_counts'][attr]['both'] += 1
            elif vw_val:
                quality_analysis['attribute_counts'][attr]['vote_winner'] += 1
            elif cw_val:
                quality_analysis['attribute_counts'][attr]['ctrl_winner'] += 1
            else:
                quality_analysis['attribute_counts'][attr]['neither'] += 1

            pos_score_vote_winner += int(vw_val)
            pos_score_ctrl_winner += int(cw_val)

    for attr in negative_attrs:
        col_vw = f'{attr}_{vw}'
        col_cw = f'{attr}_{cw}'
        val_vw = row.get(col_vw, None)
        val_cw = row.get(col_cw, None)

        if pd.notna(val_vw) or pd.notna(val_cw):
            has_any_quality = True
            vw_val = bool(val_vw) if pd.notna(val_vw) else False
            cw_val = bool(val_cw) if pd.notna(val_cw) else False

            if vw_val and cw_val:
                quality_analysis['attribute_counts'][attr]['both'] += 1
            elif vw_val:
                quality_analysis['attribute_counts'][attr]['vote_winner'] += 1
            elif cw_val:
                quality_analysis['attribute_counts'][attr]['ctrl_winner'] += 1
            else:
                quality_analysis['attribute_counts'][attr]['neither'] += 1

            neg_score_vote_winner += int(vw_val)
            neg_score_ctrl_winner += int(cw_val)

    if has_any_quality:
        quality_analysis['n_flips_with_quality'] += 1
        if pos_score_ctrl_winner > pos_score_vote_winner:
            quality_analysis['ctrl_winner_better_positive'] += 1
        elif pos_score_vote_winner > pos_score_ctrl_winner:
            quality_analysis['vote_winner_better_positive'] += 1
        else:
            quality_analysis['equal_positive'] += 1

        if neg_score_ctrl_winner < neg_score_vote_winner:
            quality_analysis['ctrl_winner_better_negative'] += 1
        elif neg_score_vote_winner < neg_score_ctrl_winner:
            quality_analysis['vote_winner_better_negative'] += 1
        else:
            quality_analysis['equal_negative'] += 1

# Convert defaultdict
quality_analysis['attribute_counts'] = dict(quality_analysis['attribute_counts'])

print(f"\n  Flips with quality attributes: {quality_analysis['n_flips_with_quality']}")
print(f"  Positive attributes (useful, creative, complete):")
print(f"    Ctrl winner rated better: {quality_analysis['ctrl_winner_better_positive']} "
      f"({100*quality_analysis['ctrl_winner_better_positive']/max(1,quality_analysis['n_flips_with_quality']):.1f}%)")
print(f"    Vote winner rated better: {quality_analysis['vote_winner_better_positive']} "
      f"({100*quality_analysis['vote_winner_better_positive']/max(1,quality_analysis['n_flips_with_quality']):.1f}%)")
print(f"    Equal: {quality_analysis['equal_positive']}")
print(f"  Negative attributes (incorrect, superficial, instructions_not_followed):")
print(f"    Ctrl winner rated better (fewer negatives): {quality_analysis['ctrl_winner_better_negative']} "
      f"({100*quality_analysis['ctrl_winner_better_negative']/max(1,quality_analysis['n_flips_with_quality']):.1f}%)")
print(f"    Vote winner rated better: {quality_analysis['vote_winner_better_negative']} "
      f"({100*quality_analysis['vote_winner_better_negative']/max(1,quality_analysis['n_flips_with_quality']):.1f}%)")

# ============================================================
# 7. FORMATTING ANALYSIS OF ALL FLIPS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Formatting analysis of all flips")
print("=" * 60)

# In flips, does the vote winner have more formatting?
for feat in ['headers', 'lists', 'bold']:
    winner_more = (flips[f'{feat}_diff'] > 0).sum()
    loser_more = (flips[f'{feat}_diff'] < 0).sum()
    equal = (flips[f'{feat}_diff'] == 0).sum()
    print(f"  {feat}: vote winner has more in {winner_more} ({100*winner_more/len(flips):.1f}%), "
          f"loser has more in {loser_more} ({100*loser_more/len(flips):.1f}%), "
          f"equal in {equal} ({100*equal/len(flips):.1f}%)")

# Overall: in how many flips does vote winner have more total formatting?
flips['total_fmt_winner'] = flips['headers_winner'] + flips['lists_winner'] + flips['bold_winner']
flips['total_fmt_loser'] = flips['headers_loser'] + flips['lists_loser'] + flips['bold_loser']
flips['winner_formats_more'] = flips['total_fmt_winner'] > flips['total_fmt_loser']
flips['loser_formats_more'] = flips['total_fmt_loser'] > flips['total_fmt_winner']

print(f"\n  Overall formatting:")
print(f"    Vote winner formats more: {flips['winner_formats_more'].sum()} "
      f"({100*flips['winner_formats_more'].mean():.1f}%)")
print(f"    Vote loser formats more: {flips['loser_formats_more'].sum()} "
      f"({100*flips['loser_formats_more'].mean():.1f}%)")

# ============================================================
# 8. CLEAR FORMATTING ATTRIBUTE
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Clear formatting attribute in flips")
print("=" * 60)

# conv_clear_formatting_a/b tells us if the user LIKED the formatting
# In flips, how often does the MORE formatted response get clear_formatting=True?

fmt_analysis = {'total': 0, 'formatted_praised': 0, 'formatted_not_praised': 0,
                'less_formatted_praised': 0, 'both_praised': 0, 'neither_praised': 0}

for _, row in flip_merged.iterrows():
    vote_winner_side = 'a' if row['winner'] == 'model_a' else 'b'
    vote_loser_side = 'b' if vote_winner_side == 'a' else 'a'

    fmt_w = row.get(f'conv_clear_formatting_{vote_winner_side}', None)
    fmt_l = row.get(f'conv_clear_formatting_{vote_loser_side}', None)

    if pd.notna(fmt_w) or pd.notna(fmt_l):
        fmt_analysis['total'] += 1
        w_val = bool(fmt_w) if pd.notna(fmt_w) else False
        l_val = bool(fmt_l) if pd.notna(fmt_l) else False
        if w_val and l_val:
            fmt_analysis['both_praised'] += 1
        elif w_val:
            fmt_analysis['formatted_praised'] += 1
        elif l_val:
            fmt_analysis['less_formatted_praised'] += 1
        else:
            fmt_analysis['neither_praised'] += 1

print(f"  Flips with clear_formatting data: {fmt_analysis['total']}")
if fmt_analysis['total'] > 0:
    print(f"  Vote winner praised for formatting: {fmt_analysis['formatted_praised']} "
          f"({100*fmt_analysis['formatted_praised']/fmt_analysis['total']:.1f}%)")
    print(f"  Vote loser praised for formatting: {fmt_analysis['less_formatted_praised']} "
          f"({100*fmt_analysis['less_formatted_praised']/fmt_analysis['total']:.1f}%)")
    print(f"  Both praised: {fmt_analysis['both_praised']} "
          f"({100*fmt_analysis['both_praised']/fmt_analysis['total']:.1f}%)")
    print(f"  Neither praised: {fmt_analysis['neither_praised']} "
          f"({100*fmt_analysis['neither_praised']/fmt_analysis['total']:.1f}%)")

# ============================================================
# 9. NON-FLIP BASELINE COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Non-flip baseline comparison")
print("=" * 60)

# Compare formatting patterns in non-flipping battles vs flipping battles
non_flip = non_tie[~non_tie['is_flip']].copy()

# Compute style boost for non-flips too
for feat in ['headers', 'lists', 'bold']:
    non_flip[f'{feat}_winner'] = np.where(
        non_flip['winner'] == 'model_a',
        non_flip[f'{feat}_a'], non_flip[f'{feat}_b']
    )
    non_flip[f'{feat}_loser'] = np.where(
        non_flip['winner'] == 'model_a',
        non_flip[f'{feat}_b'], non_flip[f'{feat}_a']
    )
    non_flip[f'{feat}_diff'] = non_flip[f'{feat}_winner'] - non_flip[f'{feat}_loser']

non_flip['style_boost'] = sum(
    style_coefs[f] * non_flip[f'{f}_diff'] for f in ['headers', 'lists', 'bold']
)

print(f"  Non-flip battles: {len(non_flip):,}")
print(f"  Style boost (vote winner):")
print(f"    Flips:     mean={flips['style_boost'].mean():.3f}, median={flips['style_boost'].median():.3f}")
print(f"    Non-flips: mean={non_flip['style_boost'].mean():.3f}, median={non_flip['style_boost'].median():.3f}")

from scipy import stats as sp_stats
t_stat, p_val = sp_stats.mannwhitneyu(
    flips['style_boost'].values, non_flip['style_boost'].values, alternative='two-sided'
)
print(f"    Mann-Whitney U test: p={p_val:.6f}")

# ============================================================
# 10. QUALITATIVE CODING OF SAMPLE CASES
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Coding sample cases")
print("=" * 60)

# Automated coding based on available signals
coded_cases = []
for case in cases:
    qa = case['quality_attributes']

    # Determine vote winner and ctrl winner sides
    vw_side = 'a' if case['vote_winner'] == 'model_a' else 'b'
    cw_side = 'a' if case['ctrl_pred'] == 'model_a' else 'b'

    # Compute quality scores
    pos_attrs_vw = sum(1 for attr in ['conv_useful', 'conv_creative', 'conv_complete']
                       if qa.get(f'{attr}_{vw_side}', False))
    pos_attrs_cw = sum(1 for attr in ['conv_useful', 'conv_creative', 'conv_complete']
                       if qa.get(f'{attr}_{cw_side}', False))
    neg_attrs_vw = sum(1 for attr in ['conv_incorrect', 'conv_superficial', 'conv_instructions_not_followed']
                       if qa.get(f'{attr}_{vw_side}', False))
    neg_attrs_cw = sum(1 for attr in ['conv_incorrect', 'conv_superficial', 'conv_instructions_not_followed']
                       if qa.get(f'{attr}_{cw_side}', False))

    # Coding
    fmt = case['formatting']
    total_fmt_winner = fmt['headers']['winner'] + fmt['lists']['winner'] + fmt['bold']['winner']
    total_fmt_loser = fmt['headers']['loser'] + fmt['lists']['loser'] + fmt['bold']['loser']

    code = {
        'conversation_pair_id': case['conversation_pair_id'],
        'vote_winner_model': case['vote_winner_model'],
        'ctrl_winner_model': case['ctrl_winner_model'],
        'style_boost': case['style_boost_to_vote_winner'],
        'vote_winner_formats_more': total_fmt_winner > total_fmt_loser,
        'formatting_ratio': f"{total_fmt_winner}:{total_fmt_loser}",
        'text_length_ratio': f"{case['text_a_length']}:{case['text_b_length']}",
        'pos_quality_vote_winner': pos_attrs_vw,
        'pos_quality_ctrl_winner': pos_attrs_cw,
        'neg_quality_vote_winner': neg_attrs_vw,
        'neg_quality_ctrl_winner': neg_attrs_cw,
        'quality_supports_ctrl': (pos_attrs_cw > pos_attrs_vw) or (neg_attrs_cw < neg_attrs_vw and neg_attrs_vw > 0),
        'quality_supports_vote': (pos_attrs_vw > pos_attrs_cw) or (neg_attrs_vw < neg_attrs_cw and neg_attrs_cw > 0),
        'has_comments': bool(case['comments_a'] or case['comments_b']),
    }
    coded_cases.append(code)

# Summary
n_quality_supports_ctrl = sum(1 for c in coded_cases if c['quality_supports_ctrl'])
n_quality_supports_vote = sum(1 for c in coded_cases if c['quality_supports_vote'])
n_ambiguous = sum(1 for c in coded_cases if not c['quality_supports_ctrl'] and not c['quality_supports_vote'])
n_winner_formats_more = sum(1 for c in coded_cases if c['vote_winner_formats_more'])

print(f"  Cases coded: {len(coded_cases)}")
print(f"  Vote winner formats more: {n_winner_formats_more}/{len(coded_cases)} ({100*n_winner_formats_more/max(1,len(coded_cases)):.0f}%)")
print(f"  Quality supports ctrl prediction: {n_quality_supports_ctrl}/{len(coded_cases)}")
print(f"  Quality supports vote: {n_quality_supports_vote}/{len(coded_cases)}")
print(f"  Ambiguous (no quality data or equal): {n_ambiguous}/{len(coded_cases)}")

# ============================================================
# 11. SELECT ILLUSTRATIVE EXAMPLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Selecting illustrative examples")
print("=" * 60)

# Pick 5 illustrative cases:
# 1. High-profile model (mistral-large) losing after style control
# 2. Reasoning model gaining after style control
# 3. Case where quality attributes support ctrl prediction
# 4. Case where vote winner had dramatically more formatting
# 5. Case with user comments

illustrative = []

# Sort by style_boost descending (vote winner had most formatting advantage)
cases_sorted = sorted(cases, key=lambda c: c['style_boost_to_vote_winner'], reverse=True)

# Highest style boost cases
for case in cases_sorted[:3]:
    case['selection_reason'] = 'highest_style_boost'
    illustrative.append(case)

# Cases involving big rank-changers
big_models = {'mistral-large-2512', 'o3-mini', 'kimi-k2-thinking', 'deepseek-r1-0528'}
for case in cases:
    if case['model_a'] in big_models or case['model_b'] in big_models:
        if case not in illustrative:
            case['selection_reason'] = 'involves_big_rank_changer'
            illustrative.append(case)
    if len(illustrative) >= 6:
        break

# Cases with comments
for case in cases:
    if case['comments_a'] or case['comments_b']:
        if case not in illustrative:
            case['selection_reason'] = 'has_user_comments'
            illustrative.append(case)
    if len(illustrative) >= 8:
        break

print(f"  Selected {len(illustrative)} illustrative examples")
for ex in illustrative:
    print(f"    - {ex['model_a']} vs {ex['model_b']}: "
          f"vote={ex['vote_winner_model']}, ctrl={ex['ctrl_winner_model']}, "
          f"boost={ex['style_boost_to_vote_winner']:.2f}, reason={ex.get('selection_reason','')}")

# ============================================================
# 12. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Saving results")
print("=" * 60)

output = {
    'summary': {
        'total_non_tie_battles': int(len(non_tie)),
        'winner_flipping_battles': int(n_flips),
        'flip_rate': round(100 * n_flips / len(non_tie), 2),
        'vote_sourced_flips': int(len(vote_flips)),
        'sample_size': len(cases),
    },
    'flip_patterns': {
        'vote_agrees_with_std': int(flips['vote_matches_std'].sum()),
        'vote_agrees_with_ctrl': int(flips['vote_matches_ctrl'].sum()),
        'pct_vote_agrees_std': round(100 * flips['vote_matches_std'].mean(), 1),
        'pct_vote_agrees_ctrl': round(100 * flips['vote_matches_ctrl'].mean(), 1),
        'style_boost_mean': round(flips['style_boost'].mean(), 3),
        'style_boost_median': round(flips['style_boost'].median(), 3),
        'nonflip_style_boost_mean': round(non_flip['style_boost'].mean(), 3),
        'mann_whitney_p': round(p_val, 6),
    },
    'formatting_in_flips': {
        'vote_winner_formats_more_pct': round(100 * flips['winner_formats_more'].mean(), 1),
        'vote_loser_formats_more_pct': round(100 * flips['loser_formats_more'].mean(), 1),
        'per_feature': {},
    },
    'quality_analysis': {
        k: v for k, v in quality_analysis.items()
        if k != 'attribute_counts'
    },
    'quality_per_attribute': {
        k: dict(v) for k, v in quality_analysis['attribute_counts'].items()
    },
    'clear_formatting_in_flips': fmt_analysis,
    'top_models_in_flips': [
        {'model': m, 'count': c, 'std_rank': std_ranks.get(m), 'ctrl_rank': ctrl_ranks.get(m)}
        for m, c in flip_models.most_common(15)
    ],
    'coded_cases_summary': {
        'n_cases': len(coded_cases),
        'vote_winner_formats_more': n_winner_formats_more,
        'quality_supports_ctrl': n_quality_supports_ctrl,
        'quality_supports_vote': n_quality_supports_vote,
        'ambiguous': n_ambiguous,
    },
    'coded_cases': coded_cases,
    'illustrative_examples': [
        {
            'conversation_pair_id': ex['conversation_pair_id'],
            'prompt': ex['prompt'],
            'model_a': ex['model_a'],
            'model_b': ex['model_b'],
            'vote_winner_model': ex['vote_winner_model'],
            'ctrl_winner_model': ex['ctrl_winner_model'],
            'style_boost': ex['style_boost_to_vote_winner'],
            'formatting': ex['formatting'],
            'text_a_length': ex['text_a_length'],
            'text_b_length': ex['text_b_length'],
            'text_a_excerpt': ex['text_a_excerpt'],
            'text_b_excerpt': ex['text_b_excerpt'],
            'quality_attributes': ex['quality_attributes'],
            'comments_a': ex['comments_a'],
            'comments_b': ex['comments_b'],
            'selection_reason': ex.get('selection_reason', ''),
        }
        for ex in illustrative
    ],
}

with open('qualitative_results.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to qualitative_results.json")
print("\nDone!")
