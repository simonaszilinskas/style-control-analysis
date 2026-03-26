#!/usr/bin/env python3
"""
Compar:IA Style Control Analysis — Clean Pipeline
Applies all data quality filters identified in the audit, then runs the analysis.
Outputs: clean_analysis_results.json with all key metrics and tables.
"""

import json, math, re, sys, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

pd.options.display.float_format = '{:.4f}'.format
np.random.seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading datasets")
print("=" * 60)

# Votes
print("\nLoading votes...")
pf = pq.ParquetFile('comparia_votes.parquet')
votes = pf.read_row_group(0).to_pandas()
print(f"  Raw votes: {len(votes):,}")

# Reactions
print("Loading reactions...")
pf = pq.ParquetFile('comparia_reactions.parquet')
dfs = []
for i in range(pf.metadata.num_row_groups):
    dfs.append(pf.read_row_group(i).to_pandas())
reactions = pd.concat(dfs, ignore_index=True)
print(f"  Raw reactions: {len(reactions):,}")

# Conversations (flat columns only for metadata)
print("Loading conversations metadata...")
pf = pq.ParquetFile('comparia_conversations.parquet')
flat_cols = ['conversation_pair_id', 'mode', 'total_conv_a_output_tokens',
             'total_conv_b_output_tokens', 'model_a_total_params', 'model_b_total_params',
             'model_a_active_params', 'model_b_active_params']
conv_meta = pf.read(columns=flat_cols).to_pandas()
print(f"  Conversations metadata: {len(conv_meta):,}")

results = {}

# ============================================================
# 1b. MERGE MODEL ALIASES
# ============================================================
MODEL_ALIASES = {
    "mistral-medium-3.1": "mistral-medium-2508",
}
print("\nApplying model name aliases...")
for old_name, new_name in MODEL_ALIASES.items():
    for col in ['model_a_name', 'model_b_name', 'chosen_model_name']:
        if col in votes.columns:
            n = (votes[col] == old_name).sum()
            votes[col] = votes[col].replace(old_name, new_name)
            if n: print(f"  votes.{col}: renamed {n} '{old_name}' -> '{new_name}'")
        if col in reactions.columns:
            n = (reactions[col] == old_name).sum()
            reactions[col] = reactions[col].replace(old_name, new_name)
            if n: print(f"  reactions.{col}: renamed {n} '{old_name}' -> '{new_name}'")

# ============================================================
# 2. APPLY DATA QUALITY FILTERS TO VOTES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Filtering votes")
print("=" * 60)

filter_log_votes = {}

# 2a. Remove same-model pairs
same_model = votes['model_a_name'] == votes['model_b_name']
filter_log_votes['same_model_pairs'] = int(same_model.sum())
votes = votes[~same_model].copy()
print(f"  Removed {filter_log_votes['same_model_pairs']} same-model pairs -> {len(votes):,}")

# 2b. Remove no-choice votes (chosen_model_name is null AND both_equal is not True)
no_choice = votes['chosen_model_name'].isna() & (votes['both_equal'] != True)
filter_log_votes['no_choice'] = int(no_choice.sum())
votes = votes[~no_choice].copy()
print(f"  Removed {filter_log_votes['no_choice']} no-choice votes -> {len(votes):,}")

# 2c. Deduplicate conversation_pair_ids (keep most recent)
dup_mask = votes.duplicated(subset='conversation_pair_id', keep='last')
filter_log_votes['duplicate_conv_pair_ids'] = int(dup_mask.sum())
votes = votes[~dup_mask].copy()
print(f"  Removed {filter_log_votes['duplicate_conv_pair_ids']} duplicate conversation_pair_ids -> {len(votes):,}")

# 2d. Derive winner column
def derive_winner(row):
    if row['both_equal'] == True:
        return 'tie'
    elif pd.notna(row['chosen_model_name']):
        if row['chosen_model_name'] == row['model_a_name']:
            return 'model_a'
        elif row['chosen_model_name'] == row['model_b_name']:
            return 'model_b'
    return 'tie'

votes['winner'] = votes.apply(derive_winner, axis=1)

winner_dist = votes['winner'].value_counts()
print(f"\n  Vote winner distribution:")
for w, c in winner_dist.items():
    print(f"    {w}: {c:,} ({c/len(votes)*100:.1f}%)")

filter_log_votes['final_count'] = len(votes)
results['vote_filters'] = filter_log_votes

# ============================================================
# 3. APPLY DATA QUALITY FILTERS TO REACTIONS -> CONVERT TO VOTES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Filtering reactions and converting to pairwise preferences")
print("=" * 60)

filter_log_reactions = {}

# 3a. Remove same-model pairs
same_model_r = reactions['model_a_name'] == reactions['model_b_name']
filter_log_reactions['same_model_pairs'] = int(same_model_r.sum())
reactions = reactions[~same_model_r].copy()
print(f"  Removed {filter_log_reactions['same_model_pairs']} same-model pairs -> {len(reactions):,}")

# 3b. Remove even msg_index (user messages)
even_idx = reactions['msg_index'] % 2 == 0
filter_log_reactions['even_msg_index'] = int(even_idx.sum())
reactions = reactions[~even_idx].copy()
print(f"  Removed {filter_log_reactions['even_msg_index']} even msg_index -> {len(reactions):,}")

# 3c. Remove empty/very short responses
short_response = reactions['response_content'].str.len() < 10
filter_log_reactions['short_responses'] = int(short_response.sum())
reactions = reactions[~short_response].copy()
print(f"  Removed {filter_log_reactions['short_responses']} short responses -> {len(reactions):,}")

# 3d. Convert reactions to pairwise preferences
# Group by conversation, model side, and sum likes/dislikes
reaction_agg = reactions.groupby(
    ['conversation_pair_id', 'model_a_name', 'model_b_name', 'model_pos']
).agg(
    liked_sum=('liked', 'sum'),
    disliked_sum=('disliked', 'sum'),
    n_reactions=('liked', 'count')
).reset_index()

# Pivot: get scores for each side
side_a = reaction_agg[reaction_agg['model_pos'] == 'a'].copy()
side_b = reaction_agg[reaction_agg['model_pos'] == 'b'].copy()

side_a = side_a.rename(columns={'liked_sum': 'liked_a', 'disliked_sum': 'disliked_a', 'n_reactions': 'n_reactions_a'})
side_b = side_b.rename(columns={'liked_sum': 'liked_b', 'disliked_sum': 'disliked_b', 'n_reactions': 'n_reactions_b'})

# Inner merge: require both sides to have reactions
reaction_pairs = side_a[['conversation_pair_id', 'model_a_name', 'model_b_name', 'liked_a', 'disliked_a', 'n_reactions_a']].merge(
    side_b[['conversation_pair_id', 'liked_b', 'disliked_b', 'n_reactions_b']],
    on='conversation_pair_id',
    how='inner'
)

filter_log_reactions['one_sided_reactions_dropped'] = int(
    reaction_agg['conversation_pair_id'].nunique() - reaction_pairs['conversation_pair_id'].nunique()
)

# Compute net scores
reaction_pairs['score_a'] = reaction_pairs['liked_a'] - reaction_pairs['disliked_a']
reaction_pairs['score_b'] = reaction_pairs['liked_b'] - reaction_pairs['disliked_b']

# Determine winner
def reaction_winner(row):
    if row['score_a'] > row['score_b']:
        return 'model_a'
    elif row['score_b'] > row['score_a']:
        return 'model_b'
    return 'tie'

reaction_pairs['winner'] = reaction_pairs.apply(reaction_winner, axis=1)

r_winner_dist = reaction_pairs['winner'].value_counts()
print(f"\n  Reaction-derived pairs: {len(reaction_pairs):,}")
print(f"  One-sided reactions dropped: {filter_log_reactions['one_sided_reactions_dropped']:,}")
print(f"  Reaction-derived winner distribution:")
for w, c in r_winner_dist.items():
    print(f"    {w}: {c:,} ({c/len(reaction_pairs)*100:.1f}%)")

filter_log_reactions['final_pairs'] = len(reaction_pairs)
results['reaction_filters'] = filter_log_reactions

# ============================================================
# 4. COMBINE VOTES AND REACTION-DERIVED PREFERENCES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Combining votes and reaction-derived preferences")
print("=" * 60)

# Tag source
votes_combined = votes[['conversation_pair_id', 'model_a_name', 'model_b_name', 'winner']].copy()
votes_combined['source'] = 'vote'

reactions_combined = reaction_pairs[['conversation_pair_id', 'model_a_name', 'model_b_name', 'winner']].copy()
reactions_combined['source'] = 'reaction'

# Check overlap
overlap = set(votes_combined['conversation_pair_id']) & set(reactions_combined['conversation_pair_id'])
print(f"  Overlap between votes and reaction pairs: {len(overlap)}")

# Remove overlapping from reactions (prefer explicit votes)
reactions_combined = reactions_combined[~reactions_combined['conversation_pair_id'].isin(overlap)]

battles = pd.concat([votes_combined, reactions_combined], ignore_index=True)
print(f"  Combined battles: {len(battles):,} ({len(votes_combined):,} votes + {len(reactions_combined):,} reaction-derived)")

results['combined'] = {
    'total_battles': len(battles),
    'from_votes': len(votes_combined),
    'from_reactions': len(reactions_combined),
    'overlap_removed': len(overlap),
}

# ============================================================
# 5. JOIN METADATA (mode, token counts, model params)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Joining conversation metadata")
print("=" * 60)

# Clean mode column
mode_map = conv_meta[['conversation_pair_id', 'mode']].copy()
valid_modes = {'random', 'custom', 'big-vs-small', 'reasoning', 'small-models'}
mode_map['mode_clean'] = mode_map['mode'].where(mode_map['mode'].isin(valid_modes), 'unknown')
mode_map.loc[mode_map['mode'].isna(), 'mode_clean'] = 'unknown'

battles = battles.merge(
    conv_meta[['conversation_pair_id', 'total_conv_a_output_tokens', 'total_conv_b_output_tokens',
               'model_a_total_params', 'model_b_total_params']],
    on='conversation_pair_id', how='left'
)
battles = battles.merge(mode_map[['conversation_pair_id', 'mode_clean']], on='conversation_pair_id', how='left')

match_rate = battles['mode_clean'].notna().mean()
print(f"  Metadata match rate: {match_rate*100:.1f}%")

# Mode distribution
mode_dist = battles['mode_clean'].value_counts(dropna=False)
print(f"\n  Mode distribution in battles:")
for m, c in mode_dist.items():
    print(f"    {m}: {c:,} ({c/len(battles)*100:.1f}%)")

results['mode_distribution'] = {str(k): int(v) for k, v in mode_dist.items()}

# ============================================================
# 6. EXTRACT STYLE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Extracting style features from conversation text")
print("=" * 60)

def count_markdown_headers(text):
    """Count markdown headers (# through ######)"""
    if not text:
        return 0
    return len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))

def count_markdown_lists(text):
    """Count markdown list items (ordered and unordered)"""
    if not text:
        return 0
    unordered = len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE))
    ordered = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
    return unordered + ordered

def count_markdown_bold(text):
    """Count markdown bold markers"""
    if not text:
        return 0
    return len(re.findall(r'\*\*[^*]+\*\*', text))

def count_code_blocks(text):
    """Count fenced code blocks (``` or ~~~)"""
    if not text:
        return 0
    return len(re.findall(r'```|~~~', text)) // 2

def count_emojis(text):
    """Count emoji characters"""
    if not text:
        return 0
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        "\U00002700-\U000027BF]+", flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))

def has_think_tags(text):
    """Check for <think> tag contamination"""
    if not text:
        return False
    return '<think>' in text or '</think>' in text

def has_reasoning_only(conv, role='assistant'):
    """Check if conversation has reasoning content but no visible content.
    This affects reasoning models where content=None but reasoning/reasoning_content is populated."""
    if conv is None:
        return False
    for msg in conv:
        if isinstance(msg, dict) and msg.get('role') == role:
            content = msg.get('content', '') or ''
            reasoning = msg.get('reasoning', '') or ''
            reasoning_content = msg.get('reasoning_content', '') or ''
            if not content.strip() and (reasoning.strip() or reasoning_content.strip()):
                return True
    return False

def extract_response_text(conv, role='assistant'):
    """Extract concatenated response text from a conversation"""
    if conv is None:
        return ''
    texts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get('role') == role:
            content = msg.get('content', '') or ''
            # Skip if contaminated with <think> tags
            if not has_think_tags(content):
                texts.append(content)
    return '\n'.join(texts)

# Extract style features from vote conversations
print("  Extracting style features from votes...")
n_processed = 0
n_contaminated = 0
style_data = []
reasoning_only_cpids = set()  # Track battles with missing content (reasoning-only)
reasoning_only_models = defaultdict(int)  # Track which models are affected

# We need to read votes with conversation text
pf = pq.ParquetFile('comparia_votes.parquet')
votes_full = pf.read_row_group(0).to_pandas()

# Build a lookup for which conversation_pair_ids are in our battles
battle_cpids = set(battles[battles['source'] == 'vote']['conversation_pair_id'])

for idx in range(len(votes_full)):
    if idx % 20000 == 0:
        print(f"    Processing vote {idx:,}/{len(votes_full):,}...")

    cpid = votes_full['conversation_pair_id'].iloc[idx]
    if cpid not in battle_cpids:
        continue

    conv_a = votes_full['conversation_a'].iloc[idx]
    conv_b = votes_full['conversation_b'].iloc[idx]

    # Check for reasoning-only content (missing user-visible text)
    ro_a = has_reasoning_only(conv_a)
    ro_b = has_reasoning_only(conv_b)
    if ro_a or ro_b:
        reasoning_only_cpids.add(cpid)
        if ro_a:
            reasoning_only_models[votes_full['model_a_name'].iloc[idx]] += 1
        if ro_b:
            reasoning_only_models[votes_full['model_b_name'].iloc[idx]] += 1

    text_a = extract_response_text(conv_a)
    text_b = extract_response_text(conv_b)

    style_data.append({
        'conversation_pair_id': cpid,
        'headers_a': count_markdown_headers(text_a),
        'headers_b': count_markdown_headers(text_b),
        'lists_a': count_markdown_lists(text_a),
        'lists_b': count_markdown_lists(text_b),
        'bold_a': count_markdown_bold(text_a),
        'bold_b': count_markdown_bold(text_b),
        'code_blocks_a': count_code_blocks(text_a),
        'code_blocks_b': count_code_blocks(text_b),
        'emoji_a': count_emojis(text_a),
        'emoji_b': count_emojis(text_b),
        'len_a': len(text_a),
        'len_b': len(text_b),
    })
    n_processed += 1

print(f"  Processed {n_processed:,} vote conversations")
print(f"  Detected {len(reasoning_only_cpids):,} vote battles with reasoning-only content (missing visible text)")

# Also extract from reaction response_content for reaction-derived battles
print("  Extracting style features from reaction-derived battles...")
reaction_cpids = set(battles[battles['source'] == 'reaction']['conversation_pair_id'])

# Group reactions by conversation to get both sides
reaction_texts = {}
for _, row in reactions.iterrows():
    cpid = row['conversation_pair_id']
    if cpid not in reaction_cpids:
        continue
    side = row['model_pos']  # 'a' or 'b'
    text = row['response_content'] or ''

    if cpid not in reaction_texts:
        reaction_texts[cpid] = {'a': '', 'b': ''}
    # Concatenate if multiple messages
    if reaction_texts[cpid][side]:
        reaction_texts[cpid][side] += '\n' + text
    else:
        reaction_texts[cpid][side] = text

for cpid, texts in reaction_texts.items():
    text_a = texts.get('a', '')
    text_b = texts.get('b', '')

    style_data.append({
        'conversation_pair_id': cpid,
        'headers_a': count_markdown_headers(text_a),
        'headers_b': count_markdown_headers(text_b),
        'lists_a': count_markdown_lists(text_a),
        'lists_b': count_markdown_lists(text_b),
        'bold_a': count_markdown_bold(text_a),
        'bold_b': count_markdown_bold(text_b),
        'code_blocks_a': count_code_blocks(text_a),
        'code_blocks_b': count_code_blocks(text_b),
        'emoji_a': count_emojis(text_a),
        'emoji_b': count_emojis(text_b),
        'len_a': len(text_a),
        'len_b': len(text_b),
    })

style_df = pd.DataFrame(style_data)
print(f"  Total style features extracted: {len(style_df):,}")

# Merge style features into battles
battles = battles.merge(style_df, on='conversation_pair_id', how='left')
has_style = battles['headers_a'].notna()
print(f"  Battles with style features: {has_style.sum():,} ({has_style.mean()*100:.1f}%)")

# Filter out reasoning-only battles (missing visible content)
print(f"\n  --- Reasoning-only content filter ---")
n_before_reasoning_filter = len(battles)
reasoning_mask = battles['conversation_pair_id'].isin(reasoning_only_cpids)
n_reasoning_only = reasoning_mask.sum()
print(f"  Battles with reasoning-only content (no visible text): {n_reasoning_only:,} ({n_reasoning_only/n_before_reasoning_filter*100:.1f}%)")
battles = battles[~reasoning_mask].copy()
print(f"  Battles after reasoning-only filter: {len(battles):,}")

# Report affected models
if reasoning_only_models:
    print(f"  Top affected models (by side count):")
    for model, count in sorted(reasoning_only_models.items(), key=lambda x: -x[1])[:15]:
        print(f"    {model}: {count:,}")

has_style = battles['headers_a'].notna()
print(f"  Battles with style features after filter: {has_style.sum():,} ({has_style.mean()*100:.1f}%)")

filter_log_votes['reasoning_only_battles'] = int(n_reasoning_only)
results['reasoning_filter'] = {
    'battles_removed': int(n_reasoning_only),
    'pct_removed': round(n_reasoning_only / n_before_reasoning_filter * 100, 2),
    'affected_models': {m: int(c) for m, c in sorted(reasoning_only_models.items(), key=lambda x: -x[1])[:20]},
    'battles_after_filter': len(battles),
}

results['style_extraction'] = {
    'total_with_features': int(has_style.sum()),
    'missing_features': int((~has_style).sum()),
}

# ============================================================
# 7. STYLE FEATURE STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Style feature statistics")
print("=" * 60)

battles_styled = battles[has_style].copy()

for feat in ['headers', 'lists', 'bold', 'code_blocks', 'emoji', 'len']:
    col_a = f'{feat}_a'
    col_b = f'{feat}_b'
    all_vals = pd.concat([battles_styled[col_a], battles_styled[col_b]])
    print(f"\n  {feat}:")
    print(f"    Mean: {all_vals.mean():.2f}, Median: {all_vals.median():.1f}")
    print(f"    P25: {all_vals.quantile(0.25):.1f}, P75: {all_vals.quantile(0.75):.1f}")
    print(f"    Min: {all_vals.min():.0f}, Max: {all_vals.max():.0f}")
    print(f"    Zero: {(all_vals == 0).sum():,} ({(all_vals == 0).mean()*100:.1f}%)")

# ============================================================
# 8. BRADLEY-TERRY MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Bradley-Terry model (standard and style-controlled)")
print("=" * 60)

SCALE = 400
BASE = 10
INIT_RATING = 1000

def compute_bt_ratings(battles_df, models):
    """Standard BT model via logistic regression"""
    model_to_idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    # Filter to decisive battles
    decisive = battles_df[battles_df['winner'].isin(['model_a', 'model_b'])].copy()

    X = np.zeros((len(decisive), n))
    y = np.zeros(len(decisive))

    for i, (_, row) in enumerate(decisive.iterrows()):
        a_idx = model_to_idx.get(row['model_a_name'])
        b_idx = model_to_idx.get(row['model_b_name'])
        if a_idx is None or b_idx is None:
            continue
        X[i, a_idx] = 1
        X[i, b_idx] = -1
        y[i] = 1 if row['winner'] == 'model_a' else 0

    # Remove all-zero rows (models not in our index)
    valid = np.any(X != 0, axis=1)
    X = X[valid]
    y = y[valid]

    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)

    ratings = INIT_RATING + SCALE * lr.coef_[0] / np.log(BASE)
    return dict(zip(models, ratings)), lr

def compute_bt_style_controlled(battles_df, models, style_features):
    """BT model with style control covariates"""
    model_to_idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    n_style = len(style_features)

    decisive = battles_df[battles_df['winner'].isin(['model_a', 'model_b'])].copy()
    decisive = decisive.dropna(subset=[f'{sf}_a' for sf in style_features])

    X_model = np.zeros((len(decisive), n))
    X_style = np.zeros((len(decisive), n_style))
    y = np.zeros(len(decisive))

    for i, (_, row) in enumerate(decisive.iterrows()):
        a_idx = model_to_idx.get(row['model_a_name'])
        b_idx = model_to_idx.get(row['model_b_name'])
        if a_idx is None or b_idx is None:
            continue
        X_model[i, a_idx] = 1
        X_model[i, b_idx] = -1
        y[i] = 1 if row['winner'] == 'model_a' else 0

        for j, sf in enumerate(style_features):
            X_style[i, j] = row[f'{sf}_a'] - row[f'{sf}_b']

    valid = np.any(X_model != 0, axis=1)
    X_model = X_model[valid]
    X_style = X_style[valid]
    y = y[valid]

    # Standardize style features
    scaler = StandardScaler()
    X_style_scaled = scaler.fit_transform(X_style)

    X = np.hstack([X_model, X_style_scaled])

    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)

    ratings = INIT_RATING + SCALE * lr.coef_[0][:n] / np.log(BASE)
    style_coefs = lr.coef_[0][n:]

    return dict(zip(models, ratings)), dict(zip(style_features, style_coefs)), scaler, lr

# Get models with enough battles
model_counts = pd.concat([battles_styled['model_a_name'], battles_styled['model_b_name']]).value_counts()
MIN_BATTLES = 100
models = sorted(model_counts[model_counts >= MIN_BATTLES].index.tolist())
print(f"  Models with >= {MIN_BATTLES} battles: {len(models)}")

# Filter battles to only include these models
battles_bt = battles_styled[
    battles_styled['model_a_name'].isin(models) &
    battles_styled['model_b_name'].isin(models)
].copy()
print(f"  Battles for BT: {len(battles_bt):,}")

# Standard BT
print("\n  Computing standard BT ratings...")
bt_standard, lr_standard = compute_bt_ratings(battles_bt, models)

# Style-controlled BT
style_features = ['headers', 'lists', 'bold', 'code_blocks', 'emoji']
print("  Computing style-controlled BT ratings...")
bt_controlled, style_coefs, scaler, lr_controlled = compute_bt_style_controlled(battles_bt, models, style_features)

# Print rankings comparison
print(f"\n  === Style Coefficients (effect on log-odds of winning) ===")
for feat, coef in sorted(style_coefs.items(), key=lambda x: -abs(x[1])):
    odds_pct = (np.exp(coef) - 1) * 100
    print(f"    {feat:15s}: {coef:+.4f} ({odds_pct:+.1f}% odds per SD)")

results['style_coefficients'] = {
    feat: {
        'coefficient': float(coef),
        'odds_change_pct': float((np.exp(coef) - 1) * 100)
    }
    for feat, coef in style_coefs.items()
}

# Rankings
standard_ranked = sorted(bt_standard.items(), key=lambda x: -x[1])
controlled_ranked = sorted(bt_controlled.items(), key=lambda x: -x[1])

standard_rank = {m: i+1 for i, (m, _) in enumerate(standard_ranked)}
controlled_rank = {m: i+1 for i, (m, _) in enumerate(controlled_ranked)}

print(f"\n  === Top 20 Models (Standard BT) ===")
print(f"  {'Rank':>4s}  {'Model':40s}  {'Rating':>8s}  {'Ctrl Rating':>11s}  {'Rank Δ':>6s}")
for i, (model, rating) in enumerate(standard_ranked[:20]):
    ctrl_rating = bt_controlled.get(model, 0)
    rank_change = standard_rank[model] - controlled_rank[model]
    print(f"  {i+1:4d}  {model:40s}  {rating:8.1f}  {ctrl_rating:11.1f}  {rank_change:+6d}")

# Correlation between standard and controlled
models_both = [m for m in models if m in bt_standard and m in bt_controlled]
r_vals = [bt_standard[m] for m in models_both]
c_vals = [bt_controlled[m] for m in models_both]
correlation = np.corrcoef(r_vals, c_vals)[0, 1]
print(f"\n  Correlation (standard vs controlled): {correlation:.4f}")

results['bt_correlation'] = float(correlation)
results['n_models'] = len(models)
results['n_battles_bt'] = len(battles_bt)

# Rank changes
rank_changes = []
for m in models:
    rank_changes.append({
        'model': m,
        'standard_rank': standard_rank[m],
        'controlled_rank': controlled_rank[m],
        'rank_change': standard_rank[m] - controlled_rank[m],
        'standard_rating': bt_standard[m],
        'controlled_rating': bt_controlled[m],
        'rating_change': bt_controlled[m] - bt_standard[m],
    })
rank_changes_df = pd.DataFrame(rank_changes)
rank_changes_df = rank_changes_df.sort_values('rank_change', key=abs, ascending=False)

print(f"\n  === Top 10 Rank Changes ===")
print(f"  {'Model':40s}  {'Std→Ctrl':>10s}  {'ΔRank':>6s}  {'ΔRating':>8s}")
for _, row in rank_changes_df.head(10).iterrows():
    print(f"  {row['model']:40s}  {row['standard_rank']:3.0f}→{row['controlled_rank']:3.0f}  {row['rank_change']:+6.0f}  {row['rating_change']:+8.1f}")

results['top_rank_changes'] = rank_changes_df.head(20).to_dict('records')

# ============================================================
# 9. POSITION BIAS CHECK
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Position bias analysis")
print("=" * 60)

decisive = battles_bt[battles_bt['winner'].isin(['model_a', 'model_b'])]
a_wins = (decisive['winner'] == 'model_a').sum()
b_wins = (decisive['winner'] == 'model_b').sum()
total_decisive = a_wins + b_wins
a_rate = a_wins / total_decisive

from scipy.stats import binomtest
binom_result = binomtest(a_wins, total_decisive, 0.5, alternative='two-sided')

print(f"  Model A wins: {a_wins:,} ({a_rate*100:.2f}%)")
print(f"  Model B wins: {b_wins:,} ({(1-a_rate)*100:.2f}%)")
print(f"  Binomial test p-value: {binom_result.pvalue:.6f}")
print(f"  Significant (p<0.05): {'YES' if binom_result.pvalue < 0.05 else 'NO'}")

results['position_bias'] = {
    'a_win_rate': float(a_rate),
    'b_win_rate': float(1 - a_rate),
    'p_value': float(binom_result.pvalue),
    'significant': bool(binom_result.pvalue < 0.05),
    'a_wins': int(a_wins),
    'b_wins': int(b_wins),
}

# ============================================================
# 10. ARENA MODE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Arena mode analysis")
print("=" * 60)

for mode in ['random', 'custom', 'big-vs-small', 'reasoning', 'small-models']:
    subset = battles_bt[battles_bt['mode_clean'] == mode]
    if len(subset) < 50:
        continue
    decisive_m = subset[subset['winner'].isin(['model_a', 'model_b'])]
    if len(decisive_m) < 30:
        continue
    a_rate_m = (decisive_m['winner'] == 'model_a').sum() / len(decisive_m)
    tie_rate_m = (subset['winner'] == 'tie').sum() / len(subset) if len(subset) > 0 else 0
    print(f"\n  {mode}: {len(subset):,} battles, {tie_rate_m*100:.1f}% ties, A win rate: {a_rate_m*100:.1f}%")

# ============================================================
# 11. SENSITIVITY: VOTES-ONLY vs COMBINED
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Sensitivity analysis (votes only vs combined)")
print("=" * 60)

votes_only = battles_bt[battles_bt['source'] == 'vote']
print(f"  Votes only: {len(votes_only):,} battles")

bt_votes_only, _ = compute_bt_ratings(votes_only, models)

# Compare with combined
v_vals = [bt_votes_only.get(m, INIT_RATING) for m in models_both]
c_combined_vals = [bt_standard.get(m, INIT_RATING) for m in models_both]
corr_sensitivity = np.corrcoef(v_vals, c_combined_vals)[0, 1]
print(f"  Correlation (votes-only vs combined): {corr_sensitivity:.4f}")

# Rank differences
votes_only_ranked = sorted(bt_votes_only.items(), key=lambda x: -x[1])
votes_only_rank = {m: i+1 for i, (m, _) in enumerate(votes_only_ranked)}

max_rank_diff = 0
for m in models:
    diff = abs(standard_rank.get(m, 999) - votes_only_rank.get(m, 999))
    if diff > max_rank_diff:
        max_rank_diff = diff

print(f"  Max rank difference: {max_rank_diff}")

results['sensitivity_votes_only'] = {
    'correlation': float(corr_sensitivity),
    'max_rank_diff': int(max_rank_diff),
    'n_battles': len(votes_only),
}

# ============================================================
# 12. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Bootstrap confidence intervals (n=1000)")
print("=" * 60)

N_BOOTSTRAP = 1000

# Bootstrap for style coefficients
print("  Bootstrapping style coefficients...")
boot_coefs = {feat: [] for feat in style_features}

for b in range(N_BOOTSTRAP):
    if (b+1) % 100 == 0:
        print(f"    Iteration {b+1}/{N_BOOTSTRAP}")

    boot_sample = battles_bt.sample(n=len(battles_bt), replace=True)
    try:
        _, boot_style_coefs, _, _ = compute_bt_style_controlled(boot_sample, models, style_features)
        for feat in style_features:
            boot_coefs[feat].append(boot_style_coefs[feat])
    except Exception:
        continue

print(f"\n  === Style Coefficient CIs (95%) ===")
style_ci_results = {}
style_pvalues = {}
for feat in style_features:
    coefs = np.array(boot_coefs[feat])
    ci_low = np.percentile(coefs, 2.5)
    ci_high = np.percentile(coefs, 97.5)
    point = style_coefs[feat]
    sig = (ci_low > 0) or (ci_high < 0)

    # Bootstrap p-value: proportion of bootstrap samples on wrong side of zero
    if point >= 0:
        p_val = 2 * max(np.mean(coefs <= 0), 1 / (len(coefs) + 1))
    else:
        p_val = 2 * max(np.mean(coefs >= 0), 1 / (len(coefs) + 1))
    style_pvalues[feat] = p_val

    odds_low = (np.exp(ci_low) - 1) * 100
    odds_high = (np.exp(ci_high) - 1) * 100
    odds_point = (np.exp(point) - 1) * 100

    print(f"    {feat:15s}: {point:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]  "
          f"({odds_point:+.1f}% [{odds_low:+.1f}%, {odds_high:+.1f}%])  "
          f"p={p_val:.4f}  {'***' if sig else 'n.s.'}")

    style_ci_results[feat] = {
        'point': float(point),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'odds_point': float(odds_point),
        'odds_ci_low': float(odds_low),
        'odds_ci_high': float(odds_high),
        'significant': bool(sig),
        'boot_std': float(coefs.std()),
        'p_value': float(p_val),
    }

# Apply Benjamini-Hochberg correction to style coefficient p-values
print(f"\n  === Style Coefficients: Benjamini-Hochberg Correction (m={len(style_features)}) ===")
feat_names = list(style_pvalues.keys())
feat_pvals = np.array([style_pvalues[f] for f in feat_names])

# BH procedure: sort p-values, compute adjusted p-values
bh_order = np.argsort(feat_pvals)
m_tests = len(feat_pvals)
bh_adjusted_style = np.zeros(m_tests)
for i, idx in enumerate(bh_order):
    rank = i + 1
    bh_adjusted_style[idx] = feat_pvals[idx] * m_tests / rank

# Enforce monotonicity (working backwards from largest rank)
for i in range(m_tests - 2, -1, -1):
    idx = bh_order[i]
    idx_next = bh_order[i + 1]
    bh_adjusted_style[idx] = min(bh_adjusted_style[idx], bh_adjusted_style[idx_next])
bh_adjusted_style = np.minimum(bh_adjusted_style, 1.0)

for i, feat in enumerate(feat_names):
    raw_p = feat_pvals[i]
    adj_p = bh_adjusted_style[i]
    sig_bh = adj_p < 0.05
    style_ci_results[feat]['p_value_bh'] = float(adj_p)
    style_ci_results[feat]['significant_bh'] = bool(sig_bh)
    print(f"    {feat:15s}: p_raw={raw_p:.4f}  p_BH={adj_p:.4f}  {'***' if sig_bh else 'n.s.'}")

results['style_coefficient_cis'] = style_ci_results

# Bootstrap for BT ratings (smaller sample for computational reasons)
print("\n  Bootstrapping BT ratings...")
N_BOOT_RATINGS = 1000
boot_ratings_std = {m: [] for m in models}
boot_ratings_ctrl = {m: [] for m in models}

for b in range(N_BOOT_RATINGS):
    if (b+1) % 50 == 0:
        print(f"    Iteration {b+1}/{N_BOOT_RATINGS}")

    boot_sample = battles_bt.sample(n=len(battles_bt), replace=True)
    try:
        br_std, _ = compute_bt_ratings(boot_sample, models)
        br_ctrl, _, _, _ = compute_bt_style_controlled(boot_sample, models, style_features)
        for m in models:
            boot_ratings_std[m].append(br_std.get(m, INIT_RATING))
            boot_ratings_ctrl[m].append(br_ctrl.get(m, INIT_RATING))
    except Exception:
        continue

# Compute CIs, p-values, and significance for ALL models
print(f"\n  === Rank Change Significance (all {len(models)} models) ===")
all_rank_significance = []

for _, row in rank_changes_df.iterrows():
    m = row['model']
    std_boots = np.array(boot_ratings_std[m])
    ctrl_boots = np.array(boot_ratings_ctrl[m])

    if len(std_boots) == 0 or len(ctrl_boots) == 0:
        continue

    diffs = ctrl_boots - std_boots
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    point_diff = row['rating_change']

    # Bootstrap p-value: two-sided test of H0: no rating change
    if point_diff >= 0:
        p_val = 2 * max(np.mean(diffs <= 0), 1 / (len(diffs) + 1))
    else:
        p_val = 2 * max(np.mean(diffs >= 0), 1 / (len(diffs) + 1))
    p_val = min(p_val, 1.0)

    all_rank_significance.append({
        'model': m,
        'rank_change': int(row['rank_change']),
        'rating_change': float(point_diff),
        'rating_diff_ci': [float(ci_low), float(ci_high)],
        'p_value': float(p_val),
    })

# Apply Benjamini-Hochberg correction across all models
print(f"\n  === Benjamini-Hochberg Correction (m={len(all_rank_significance)} models) ===")
raw_pvals = np.array([r['p_value'] for r in all_rank_significance])
n_models_tested = len(raw_pvals)

# BH procedure
bh_order = np.argsort(raw_pvals)
bh_adjusted = np.zeros(n_models_tested)
for i, idx in enumerate(bh_order):
    rank = i + 1
    bh_adjusted[idx] = raw_pvals[idx] * n_models_tested / rank

# Enforce monotonicity (step-up: working backwards from largest rank)
for i in range(n_models_tested - 2, -1, -1):
    idx = bh_order[i]
    idx_next = bh_order[i + 1]
    bh_adjusted[idx] = min(bh_adjusted[idx], bh_adjusted[idx_next])
bh_adjusted = np.minimum(bh_adjusted, 1.0)

# Store results and count
significant_raw = 0
significant_bh = 0
for i, entry in enumerate(all_rank_significance):
    sig_raw = raw_pvals[i] < 0.05
    sig_bh = bh_adjusted[i] < 0.05
    entry['significant_raw'] = bool(sig_raw)
    entry['p_value_bh'] = float(bh_adjusted[i])
    entry['significant_bh'] = bool(sig_bh)
    if sig_raw:
        significant_raw += 1
    if sig_bh:
        significant_bh += 1

# Print top 20 by absolute rank change
print(f"\n  {'Model':40s}  {'ΔRank':>6s}  {'p_raw':>8s}  {'p_BH':>8s}  {'Sig?':>5s}")
for entry in all_rank_significance[:20]:
    flag = '***' if entry['significant_bh'] else ('*' if entry['significant_raw'] else 'n.s.')
    print(f"  {entry['model']:40s}  {entry['rank_change']:+6d}  "
          f"{entry['p_value']:8.4f}  {entry['p_value_bh']:8.4f}  {flag:>5s}")

print(f"\n  Significant before correction: {significant_raw}/{n_models_tested}")
print(f"  Significant after BH (FDR<0.05): {significant_bh}/{n_models_tested}")

results['rank_significance'] = all_rank_significance
results['n_models_tested'] = n_models_tested
results['n_significant_raw'] = significant_raw
results['n_significant_bh'] = significant_bh
results['multiple_comparison_method'] = 'Benjamini-Hochberg (FDR < 0.05)'

# ============================================================
# 13. REACTION-DERIVED vs VOTE-DERIVED COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: Comparing reaction-derived vs vote-derived rankings")
print("=" * 60)

reaction_only = battles_bt[battles_bt['source'] == 'reaction']
if len(reaction_only) > 100:
    # Models that appear in both
    r_model_counts = pd.concat([reaction_only['model_a_name'], reaction_only['model_b_name']]).value_counts()
    r_models = sorted(r_model_counts[r_model_counts >= 30].index.tolist())
    common_models = sorted(set(r_models) & set(models))

    if len(common_models) >= 10:
        bt_reaction_only, _ = compute_bt_ratings(reaction_only, common_models)
        bt_vote_only, _ = compute_bt_ratings(votes_only, common_models)

        rv_vals = [bt_reaction_only.get(m, INIT_RATING) for m in common_models]
        vv_vals = [bt_vote_only.get(m, INIT_RATING) for m in common_models]

        corr_rv = np.corrcoef(rv_vals, vv_vals)[0, 1]
        print(f"  Models in both: {len(common_models)}")
        print(f"  Correlation (reaction BT vs vote BT): {corr_rv:.4f}")

        results['reaction_vs_vote_bt'] = {
            'n_common_models': len(common_models),
            'correlation': float(corr_rv),
            'n_reaction_battles': len(reaction_only),
        }
    else:
        print(f"  Not enough common models for comparison ({len(common_models)})")
else:
    print(f"  Not enough reaction-only battles ({len(reaction_only)})")

# ============================================================
# 14. ABLATION STUDY
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: Ablation study — one feature at a time")
print("=" * 60)

ablation_results = {}
for feat in style_features:
    single_feat = [feat]
    _, single_coefs, _, _ = compute_bt_style_controlled(battles_bt, models, single_feat)

    single_ratings, _, _, _ = compute_bt_style_controlled(battles_bt, models, single_feat)
    corr_single = np.corrcoef(
        [bt_standard.get(m, INIT_RATING) for m in models],
        [single_ratings.get(m, INIT_RATING) for m in models]
    )[0, 1]

    odds_pct = (np.exp(single_coefs[feat]) - 1) * 100
    print(f"  {feat:15s}: coef={single_coefs[feat]:+.4f} ({odds_pct:+.1f}%), "
          f"corr with standard={corr_single:.4f}")

    ablation_results[feat] = {
        'coefficient': float(single_coefs[feat]),
        'odds_change_pct': float(odds_pct),
        'correlation_with_standard': float(corr_single),
    }

# All features combined correlation
all_corr = np.corrcoef(
    [bt_standard.get(m, INIT_RATING) for m in models],
    [bt_controlled.get(m, INIT_RATING) for m in models]
)[0, 1]
print(f"  {'ALL':15s}: corr with standard={all_corr:.4f}")
ablation_results['ALL'] = {'correlation_with_standard': float(all_corr)}

results['ablation'] = ablation_results

# ============================================================
# 15. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 15: Saving results")
print("=" * 60)

# Save full rankings
results['rankings'] = {
    'standard': {m: {'rank': standard_rank[m], 'rating': float(bt_standard[m])} for m in models},
    'controlled': {m: {'rank': controlled_rank[m], 'rating': float(bt_controlled[m])} for m in models},
}

# Save to JSON
with open('clean_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  Saved to clean_analysis_results.json")
print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 60}")
