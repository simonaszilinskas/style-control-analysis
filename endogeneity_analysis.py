#!/usr/bin/env python3
"""
Endogeneity analysis: Does formatting bias vary by model quality tier?

Tests whether style effects are stronger for weaker models (confounder hypothesis)
or uniform across tiers (mediator hypothesis).

Approach:
1. Assign models to top/middle/bottom tiers by standard BT rating
2. Run style-controlled BT separately on within-tier and cross-tier battles
3. Test interaction effects in a unified logistic regression
"""

import json, re, sys, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

os.chdir('/Users/dinum-304878/dev/paper')
np.random.seed(42)

# ============================================================
# 1. LOAD PRE-COMPUTED RATINGS FOR TIER ASSIGNMENT
# ============================================================
print("Loading pre-computed rankings...")
with open('clean_analysis_results.json') as f:
    results = json.load(f)

standard_ratings_raw = results['rankings']['standard']
standard_ratings = {m: v['rating'] for m, v in standard_ratings_raw.items()}
print(f"  {len(standard_ratings)} models with standard BT ratings")

# Assign tiers (top/middle/bottom third)
sorted_models = sorted(standard_ratings.items(), key=lambda x: -x[1])
n = len(sorted_models)
tier_size = n // 3
tiers = {}
for i, (model, rating) in enumerate(sorted_models):
    if i < tier_size:
        tiers[model] = 'top'
    elif i < 2 * tier_size:
        tiers[model] = 'middle'
    else:
        tiers[model] = 'bottom'

tier_counts = defaultdict(int)
for t in tiers.values():
    tier_counts[t] += 1
print(f"  Tiers: top={tier_counts['top']}, middle={tier_counts['middle']}, bottom={tier_counts['bottom']}")

# Print tier boundaries
top_models = [(m, r) for m, r in sorted_models if tiers[m] == 'top']
mid_models = [(m, r) for m, r in sorted_models if tiers[m] == 'middle']
bot_models = [(m, r) for m, r in sorted_models if tiers[m] == 'bottom']
print(f"  Top tier: {top_models[0][0]} ({top_models[0][1]:.0f}) to {top_models[-1][0]} ({top_models[-1][1]:.0f})")
print(f"  Middle tier: {mid_models[0][0]} ({mid_models[0][1]:.0f}) to {mid_models[-1][0]} ({mid_models[-1][1]:.0f})")
print(f"  Bottom tier: {bot_models[0][0]} ({bot_models[0][1]:.0f}) to {bot_models[-1][0]} ({bot_models[-1][1]:.0f})")

# ============================================================
# 2. REBUILD BATTLES WITH STYLE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading and preparing battle data")
print("=" * 60)

# --- Style feature extraction functions ---
def count_markdown_headers(text):
    if not text: return 0
    return len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))

def count_markdown_lists(text):
    if not text: return 0
    return len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE)) + len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))

def count_markdown_bold(text):
    if not text: return 0
    return len(re.findall(r'\*\*[^*]+\*\*', text))

def count_code_blocks(text):
    if not text: return 0
    return len(re.findall(r'```|~~~', text)) // 2

def count_emojis(text):
    if not text: return 0
    return len(re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        "\U00002700-\U000027BF]+", flags=re.UNICODE
    ).findall(text))

def has_think_tags(text):
    if not text: return False
    return '<think>' in text or '</think>' in text

def has_reasoning_only(conv, role='assistant'):
    """Check if conversation has reasoning content but no visible content."""
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
    if conv is None: return ''
    texts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get('role') == role:
            content = msg.get('content', '') or ''
            if not has_think_tags(content):
                texts.append(content)
    return '\n'.join(texts)

# --- Model aliases ---
MODEL_ALIASES = {
    "mistral-medium-3.1": "mistral-medium-2508",
}

# --- Load votes ---
print("\nLoading votes...")
pf = pq.ParquetFile('comparia_votes.parquet')
votes = pf.read_row_group(0).to_pandas()
print(f"  Raw votes: {len(votes):,}")

# Apply model aliases
for old_name, new_name in MODEL_ALIASES.items():
    for col in ['model_a_name', 'model_b_name', 'chosen_model_name']:
        if col in votes.columns:
            votes[col] = votes[col].replace(old_name, new_name)

# Clean votes (matching original pipeline)
votes = votes[votes['model_a_name'] != votes['model_b_name']].copy()

# Remove no-choice votes
no_choice = votes['chosen_model_name'].isna() & (votes['both_equal'] != True)
votes = votes[~no_choice].copy()

# Deduplicate
dup_mask = votes.duplicated(subset='conversation_pair_id', keep='last')
votes = votes[~dup_mask].copy()

# Derive winner
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
votes_combined = votes[['conversation_pair_id', 'model_a_name', 'model_b_name', 'winner']].copy()
votes_combined['source'] = 'vote'
print(f"  Cleaned votes: {len(votes_combined):,}")

# --- Load reactions ---
print("Loading reactions...")
pf = pq.ParquetFile('comparia_reactions.parquet')
dfs = []
for i in range(pf.metadata.num_row_groups):
    dfs.append(pf.read_row_group(i).to_pandas())
reactions = pd.concat(dfs, ignore_index=True)
print(f"  Raw reactions: {len(reactions):,}")

# Apply model aliases to reactions
for old_name, new_name in MODEL_ALIASES.items():
    for col in ['model_a_name', 'model_b_name']:
        if col in reactions.columns:
            reactions[col] = reactions[col].replace(old_name, new_name)

# Clean reactions (matching original pipeline)
reactions = reactions[reactions['model_a_name'] != reactions['model_b_name']].copy()
reactions = reactions[reactions['msg_index'] % 2 == 1].copy()
reactions = reactions[reactions['response_content'].str.len() >= 10].copy()

# Convert to pairwise (matching original pipeline)
reaction_agg = reactions.groupby(
    ['conversation_pair_id', 'model_a_name', 'model_b_name', 'model_pos']
).agg(
    liked_sum=('liked', 'sum'),
    disliked_sum=('disliked', 'sum'),
    n_reactions=('liked', 'count')
).reset_index()

side_a = reaction_agg[reaction_agg['model_pos'] == 'a'].copy()
side_b = reaction_agg[reaction_agg['model_pos'] == 'b'].copy()
side_a = side_a.rename(columns={'liked_sum': 'liked_a', 'disliked_sum': 'disliked_a'})
side_b = side_b.rename(columns={'liked_sum': 'liked_b', 'disliked_sum': 'disliked_b'})

reaction_pairs = side_a[['conversation_pair_id', 'model_a_name', 'model_b_name', 'liked_a', 'disliked_a']].merge(
    side_b[['conversation_pair_id', 'liked_b', 'disliked_b']],
    on='conversation_pair_id', how='inner'
)

reaction_pairs['score_a'] = reaction_pairs['liked_a'] - reaction_pairs['disliked_a']
reaction_pairs['score_b'] = reaction_pairs['liked_b'] - reaction_pairs['disliked_b']

def reaction_winner(row):
    if row['score_a'] > row['score_b']: return 'model_a'
    elif row['score_b'] > row['score_a']: return 'model_b'
    return 'tie'

reaction_pairs['winner'] = reaction_pairs.apply(reaction_winner, axis=1)
reactions_combined = reaction_pairs[['conversation_pair_id', 'model_a_name', 'model_b_name', 'winner']].copy()
reactions_combined['source'] = 'reaction'
print(f"  Reaction-derived battles: {len(reactions_combined):,}")

# Remove overlapping IDs
overlap = set(votes_combined['conversation_pair_id']) & set(reactions_combined['conversation_pair_id'])
reactions_combined = reactions_combined[~reactions_combined['conversation_pair_id'].isin(overlap)]

battles = pd.concat([votes_combined, reactions_combined], ignore_index=True)
print(f"  Combined battles: {len(battles):,}")

# --- Extract style features ---
print("\nExtracting style features from votes...")
battle_cpids = set(battles[battles['source'] == 'vote']['conversation_pair_id'])
style_data = []
reasoning_only_cpids = set()

for idx in range(len(votes)):
    if idx % 30000 == 0:
        print(f"  Processing vote {idx:,}/{len(votes):,}...")
    cpid = votes['conversation_pair_id'].iloc[idx]
    if cpid not in battle_cpids:
        continue

    conv_a = votes['conversation_a'].iloc[idx]
    conv_b = votes['conversation_b'].iloc[idx]

    # Check for reasoning-only content (missing user-visible text)
    if has_reasoning_only(conv_a) or has_reasoning_only(conv_b):
        reasoning_only_cpids.add(cpid)

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
    })

print("Extracting style features from reactions...")
reaction_cpids = set(battles[battles['source'] == 'reaction']['conversation_pair_id'])
reaction_texts = {}
for _, row in reactions.iterrows():
    cpid = row['conversation_pair_id']
    if cpid not in reaction_cpids: continue
    side = row['model_pos']
    text = row['response_content'] or ''
    if cpid not in reaction_texts:
        reaction_texts[cpid] = {'a': '', 'b': ''}
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
    })

style_df = pd.DataFrame(style_data)
battles = battles.merge(style_df, on='conversation_pair_id', how='left')

# Filter out reasoning-only battles (missing visible content)
n_before = len(battles)
reasoning_mask = battles['conversation_pair_id'].isin(reasoning_only_cpids)
n_reasoning_only = reasoning_mask.sum()
print(f"  Reasoning-only battles removed: {n_reasoning_only:,} ({n_reasoning_only/n_before*100:.1f}%)")
battles = battles[~reasoning_mask].copy()

battles_styled = battles[battles['headers_a'].notna()].copy()
print(f"  Battles with style features: {len(battles_styled):,}")

# Filter to models with >= 100 battles
model_counts = pd.concat([battles_styled['model_a_name'], battles_styled['model_b_name']]).value_counts()
MIN_BATTLES = 100
models = sorted(model_counts[model_counts >= MIN_BATTLES].index.tolist())
battles_bt = battles_styled[
    battles_styled['model_a_name'].isin(models) &
    battles_styled['model_b_name'].isin(models)
].copy()
print(f"  Models with >= {MIN_BATTLES} battles: {len(models)}")
print(f"  Battles for BT: {len(battles_bt):,}")

# Save for future use
battles_bt.to_parquet('battles_bt_styled.parquet', index=False)
print("  Saved battles_bt_styled.parquet for future analyses")

# ============================================================
# 3. BT MODEL FUNCTIONS
# ============================================================
SCALE = 400
BASE = 10
INIT_RATING = 1000

style_features = ['headers', 'lists', 'bold', 'code_blocks', 'emoji']

def compute_style_coefficients(battles_df, models_list):
    """Compute style-controlled BT and return style coefficients"""
    model_to_idx = {m: i for i, m in enumerate(models_list)}
    n_m = len(models_list)
    n_style = len(style_features)

    decisive = battles_df[battles_df['winner'].isin(['model_a', 'model_b'])].copy()
    decisive = decisive.dropna(subset=[f'{sf}_a' for sf in style_features])

    X_model = np.zeros((len(decisive), n_m))
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

    scaler = StandardScaler()
    X_style_scaled = scaler.fit_transform(X_style)
    X = np.hstack([X_model, X_style_scaled])

    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)

    style_coefs = dict(zip(style_features, lr.coef_[0][n_m:]))
    return style_coefs, len(decisive)


# ============================================================
# 4. TIER ASSIGNMENT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Tier-stratified analysis")
print("=" * 60)

# Add tiers to battles
battles_bt['tier_a'] = battles_bt['model_a_name'].map(tiers)
battles_bt['tier_b'] = battles_bt['model_b_name'].map(tiers)

# Drop rows where either model doesn't have a tier (shouldn't happen)
battles_bt = battles_bt.dropna(subset=['tier_a', 'tier_b'])

# Classify battle pairs
def classify_pair(row):
    tiers_pair = sorted([row['tier_a'], row['tier_b']])
    return f"{tiers_pair[0]}-{tiers_pair[1]}"

battles_bt['pair_tier'] = battles_bt.apply(classify_pair, axis=1)
pair_counts = battles_bt['pair_tier'].value_counts()
print("\n  Battle pair tier distribution:")
for pt, count in pair_counts.items():
    pct = count / len(battles_bt) * 100
    print(f"    {pt:20s}: {count:,} ({pct:.1f}%)")


# ============================================================
# 5. STYLE COEFFICIENTS BY PAIR TIER
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Style coefficients by battle pair tier")
print("=" * 60)

tier_results = {}

# Within-tier analyses
for tier in ['top', 'middle', 'bottom']:
    subset = battles_bt[(battles_bt['tier_a'] == tier) & (battles_bt['tier_b'] == tier)]
    tier_models = sorted(set(subset['model_a_name']) | set(subset['model_b_name']))

    # Need enough models to fit
    if len(tier_models) < 3:
        print(f"\n  {tier}-{tier}: skipped (only {len(tier_models)} models)")
        continue

    coefs, n_battles = compute_style_coefficients(subset, tier_models)
    tier_results[f'{tier}-{tier}'] = {'coefs': coefs, 'n_battles': n_battles, 'n_models': len(tier_models)}

    print(f"\n  {tier}-{tier} ({n_battles:,} decisive battles, {len(tier_models)} models):")
    for feat in ['bold', 'lists', 'headers']:
        odds_pct = (np.exp(coefs[feat]) - 1) * 100
        print(f"    {feat:10s}: {coefs[feat]:+.4f} ({odds_pct:+.1f}% odds)")

# Cross-tier analyses
for tier_pair in [('top', 'middle'), ('top', 'bottom'), ('middle', 'bottom')]:
    t1, t2 = tier_pair
    subset = battles_bt[
        ((battles_bt['tier_a'] == t1) & (battles_bt['tier_b'] == t2)) |
        ((battles_bt['tier_a'] == t2) & (battles_bt['tier_b'] == t1))
    ]
    tier_models = sorted(set(subset['model_a_name']) | set(subset['model_b_name']))

    if len(tier_models) < 3:
        print(f"\n  {t1}-{t2}: skipped (only {len(tier_models)} models)")
        continue

    coefs, n_battles = compute_style_coefficients(subset, tier_models)
    tier_results[f'{t1}-{t2}'] = {'coefs': coefs, 'n_battles': n_battles, 'n_models': len(tier_models)}

    print(f"\n  {t1}-{t2} ({n_battles:,} decisive battles, {len(tier_models)} models):")
    for feat in ['bold', 'lists', 'headers']:
        odds_pct = (np.exp(coefs[feat]) - 1) * 100
        print(f"    {feat:10s}: {coefs[feat]:+.4f} ({odds_pct:+.1f}% odds)")


# ============================================================
# 6. INTERACTION MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Interaction model (full logistic regression with tier × style)")
print("=" * 60)

# For each battle, compute style diffs and assign the average tier
# Using the "higher-ranked model's tier" as the battle context
decisive = battles_bt[battles_bt['winner'].isin(['model_a', 'model_b'])].copy()
decisive = decisive.dropna(subset=[f'{sf}_a' for sf in style_features])

# Compute style diffs
for feat in style_features:
    decisive[f'{feat}_diff'] = decisive[f'{feat}_a'] - decisive[f'{feat}_b']
    # Standardize
    mean_val = decisive[f'{feat}_diff'].mean()
    std_val = decisive[f'{feat}_diff'].std()
    if std_val > 0:
        decisive[f'{feat}_diff_z'] = (decisive[f'{feat}_diff'] - mean_val) / std_val
    else:
        decisive[f'{feat}_diff_z'] = 0

decisive['y'] = (decisive['winner'] == 'model_a').astype(int)

# Use pair_tier as the grouping variable
# Simplify to: same_tier (within) vs cross_tier (between)
# And also: highest_tier in the pair
decisive['is_within_tier'] = (decisive['tier_a'] == decisive['tier_b']).astype(int)

# For the interaction: does the best model in the pair being "top" reduce style effects?
tier_rank = {'top': 3, 'middle': 2, 'bottom': 1}
decisive['best_tier_rank'] = decisive.apply(
    lambda r: max(tier_rank[r['tier_a']], tier_rank[r['tier_b']]), axis=1
)
decisive['worst_tier_rank'] = decisive.apply(
    lambda r: min(tier_rank[r['tier_a']], tier_rank[r['tier_b']]), axis=1
)

# Build interaction model: model indicators + style_diff_z + style_diff_z * tier indicators
model_to_idx = {m: i for i, m in enumerate(models)}
n_m = len(models)

# Model indicators
X_model = np.zeros((len(decisive), n_m))
for i, (_, row) in enumerate(decisive.iterrows()):
    a_idx = model_to_idx.get(row['model_a_name'])
    b_idx = model_to_idx.get(row['model_b_name'])
    if a_idx is not None:
        X_model[i, a_idx] = 1
    if b_idx is not None:
        X_model[i, b_idx] = -1

valid = np.any(X_model != 0, axis=1)
X_model = X_model[valid]
decisive_valid = decisive[valid].reset_index(drop=True)

# Style features (standardized)
X_style = decisive_valid[[f'{sf}_diff_z' for sf in style_features]].values

# Tier indicators for interaction (reference = bottom)
# is_top_pair: both models in top tier
# is_mid_pair: both in middle
# is_cross: different tiers
decisive_valid['is_top_pair'] = ((decisive_valid['tier_a'] == 'top') & (decisive_valid['tier_b'] == 'top')).astype(float)
decisive_valid['is_mid_pair'] = ((decisive_valid['tier_a'] == 'middle') & (decisive_valid['tier_b'] == 'middle')).astype(float)
decisive_valid['is_bot_pair'] = ((decisive_valid['tier_a'] == 'bottom') & (decisive_valid['tier_b'] == 'bottom')).astype(float)

# Interaction: style_diff * is_top_pair, style_diff * is_mid_pair
# (bottom-bottom is reference, cross-tier captured by not being any of these)
# Simplify: just use the 3 significant features for interactions
sig_features = ['bold', 'lists', 'headers']
sig_indices = [style_features.index(sf) for sf in sig_features]

X_interact_top = X_style[:, sig_indices] * decisive_valid['is_top_pair'].values[:, np.newaxis]
X_interact_mid = X_style[:, sig_indices] * decisive_valid['is_mid_pair'].values[:, np.newaxis]

# Full design matrix: model indicators + style + top interactions + mid interactions
X_full = np.hstack([X_model, X_style, X_interact_top, X_interact_mid])
y_full = decisive_valid['y'].values

print(f"  Design matrix: {X_full.shape[0]:,} battles × {X_full.shape[1]} features")
print(f"    Model indicators: {n_m}")
print(f"    Style features: {len(style_features)}")
print(f"    Tier×style interactions: {len(sig_features) * 2}")

lr_interact = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
lr_interact.fit(X_full, y_full)

# Extract coefficients
style_coefs_main = lr_interact.coef_[0][n_m:n_m+len(style_features)]
interact_top_coefs = lr_interact.coef_[0][n_m+len(style_features):n_m+len(style_features)+len(sig_features)]
interact_mid_coefs = lr_interact.coef_[0][n_m+len(style_features)+len(sig_features):]

print("\n  === Main style effects (reference: bottom-bottom + cross-tier) ===")
for i, feat in enumerate(style_features):
    odds_pct = (np.exp(style_coefs_main[i]) - 1) * 100
    print(f"    {feat:15s}: {style_coefs_main[i]:+.4f} ({odds_pct:+.1f}% odds)")

print("\n  === Interaction: top-top pair × style (additional effect for top-tier battles) ===")
for i, feat in enumerate(sig_features):
    print(f"    {feat:15s}: {interact_top_coefs[i]:+.4f}")

print("\n  === Interaction: mid-mid pair × style (additional effect for mid-tier battles) ===")
for i, feat in enumerate(sig_features):
    print(f"    {feat:15s}: {interact_mid_coefs[i]:+.4f}")

# Implied coefficients by tier
print("\n  === Implied total style effects by pair tier ===")
print(f"  {'Feature':15s}  {'Bottom-Bot':>12s}  {'Middle-Mid':>12s}  {'Top-Top':>12s}  {'Cross-tier':>12s}")
for i, feat in enumerate(sig_features):
    base = style_coefs_main[style_features.index(feat)]
    top_total = base + interact_top_coefs[i]
    mid_total = base + interact_mid_coefs[i]
    bot_total = base  # reference
    cross_total = base  # also reference (not interacted)

    top_odds = (np.exp(top_total) - 1) * 100
    mid_odds = (np.exp(mid_total) - 1) * 100
    bot_odds = (np.exp(bot_total) - 1) * 100
    cross_odds = (np.exp(cross_total) - 1) * 100

    print(f"  {feat:15s}  {bot_odds:+10.1f}%  {mid_odds:+10.1f}%  {top_odds:+10.1f}%  {cross_odds:+10.1f}%")


# ============================================================
# 7. BOOTSTRAP CIS FOR INTERACTION TERMS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Bootstrap CIs for interaction terms (1000 iterations)")
print("=" * 60)

N_BOOT = 1000
boot_interact_top = np.zeros((N_BOOT, len(sig_features)))
boot_interact_mid = np.zeros((N_BOOT, len(sig_features)))
boot_style_main = np.zeros((N_BOOT, len(style_features)))

for b in range(N_BOOT):
    if b % 50 == 0:
        print(f"  Bootstrap iteration {b}/{N_BOOT}...")

    # Resample battles
    idx = np.random.choice(len(X_full), size=len(X_full), replace=True)
    X_b = X_full[idx]
    y_b = y_full[idx]

    try:
        lr_b = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
        lr_b.fit(X_b, y_b)
        boot_style_main[b] = lr_b.coef_[0][n_m:n_m+len(style_features)]
        boot_interact_top[b] = lr_b.coef_[0][n_m+len(style_features):n_m+len(style_features)+len(sig_features)]
        boot_interact_mid[b] = lr_b.coef_[0][n_m+len(style_features)+len(sig_features):]
    except Exception as e:
        print(f"  Bootstrap {b} failed: {e}")
        boot_style_main[b] = np.nan
        boot_interact_top[b] = np.nan
        boot_interact_mid[b] = np.nan

# Print CIs for interaction terms
print("\n  === Bootstrap 95% CIs for interaction terms ===")
print(f"  {'Term':30s}  {'Point':>8s}  {'CI low':>8s}  {'CI high':>8s}  {'Sig?':>5s}")

for i, feat in enumerate(sig_features):
    # Top interaction
    point = interact_top_coefs[i]
    ci = np.nanpercentile(boot_interact_top[:, i], [2.5, 97.5])
    sig = 'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'
    print(f"  top×{feat:24s}  {point:+8.4f}  {ci[0]:+8.4f}  {ci[1]:+8.4f}  {sig:>5s}")

    # Mid interaction
    point = interact_mid_coefs[i]
    ci = np.nanpercentile(boot_interact_mid[:, i], [2.5, 97.5])
    sig = 'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'
    print(f"  mid×{feat:24s}  {point:+8.4f}  {ci[0]:+8.4f}  {ci[1]:+8.4f}  {sig:>5s}")


# ============================================================
# 8. ADDITIONAL TEST: CORRELATION BETWEEN MODEL QUALITY AND FORMATTING
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Correlation between model quality and formatting intensity")
print("=" * 60)

# For each model, compute average style feature values
model_style_means = {}
for model in models:
    as_a = battles_bt[battles_bt['model_a_name'] == model]
    as_b = battles_bt[battles_bt['model_b_name'] == model]

    for feat in ['bold', 'lists', 'headers']:
        vals = pd.concat([as_a[f'{feat}_a'], as_b[f'{feat}_b']])
        if model not in model_style_means:
            model_style_means[model] = {}
        model_style_means[model][feat] = vals.mean()

# Compute composite formatting index
for model in models:
    model_style_means[model]['composite'] = sum(
        model_style_means[model][f] for f in ['bold', 'lists', 'headers']
    )

# Correlation with standard BT rating
models_with_both = [m for m in models if m in standard_ratings and m in model_style_means]
ratings_arr = np.array([standard_ratings[m] for m in models_with_both])
composite_arr = np.array([model_style_means[m]['composite'] for m in models_with_both])

r_corr, p_corr = stats.pearsonr(ratings_arr, composite_arr)
r_spearman, p_spearman = stats.spearmanr(ratings_arr, composite_arr)

print(f"\n  Model quality vs. formatting intensity (n={len(models_with_both)} models):")
print(f"    Pearson r  = {r_corr:.3f} (p = {p_corr:.4f})")
print(f"    Spearman ρ = {r_spearman:.3f} (p = {p_spearman:.4f})")

# Break down by feature
for feat in ['bold', 'lists', 'headers']:
    feat_arr = np.array([model_style_means[m][feat] for m in models_with_both])
    r, p = stats.pearsonr(ratings_arr, feat_arr)
    print(f"    {feat:10s}: r = {r:.3f} (p = {p:.4f})")

# Correlation with CONTROLLED rating
ctrl_ratings_raw = results['rankings']['controlled']
ctrl_ratings = {m: v['rating'] for m, v in ctrl_ratings_raw.items()}
ctrl_arr = np.array([ctrl_ratings[m] for m in models_with_both], dtype=float)
r_ctrl, p_ctrl = stats.pearsonr(ctrl_arr, composite_arr)
print(f"\n  Controlled rating vs. formatting:")
print(f"    Pearson r  = {r_ctrl:.3f} (p = {p_ctrl:.4f})")

# Rating CHANGE vs formatting intensity
rating_change_arr = ctrl_arr - ratings_arr
r_change, p_change = stats.pearsonr(rating_change_arr, composite_arr)
print(f"\n  Rating change (ctrl - std) vs. formatting intensity:")
print(f"    Pearson r  = {r_change:.3f} (p = {p_change:.4f})")
print(f"    (Negative r means formatting-heavy models lose rating after style control)")


# ============================================================
# 9. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("Saving endogeneity analysis results")
print("=" * 60)

endo_results = {
    'tier_stratified_coefficients': {
        tier_name: {
            'n_battles': info['n_battles'],
            'n_models': info['n_models'],
            'coefficients': {feat: float(info['coefs'][feat]) for feat in style_features},
            'odds_change_pct': {feat: float((np.exp(info['coefs'][feat]) - 1) * 100) for feat in ['bold', 'lists', 'headers']}
        }
        for tier_name, info in tier_results.items()
    },
    'interaction_model': {
        'main_style_effects': {feat: float(style_coefs_main[i]) for i, feat in enumerate(style_features)},
        'top_interactions': {feat: float(interact_top_coefs[i]) for i, feat in enumerate(sig_features)},
        'mid_interactions': {feat: float(interact_mid_coefs[i]) for i, feat in enumerate(sig_features)},
    },
    'quality_formatting_correlation': {
        'pearson_r': float(r_corr),
        'pearson_p': float(p_corr),
        'spearman_rho': float(r_spearman),
        'spearman_p': float(p_spearman),
        'ctrl_rating_vs_formatting_r': float(r_ctrl),
        'rating_change_vs_formatting_r': float(r_change),
        'rating_change_vs_formatting_p': float(p_change),
    }
}

with open('endogeneity_results.json', 'w') as f:
    json.dump(endo_results, f, indent=2)

print("\nDone! Results saved to endogeneity_results.json")
