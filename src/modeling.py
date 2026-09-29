"""Shared statistical primitives for the style-control analyses.

This module is the single implementation of the Bradley--Terry design used by
the executable analyses.  Callers choose whether feature contrasts are
winsorized; the core formatting analysis retains its original untrimmed
specification, while the joint/linguistic analyses use 1/99% winsorization.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCALE, BASE, INIT_RATING = 400, 10, 1000
MIN_BATTLES = 100

FORMATTING = ["headers", "lists", "bold", "code_blocks", "emoji"]
READABILITY = ["rel", "cli", "fkg"]
DIVERSITY = ["ttr", "mattr"]
STRUCTURE = ["asl", "long_sent_ratio"]
LINGUISTIC = READABILITY + DIVERSITY + STRUCTURE


def feature_contrasts(battles, features, *, winsorize=True):
    """Return unstandardized A-minus-B feature contrasts.

    Winsorization is computed independently for each supplied sample, matching
    the historical analysis behavior (including bootstrap refits).
    """
    columns = []
    for feature in features:
        difference = (
            battles[f"{feature}_a"].to_numpy(float)
            - battles[f"{feature}_b"].to_numpy(float)
        )
        if winsorize:
            low, high = np.nanpercentile(difference, [1, 99])
            difference = np.clip(difference, low, high)
        columns.append(difference)
    return np.column_stack(columns)


def design_matrix(battles, models, features, *, winsorize=True):
    """Build model contrasts, standardized feature contrasts, and outcomes."""
    data = battles[battles["winner"].isin(["model_a", "model_b"])]
    if features:
        data = data.dropna(
            subset=[f"{feature}_{side}" for feature in features for side in ("a", "b")]
        )
    data = data[
        data["model_a_name"].isin(models) & data["model_b_name"].isin(models)
    ]

    model_index = {model: index for index, model in enumerate(models)}
    model_design = np.zeros((len(data), len(models)))
    rows = np.arange(len(data))
    model_design[rows, data["model_a_name"].map(model_index).to_numpy()] = 1
    model_design[rows, data["model_b_name"].map(model_index).to_numpy()] = -1
    outcomes = (data["winner"].to_numpy() == "model_a").astype(float)

    if features:
        style_design = StandardScaler().fit_transform(
            feature_contrasts(data, features, winsorize=winsorize)
        )
    else:
        style_design = np.empty((len(data), 0))
    return model_design, style_design, outcomes


def fit_bt(battles, models, features, *, winsorize=True):
    """Fit Bradley--Terry model strengths plus standardized style contrasts."""
    model_design, style_design, outcomes = design_matrix(
        battles, models, features, winsorize=winsorize
    )
    design = np.hstack([model_design, style_design])
    estimator = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    estimator.fit(design, outcomes)

    n_models = len(models)
    model_coefs = estimator.coef_[0][:n_models]
    model_coefs = model_coefs - model_coefs.mean()
    ratings = dict(
        zip(models, INIT_RATING + SCALE * model_coefs / np.log(BASE))
    )
    coefficients = dict(zip(features, estimator.coef_[0][n_models:]))
    return ratings, coefficients, outcomes, estimator, design


def fit_pooled(battles, models, features, *, winsorize=True):
    """Fit style contrasts and an intercept without per-model strengths."""
    _, style_design, outcomes = design_matrix(
        battles, models, features, winsorize=winsorize
    )
    design = np.column_stack([np.ones(len(style_design)), style_design])
    estimator = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    estimator.fit(design, outcomes)
    return dict(zip(features, estimator.coef_[0][1:]))


def benjamini_hochberg(p_values):
    """Return Benjamini--Hochberg adjusted p-values in input order."""
    values = np.asarray(p_values, float)
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count)
    for rank, index in enumerate(order, start=1):
        adjusted[index] = values[index] * count / rank
    for index in range(count - 2, -1, -1):
        adjusted[order[index]] = min(
            adjusted[order[index]], adjusted[order[index + 1]]
        )
    return np.minimum(adjusted, 1.0)


def log_likelihood(outcomes, design, estimator):
    probabilities = np.clip(estimator.predict_proba(design)[:, 1], 1e-12, 1 - 1e-12)
    return float(
        np.sum(outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities))
    )


def ranks(ratings):
    return {
        model: rank
        for rank, model in enumerate(
            sorted(ratings, key=lambda model: -ratings[model]), start=1
        )
    }
