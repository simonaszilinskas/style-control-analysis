# What Wins a Vote? Formatting, Length, and Lexical Diversity in the French Compar:IA LLM Arena

---

## Abstract

LLM evaluation arenas, where users compare two model outputs side-by-side, have become a primary source of model rankings, and a standing worry is that these rankings reward *presentation* over substance. We study this on Compar:IA, a French government-backed arena, using revision `8cd6488` of its consolidated `comparia-fr-arena` release (about 138,000 decisive French votes across 116 models). We decompose preference into model identity and presentation with a style-controlled Bradley-Terry model, and go beyond markdown formatting to a full presentation account: **formatting** (bold, lists, headers, code, emoji), **length**, and **linguistic** properties of the text (readability, lexical diversity, sentence structure). Because the features are observed rather than manipulated, we report conditional associations, not causal effects. Four findings emerge. First, presentation is systematically associated with votes, but much of it behaves as a collinear "verbosity" bundle: length, bold, and lists rise and fall together, and their individual coefficients are specification-dependent. Second, two signals stand out from that bundle and survive the full joint control: **bold usage** (about +10% higher win odds per standard deviation) and **length-independent lexical diversity** (MATTR, about +18%); readability metrics are collinear and mostly weak, and raw type-token ratio is largely length in disguise. Third, much of the *apparent* linguistic association is between models rather than within models: several coefficients shrink substantially when per-model fixed effects are included, although that decomposition does not identify whether presentation is bias, a quality signal, or both. Fourth, in an external comparison on exact shared model identifiers, the narrower formatting-controlled ranking correlates most strongly with each of four LMArena rankings (Spearman 0.770–0.806), but its improvement over the raw Compar:IA ranking is not statistically decisive; the full joint adjustment usually correlates less strongly. Adjusting for the full presentation bundle reshuffles the leaderboard (rating correlation 0.94; 37 of 116 models move by ≥10 ranks). A timing audit, however, finds that the completed-conversation fields used by the present pipeline contain turns after the retained vote in 11.2% of battles; the conversation-depth result is therefore not interpretable as pre-vote engagement, and all estimates should be replicated on vote-truncated prefixes before publication. The practical message is more qualified than "adjust everything": arena operators should report the raw ranking alongside clearly labelled sensitivity analyses, because neither the raw nor an adjusted ranking is a pure measure of capability.

---

## 1. Introduction

The rise of LLM evaluation arenas, platforms where users interact with two anonymous models and select a preferred response, has established a new paradigm for model comparison. The LMSYS Chatbot Arena pioneered this approach (Chiang et al., 2024). Its rankings are now Bradley-Terry estimates, rather than online Elo scores, and are widely used as summaries of human preference. They are not direct technical-performance scores: Compar:IA itself explicitly describes its ranking as subjective preference rather than factuality or truth (Compar:IA, 2026a). The methodology has since been adopted by multiple platforms, including Compar:IA, a French government-backed arena opened to the public in October 2024 (Ministère de la Culture, 2025).

A key concern with arena-based evaluation is the extent to which user preferences reflect content quality versus presentation. The evidence differs by evaluator: work on LLM judges and reward models documents position, length, and verbosity biases (Zheng et al., 2023; Singhal et al., 2024; Saito et al., 2023; Dubois et al., 2024), while Wu and Aji (2025) find that crowd and expert judgements can penalise shortness or grammatical errors more than factual errors in deliberately flawed answers. The LMSYS team subsequently introduced "style control" (Li et al., 2024), adding response length and markdown contrasts to a Bradley-Terry model (Bradley & Terry, 1952). Their observational analysis found that controlling for those features reshuffled the English-language arena ranking, while explicitly warning that unobserved substantive quality may remain correlated with style.

Most style-control work stops at markdown and length. But "presentation" is broader: how readable the prose is, how varied its vocabulary, how long its sentences, all things a user can respond to without them tracking correctness. These linguistic features are correlated with formatting and length, so studying any one in isolation risks mis-attributing a shared effect. We therefore analyze all three families together, **formatting** (bold, lists, headers, code, emoji), **length**, and **linguistic** properties (readability, lexical diversity, sentence structure), in a single style-controlled model, and ask which presentation features carry an *independent association* with the vote once the others, and model identity, are held fixed. Throughout, our estimates are conditional associations, not the effect of manipulating a feature while holding substance fixed; §5.3 discusses why.

One tentative way to organise these features is by the level of reading they might act on. Text-comprehension research distinguishes a surface reading of the words (Rayner, 1998) from deeper processing of a text's structure and meaning (Kintsch & van Dijk, 1978; van Dijk & Kintsch, 1983; Just & Carpenter, 1980). Adapting that loosely to an arena vote, an evaluation might engage at three levels: (a) a glance at the *shape* of the answer; (b) a reading of its *argument*; and (c) the *words themselves*. Our feature families map roughly onto these levels, formatting to (a), length to (b) as a crude size proxy, and lexical diversity and readability to (c), though the mapping is imperfect (length and diversity are both surface-text statistics, not clean levels). We offer this as an interpretive hypothesis, not a validated theory. Section 4.6 explores completed conversation length as heterogeneity, but a timing audit shows that it cannot serve as a clean measure of engagement before the vote.

We apply this analysis to the Compar:IA data, which provides a strong setting:

1. **Scale:** about 138,000 cleaned decisive battles across 116 models, from organic user interactions.
2. **Language:** French-language evaluation, a setting in which presentation preferences have not, to our knowledge, been studied this way.
3. **Native metadata:** each battle carries an LLM-assigned topic taxonomy and the completed multi-turn conversation, which lets us adjust for subject matter and audit the timing of reactions relative to later turns.
4. **Model breadth:** the roster spans model families and providers, allowing us to separate presentation differences between models from variation within a model.

Our central question is: **which presentation features retain an association with votes after jointly adjusting for the other measured features and model identity, and how much do the resulting rankings move?** Our contributions are: (i) a joint presentation-adjusted ranking that separates formatting, length, and linguistic associations from one another and from between-model differences; (ii) the finding that presentation behaves largely as a collinear verbosity bundle, with bold usage and length-independent lexical diversity standing apart as stable signals; (iii) a pooled-versus-fixed-effect decomposition showing that much apparent linguistic association is between models, without claiming that this identifies a causal mechanism; (iv) an external comparison showing that narrow formatting control, but not the full joint control, has the strongest correlation with contemporary LMArena rankings on the matched models; and (v) a timing audit that identifies post-vote look-ahead in the completed-conversation features and sharply limits the conversation-depth analysis.

---

## 2. Data

### 2.1 The Compar:IA Platform

Compar:IA is an LLM evaluation arena operated by the French Government's Ministry of Culture and Direction interministérielle du numérique (DINUM). Users submit prompts and receive responses from two anonymous models side-by-side, may continue the conversation over several turns, and then vote for a winner or declare a tie. Models are identified only after voting.

The platform offers several arena modes; in our data the decisive votes come from **random** (72%, random model pairs), **custom** (19%, user-selected pairs), **big-vs-small** (8%, deliberately pairing large and small models), and **small-models** (2%).

### 2.2 Dataset

We use **`ministere-culture/comparia-fr-arena`** (compar:IA & Ministère de la Culture, 2026b), the consolidated Compar:IA release published on Hugging Face under Open Licence 2.0 (Etalab) and CC-BY-4.0. We pin revision **`8cd6488c5d0c3b8dfcb9339d11ae9624c84359be`** (published and accessed 8 July 2026). The gated release contains 641,277 turn rows and roughly 208K human reactions across 115+ models before our French/decisive filtering. It is organised by turn: one row per conversation turn, and `choice` records any reaction made on that turn. A small number of conversations contain multiple decisive reactions; we retain the last decisive reaction per `comparison_id`, yielding one battle per conversation.

| | Raw turns | Decisive French battles | Models (≥100 battles) |
|---|---:|---:|---:|
| comparia-fr-arena | 641,277 | 137,217 | 116 |

Ties and no-vote turns are dropped (decisive votes only, matching prior style-control work). Winner distribution is balanced (model A wins 49.9%, see §4.4).

### 2.3 Feature extraction and data quality

For each battle the current pipeline concatenates assistant messages from `full_conversation_a` and `full_conversation_b` and computes all features on that text. The source repeats the **completed** conversation on its turn rows; it is not necessarily truncated at the retained reaction. Formatting features use the markdown regexes below; length is the completed conversation's cumulative output-token count from the dataset metadata; the linguistic features are the set designed by Maayeesha Farzana (PSL) in a companion internship analysis, recomputed here using her formulas. Battles whose response text is empty (for example when only hidden reasoning was recorded, or content contained `<think>` leakage) are dropped. Topic and completed turn count come directly from the dataset. Every feature is present for essentially all battles (topic 100%, length 99.9%, linguistic 97.5%). Section 2.4 quantifies the resulting timing limitation.

### 2.4 Native Topic Metadata and Vote-Timing Audit

Each conversation carries an LLM-assigned **topic taxonomy** (`categories`, about 18 subject classes), which we use to control for subject matter (§4.7). It also carries the turn index of each reaction and the completed conversation. Comparing the retained decisive-reaction turn with the final turn reveals that **15,352 of 137,217 analysed battles (11.2%) contain one or more later turns in the text used for feature extraction**. Moreover, 10,593 of the 26,137 battles classified as multi-turn from the completed conversation (40.5%) were still single-turn when the retained vote occurred. The audit is reproduced by `src/audit_vote_timing.py`. Consequently, §4.6 describes heterogeneity by completed conversation length, not reading depth before the vote, and the headline coefficient and ranking analyses require replication on vote-truncated prefixes before publication.

---

## 3. Methodology

### 3.1 Presentation Features

We measure presentation in three families, all computed per response.

**Formatting**, five markdown features:

| Feature | Description | Regex Pattern |
|---------|-------------|---------------|
| **Headers** | Markdown headers (# through ######) | `^#{1,6}\s` (multiline) |
| **Lists** | Ordered and unordered list items | `^\s*[-*+]\s` and `^\s*\d+\.\s` |
| **Bold** | Bold-formatted text | `\*\*[^*]+\*\*` |
| **Code blocks** | Fenced code blocks | ` ``` ` or `~~~` (counted in pairs) |
| **Emoji** | Emoji characters | Unicode emoji ranges |

**Length**, the response's cumulative output-token count from the dataset metadata.

**Linguistic**, the feature set designed by Maayeesha Farzana, covering text properties beyond formatting: readability (Kandel-Moles REL, calibrated for French; Coleman-Liau; Flesch-Kincaid grade), lexical diversity (type-token ratio TTR and its length-robust moving-average variant MATTR, over a 50-token window), and sentence structure (mean sentence length and the fraction of long sentences).

Our primary formatting analysis (§4.1–§4.4) uses the five markdown features only, making it structurally comparable to English-language style control. The numerical coefficients are not directly comparable: Li et al. (2024) standardise a relative difference, $(f_A-f_B)/(f_A+f_B)$, whereas we standardise the raw A−B contrast (and include code and emoji). We hold length and the linguistic family back to §4.5 for a specific reason: **length is entangled with completeness**, a genuine quality dimension. Users may prefer more thorough answers, so controlling for length can remove legitimate quality signal rather than bias. §4.5 confronts that trade-off by adding length and all linguistic features to the same model.

### 3.2 Bradley-Terry Model

We rank models using a Bradley-Terry model estimated via logistic regression, following the methodology of the LMSYS Chatbot Arena.

**Standard model.** For each decisive battle we construct a feature vector with +1 for model A's index and −1 for model B's index, set the binary outcome to 1 when A wins and 0 when B wins, and fit a logistic regression without regularization. Because only coefficient differences are identified, the 1000-point rating origin is arbitrary; ranks and pairwise probabilities are invariant to a common shift. Ratings are computed as:

$$\text{Rating}_i = 1000 + \frac{400 \cdot \beta_i}{\ln(10)}$$

**Style-controlled model.** We augment the model indicator features with style difference features. For each feature $f$, we compute $\Delta f = f_A - f_B$ (standardized to zero mean and unit variance), then fit:

$$P(\text{A wins}) = \sigma\left(\sum_i \beta_i \cdot \mathbb{1}_i + \sum_f \gamma_f \cdot \Delta f\right)$$

where $\beta_i$ are per-model fixed effects (a residual model-preference term, not skill per se, since they absorb every unmeasured model characteristic) and $\gamma_f$ are style coefficients. For the joint model (§4.5) the $\Delta f$ contrasts are additionally winsorized at the 1/99th percentile before standardizing, because several linguistic features are heavy-tailed and a handful of extreme battles would otherwise drive the coefficient.

### 3.3 Bootstrap Inference

We computed 95% confidence intervals via nonparametric bootstrap (Efron, 1979), with 1,000 iterations for both style coefficients and BT ratings. Each bootstrap sample drew battles with replacement and re-estimated the model. Two-sided bootstrap p-values are $p = 2 \cdot \min(\hat{F}(0), 1 - \hat{F}(0))$, where $\hat{F}(0)$ is the proportion of replicates with the statistic $\leq 0$, with a floor of $1/(B+1)$.

### 3.4 Multiple Comparison Correction

We applied the Benjamini-Hochberg (BH) procedure (Benjamini & Hochberg, 1995) to control the false discovery rate (FDR) at 0.05, separately for the style-coefficient tests and the per-model controlled-minus-raw **rating-change** tests. The displayed rank changes are descriptive transformations of those ratings, not separately tested rank statistics.

---

## 4. Results

### 4.1 Style Coefficients

Table 1 shows the estimated effect of each formatting feature on win probability, after controlling for model identity.

**Table 1. Style coefficients from the Bradley-Terry model (137,037 battles, 116 models). p-values are BH-adjusted across 5 tests.**

| Feature | % Odds Change | 95% CI | p (BH) | Significant? |
|---------|--------------:|--------|--------|-------------|
| **Bold** | **+14.8%** | [+11.8%, +18.0%] | 0.003 | Yes |
| **Headers** | +9.2% | [+3.5%, +12.5%] | 0.003 | Yes |
| **Lists** | +6.8% | [+4.7%, +9.3%] | 0.003 | Yes |
| **Emoji** | +3.5% | [+1.4%, +6.0%] | 0.003 | Yes |
| Code blocks | −0.2% | [−1.6%, +3.5%] | 0.780 | No |

**Interpretation.** A one-standard-deviation difference in bold usage is associated with 14.8% higher win odds, after adjusting for model identity and the other formatting features. Headers, lists, and emoji have smaller but significant associations; code blocks show no detectable pooled association. Bold is the strongest formatting correlate in this specification. These associations are more modest than the +16–19% reported on the older, reaction-augmented Compar:IA export, and we take the votes-only numbers here as the more clearly defined estimate.

### 4.2 Ablation Study

Controlling for one feature at a time inflates each coefficient, because the formatting features are correlated and each alone absorbs the shared variance.

**Table 2. Ablation: single-feature style control (odds change per SD).**

| Feature | Alone | Joint (Table 1) |
|---------|------:|----------------:|
| Bold | +23.0% | +14.8% |
| Headers | +20.5% | +9.2% |
| Lists | +16.0% | +6.8% |
| Emoji | +9.1% | +3.5% |
| Code blocks | +1.4% | −0.2% |

Every coefficient roughly halves when the others are added, the first sign of the collinear "verbosity" bundle that recurs throughout the paper.

### 4.3 Ranking Impact

The correlation between standard and formatting-controlled BT ratings is **r = 0.990**: rankings are stable overall, but specific models shift. Across all 116 models, **84 (72%)** show a statistically significant rating change after style control.

**Table 3. Largest rank changes after formatting control. “Sig?” refers to the BH-adjusted test of the underlying rating change.**

| Model | Std → Ctrl | ΔRank | Sig? |
|-------|:----------:|------:|------|
| qwen-3-8b | 80 → 98 | **−18** | Yes |
| gpt-5.3 | 45 → 28 | **+17** | Yes |
| mistral-small-2603 | 10 → 27 | **−17** | Yes |
| mistral-large-2512 | 3 → 19 | **−16** | Yes |
| qwen3-30b-a3b | 78 → 93 | **−15** | Yes |
| gpt-oss-120b | 37 → 50 | **−13** | Yes |
| qwen3-32b | 54 → 65 | **−11** | Yes |
| o4-mini | 76 → 66 | **+10** | Yes |
| mistral-medium-2508 | 2 → 11 | **−9** | Yes |

Heavy formatters (mistral-large-2512, mistral-small-2603, gpt-oss-120b) fall, while several more concise models rise. These shifts establish sensitivity to the adjustment; they do not by themselves show that the models which rise have greater reasoning depth or substantive quality.

### 4.4 Position Bias and Arena Modes

Position bias is negligible and not significant: model A wins 49.94% of decisive battles (binomial p = 0.67). Win rates are close to 50% within every arena mode (random 49.9%, custom 50.1%, big-vs-small 49.6%, small-models 50.7%). This is cleaner than the older export, which carried a small but significant A-side bias, and reflects the balanced construction of the consolidated release.

### 4.5 Length and Linguistic Features

The formatting model of §4.1–§4.4 excludes length and considers only markdown. Here we add, on the same battles, the answer's **length** (output-token count) and the **linguistic** features: readability (REL, Coleman-Liau, Flesch-Kincaid), diversity (TTR and length-robust MATTR), and sentence structure (mean sentence length, long-sentence ratio). The common support with every feature and ≥100 battles per model is **127,893 battles across 116 models**. Coefficients use the same standardized A−B contrast as §3.2, winsorized at 1/99%; bootstrap and BH follow §3.3–§3.4.

**Length is large, and it is partly what "formatting" was measuring.** Adding length shrinks the formatting coefficients, unsurprising given how correlated length is with markdown density: longer answers carry more markup. The team's original argument for excluding length (it proxies completeness) still stands; the point here is quantitative, that a share of the headline formatting effect travels with length.

**Formatting nonetheless survives the joint model.** With length and all linguistic features held fixed, the markdown effects remain: bold **+10.4%** [+7.9, +12.7], headers +6.6%, code blocks +2.2%, emoji +3.5% (all significant after BH); lists falls to +1.7% and is no longer significant, absorbed by its correlates.

**MATTR is the most stable linguistic correlate in our specifications.** It carries **+17.7%** [+15.5, +19.5] and is nearly uncorrelated with length in these data, so unlike raw TTR it is not simply length in disguise: at equal length, higher moving-average type-token ratio is associated with winning. We are cautious about reading this as "vocabulary richness": high MATTR can also reflect named entities, technical terminology, or lower repetition rather than genuinely richer language, and MATTR's length-independence is itself window- and length-dependent (§5.5). Raw TTR (−27.5%) is the opposite face of the same coin, mechanically decreasing in length, and should not be read as an independent diversity effect. Length itself is +6.5% [+4.4, +8.9]. The honest reading is that length, markdown density, and raw diversity form one collinear "verbosity" bundle whose individual coefficients shift as the specification changes (lists is significant alone but not jointly), while MATTR is the one linguistic measure that stands apart from the bundle.

**Readability adds little.** REL (+0.9%) and Flesch-Kincaid (−2.8%) are not significant; only Coleman-Liau (+4.2%) reaches significance, and the three are mutually collinear, so we do not interpret them individually. Mean sentence length is not significant.

![Figure 1. Left: joint style coefficients with 95% bootstrap CIs, coloured by feature family (grey = not significant after BH). Length dominates alone but is absorbed in the joint model; bold survives; MATTR is the most stable length-independent linguistic correlate. Right: formatting coefficients shrink as length and then the linguistic features are added.](figures/fig9_linguistic.png)

**Adding these features improves fit**: battle-prediction accuracy rises from 0.634 (formatting) to 0.637 (+length) to **0.642** (joint), a modest gain showing that the measured features carry incremental predictive information. This does not establish that presentation itself caused the additional correctly predicted votes.

**The joint control reshuffles the leaderboard more than formatting alone.** On the common support, the standard ranking correlates with the formatting-only controlled ranking at r = 0.973 but with the *joint* controlled ranking at only **0.938** (Spearman 0.920); **37 of 116 models move by ≥10 ranks**. Heavy formatters fall (gpt-oss-120b −44, mistral-small-2603 −37, mistral-large-2512 −31, glm-4.5 −30), while concise strong models rise (gpt-5.3 +40, claude-3-5-sonnet-v2 +34, claude-3-7-sonnet +29, claude-4-sonnet +29).

**Much of the apparent association is between models.** Moving from a reduced-form logit (no per-model fixed effects) to the full Bradley-Terry roughly halves bold (+28.2%→+10.4%), collapses mean sentence length (+25.4%→+2.9%), and cuts MATTR (+33.1%→+17.7%). Thus a large share of the pooled association comes from differences between models that tend to present differently; the fixed-effect coefficient is the remaining within-model conditional association, not a causal effect.

### 4.6 Exploratory Heterogeneity by Completed Conversation Length

We originally intended to ask whether the presentation-vote association differs when a conversation ran longer before the vote. The implemented split instead uses the number of user turns in the **completed conversation**: one turn versus two or more. Of 137,037 battles, 80.9% fall in the completed-single-turn group and 19.1% in completed-multi-turn. As §2.4 shows, this does not preserve temporal order: 40.5% of the nominal multi-turn battles were single-turn when the retained vote occurred. The estimates below remain a reproducible description of heterogeneity by eventual conversation length, but they cannot test an attentiveness or reading-depth mechanism.

We fit a single pooled Bradley-Terry model with per-model fixed effects, the standardized A−B contrasts, and each contrast **interacted with a multi-turn indicator** (contrasts standardized once on the full sample). For a feature with main coefficient γ and interaction δ, the single-turn odds ratio is $e^{\gamma}$ and the multi-turn odds ratio is $e^{\gamma+\delta}$; a negative δ means the feature is associated with the vote less strongly in multi-turn battles. Table 4 reports both, from this one model, so the columns reconcile.

**Formatting associations are substantially smaller in multi-turn battles**, and four of five interactions are significant after BH:

**Table 4. Formatting association by completed conversation length (win-odds change per SD, from the pooled interaction model; single-turn = $e^{\gamma}$, multi-turn = $e^{\gamma+\delta}$).**

| Feature | Single-turn | Multi-turn | Interaction δ | Sig? |
|---------|:-----------:|:----------:|:-------------:|------|
| **Bold** | **+35.0%** | **+8.5%** | −0.219 | Yes |
| Headers | +12.3% | +7.5% | −0.044 | Yes |
| Lists | +10.0% | +6.6% | −0.032 | Yes |
| Code blocks | +5.3% | +1.2% | −0.039 | Yes |
| Emoji | +2.6% | +4.8% | +0.021 | No |

Bold, the strongest cue, falls the most: its association drops from +35% win odds per SD in completed-single-turn battles to +8% in completed-multi-turn ones, about a 76% reduction. Because later turns enter both the stratum definition and, in 11.2% of battles, the response features, this contrast has no clean temporal interpretation.

![Figure 2. Completed conversation length. Left: win-odds change per SD for each feature in completed-single-turn (circle) vs completed-multi-turn (diamond) battles; formatting and length associations both shrink toward zero, and MATTR (orange) is comparatively unchanged. Right: the interaction with 95% bootstrap CIs; grey is not significant after BH. The timing audit in §2.4 prevents interpreting this as pre-vote engagement.](figures/fig10_reading_depth.png)

**Length behaves the same way; lexical diversity does not.** In the joint model, length's association falls from +12.4% to +6.4% across the completed-single-turn and completed-multi-turn strata (interaction −0.055, significant). Length-independent lexical diversity is essentially unchanged: MATTR moves from +16.3% to +15.1% (interaction −0.011, not significant). This is descriptive heterogeneity only.

**Caveats.** Eventual conversation length is endogenous, and the current feature construction has post-vote look-ahead. Section 4.7 shows that the numerical interaction is not explained by topic composition, but that robustness check cannot repair its temporal ordering. The section should be rerun after truncating each conversation and its token totals at the retained vote; until then, it is not evidence about attentiveness, engagement, or causal moderation. Model fixed effects are held common across the two descriptive strata.

### 4.7 Topic Controls: Is the Formatting Premium Just a Proxy for Subject Matter?

A natural objection is that presentation stands in for topic: technical questions invite code blocks and lists. We use the dataset's `categories` taxonomy (about 18 subject classes, present for all battles; we take the first as the primary topic). Because topic is a property of the shared prompt, it differences out of the pairwise model, so it can only enter through **topic × style interactions**, i.e. by letting each subject have its own formatting slope.

**The formatting premium holds within every topic.** Refitting the formatting Bradley-Terry model separately inside each topic with at least 2,500 battles, the bold effect is positive in **every** topic and significant in most:

**Table 5. Bold effect (odds change per SD) within each topic.**

| Topic | Bold |
|-------|-----:|
| Natural Science & Technology | +19.5% |
| Politics & Government | +61.0% |
| Health & Wellness & Medicine | +30.7% |
| Arts | +34.7% |
| Personal Development & Career | +25.2% |
| Food & Drink & Cooking | +21.1% |
| Law & Justice | +21.4% |
| Environment | +21.7% |
| Entertainment & Travel & Hobby | +17.3% |
| Education | +13.5% |
| Business & Economics & Finance | +12.9% |

The premium is present everywhere, not driven by a few formatting-heavy subjects. The magnitude varies (bold looks unusually strong in Politics and Arts), but the smaller strata carry wide intervals, so we do not over-read the topic-by-topic ordering.

![Figure 3. Topic controls. Left: the bold association (win-odds change per SD) estimated within each topic, with 95% bootstrap CIs; positive in every subject. Right: the completed-multi-turn interactions of §4.6 (circle) versus the same model with topic × formatting interactions added (diamond); the two nearly coincide. This does not resolve the timing problem in §2.4.](figures/fig11_topic_controls.png)

**The completed-length interaction survives topic controls.** Re-fitting the §4.6 formatting × multi-turn interaction model with topic × formatting interactions added leaves the interactions essentially unchanged: bold −0.214, headers −0.044, lists −0.031, code blocks −0.035 (all significant), emoji +0.017 (n.s.). Topic mix therefore does not explain the numerical pattern, but this does not convert completed conversation length into a pre-vote measure.

Topic here is subject matter, not task type (summarise, translate, write code, give advice), which §4.8 addresses separately.

### 4.8 Task-Type Controls: Is the Formatting Premium a Proxy for Task Form?

Subject matter (§4.7) is not the sharpest confound. Task *form* drives presentation more directly: a request to write code produces code blocks, a request for a list produces lists, a translation request produces neither. If the formatting premium simply reflected which tasks invite which markup, it should weaken within a task.

We label each battle's opening prompt with a coarse ten-class task taxonomy (translation, code, summarization, math, list/table, writing, ideas, advice, explanation, other) using ordered keyword rules over the mostly-French prompt text. This is a deliberately imperfect proxy: validated against independent LLM labelling of 120 held-out prompts, the rules agree 56% of the time overall and 64% when they assign a specific (non-"other") label, with recall the main weakness (about a third of prompts fall to "other" because they state the task only implicitly). We therefore read this as an exploratory robustness check, not a definitive control, and analyse only the classes with at least 2,500 battles.

**The bold premium holds within tasks.** Refitting the formatting Bradley-Terry model inside each task, bold is positive in eight of the nine task strata, and its 95% CI excludes zero in the five largest (explanation +21.6%, code +24.5%, summarization +29.1%, advice +27.7%, writing +16.1%); it is positive but not significant in the smaller ideas/math strata and negative only in translation (−13.1%, n.s.). So the bold association is not an artefact of task form: it is present whether the user asked for an explanation, code, a draft, or a summary.

**Table 6. Bold and code-block association (odds change per SD) within each task.**

| Task | Battles | Bold | Code blocks |
|------|:------:|:----:|:-----------:|
| explanation | 44,841 | +21.6% | −0.8% |
| writing | 12,579 | +16.1% | −13.1% |
| code | 9,746 | +24.5% | +15.2% |
| ideas | 4,405 | +14.0% | +1.1% |
| list/table | 2,955 | +2.6% | +12.9% |
| summarization | 2,941 | +29.1% | −4.6% |
| advice | 2,616 | +27.7% | −6.0% |
| translation | 2,449 | −13.1% | +25.2% |
| math | 1,472 | +12.5% | +9.9% |

**Code blocks are task-specific, which is why they vanish in the pool.** Unlike bold, code blocks help only where code is plausibly involved: +15% in the code task (and positive in translation, list/table, math), but near-zero or negative in prose tasks (explanation, writing, summarization, advice). This is the expected pattern if code formatting is valued only when a coding answer is wanted, and it explains the null pooled code-block coefficient of §4.1: averaged over mostly-non-coding battles, a task-specific effect washes out. It also serves as a face-validity check on the noisy classifier, the one feature that should be task-specific is.

A validated task taxonomy (an LLM classifier over all prompts, or embedding-based clustering) would let us control for task directly rather than stratify with a rough proxy; we leave that to future work.

### 4.9 Robustness to Arena Mode and Temporal Dependence

The data mix random pairings (72% of decisive battles) with user-selected "custom" pairs (19%) and specialist modes. Custom pairing is non-random: users choose models for reasons that may correlate with expected quality and style, a potential selection confound. As a check, we refit the headline coefficients on **random-pair battles only** (98,240 in the formatting model, 91,769 in the joint model, 116 models) and compare to the full sample.

The two robust signals are stable: bold is +16.4% in the formatting-only model on random battles (vs +14.8% full) and +11.4% in the joint model (vs +10.4%), and MATTR is +16.5% (vs +17.7%). The collinear-bundle features we already flag as unstable do move, headers falls to +3.6% (from +9.2%) and length to +4.0% (from +6.5%), exactly the specification-sensitivity we describe elsewhere. So restricting to random pairings leaves the conclusions we actually draw (bold and MATTR) intact and only reshuffles the coefficients we already decline to over-read. A fully random-only primary specification would be cleaner and is a natural next version; on this evidence, arena-mode selection is not what drives the headline associations.

We also probe the bootstrap's independence assumption. The release carries no user identifier (§5.5), so we cannot cluster by user, but we can **block-bootstrap by calendar week** (resampling the 89 weekly blocks with replacement, keeping each week's battles together) to absorb temporal dependence from a shifting model roster and user population. The block-bootstrap 95% intervals are only slightly wider than the i.i.d. ones and still exclude zero comfortably: bold +14.8% [+11.4, +19.0] (i.i.d. [+11.8, +18.0]), joint bold +10.4% [+7.7, +12.6], MATTR +17.7% [+15.2, +20.0] (i.i.d. [+15.5, +19.5]). Temporal clustering therefore widens the intervals only modestly and does not threaten the headline signals; user-level clustering remains untestable here.

### 4.10 MATTR Stress Tests

Because MATTR is one of our two headline signals, we probe it harder, on two concerns: whether its length-independence is real in these data, and whether it holds for short answers.

Responses here are long: the median is 736 output tokens (IQR 406–1,257), and only 4.5% of responses fall below the 50-word-token MATTR window (those are dropped, so MATTR is computed on the 95.5% that are long enough). Across the full length range MATTR is essentially uncorrelated with length (Spearman +0.04 overall, and within length quartiles |ρ| ≤ 0.08), so its independence from length is not an artefact of a particular length regime.

Splitting battles by response length, the MATTR association is concentrated in longer answers, +17.3% per SD among above-median-length answers versus +4.9% among shorter ones, while length itself does the opposite (+4.7% long vs +45.6% short). This is coherent: when answers are short, getting longer wins big and vocabulary variety is barely measurable; when answers are already long, extra length adds little and lexical diversity carries the signal. MATTR stays positive in both strata.

Finally, is the signal really lexical variety, or a proxy for named entities and technical terms? We recomputed three alternative diversity measures on the response text: MTLD (a window-free metric), content-word MATTR (French function words removed), and MATTR with capitalised tokens excluded (a rough proper-noun exclusion). MATTR tracks MTLD very closely (Spearman 0.95), so it is not an artefact of the 50-token window. More to the point, the joint diversity coefficient survives every substitution: **+17.7% with MATTR, +13.6% with MTLD, +12.4% with content-word MATTR, and +13.9% with proper-noun-excluded MATTR**. Stripping function words and proper nouns lowers the estimate but leaves it clearly positive, so the association is not merely named entities or padding. We still call this a stable diversity *correlate* rather than validated "richness" (topical specificity is not fully excluded, and we did not test HD-D), but it is robust to how diversity is measured.

### 4.11 Correlation with External Leaderboards

An adjusted ranking is useful only if it clarifies what the leaderboard measures; rank movement alone cannot show that an adjustment is an improvement. We therefore compare the three Compar:IA rankings fitted on the common support of §4.5—raw, formatting-controlled, and full joint-controlled—with an external LMArena Text Arena snapshot published 16 July 2026 (Arena Team, 2026). LMArena publishes both raw and style-controlled rankings and an overall and French category. We pin dataset revision `afed939e10281b660a4369206ca505b2bf5e0208` and match **exact model identifiers only**, yielding 49 shared models for the overall rankings and 40 for French. Spearman correlation is appropriate because the two platforms' rating scales are arbitrary and only rank agreement is of interest.

**Table 8. Spearman rank correlation between Compar:IA variants and external LMArena rankings. Bold marks the strongest Compar:IA variant in each row.**

| External ranking | Exact matches | Compar:IA raw | Formatting-controlled | Full joint-controlled |
|---|---:|---:|---:|---:|
| LMArena raw, overall | 49 | 0.791 | **0.800** | 0.731 |
| LMArena style-controlled, overall | 49 | 0.767 | **0.806** | 0.752 |
| LMArena raw, French | 40 | 0.778 | **0.798** | 0.721 |
| LMArena style-controlled, French | 40 | 0.701 | **0.770** | 0.715 |

**Narrow formatting control has the strongest external agreement in all four comparisons.** Relative to raw Compar:IA, its gain ranges from +0.009 against the raw overall leaderboard to +0.069 against the style-controlled French leaderboard. The latter is the most conceptually aligned comparison: both sides are French-category human-preference rankings with markdown-style adjustment. The pattern is consistent with formatting control removing some platform-specific presentation preference while retaining most of the shared preference signal.

The evidence is suggestive rather than decisive. Paired model-level bootstrap intervals for the gain all include zero: raw overall [−0.034, +0.059], style-controlled overall [−0.010, +0.098], raw French [−0.051, +0.105], and style-controlled French [−0.012, +0.177]. Models and model variants are also not independent draws, so even these intervals are optimistic as formal inference. The result should therefore be stated as a repeated directional pattern, not proof that formatting control improves validity.

**More control is not monotonically better.** The full joint-controlled ranking correlates less strongly than formatting-only control in every comparison, and less strongly than raw Compar:IA in three of four. A plausible explanation is that length and linguistic properties partly carry genuine quality, language, or task-fit information shared across platforms; removing them can discard signal along with presentation preference. This result reinforces the paper's endogeneity argument: external agreement supports the narrower formatting sensitivity analysis more than the claim that the fully adjusted ranking is a purer capability measure. LMArena is itself another observational human-preference arena, not an objective ground truth, so this is cross-platform convergent validity rather than validation against correctness.

---

## 5. Discussion

### 5.1 Presentation Is Associated with Votes, Mostly as One Dimension

Presentation is systematically associated with French arena votes: a one-SD difference in bold usage is associated with about 15% higher win odds in the formatting-only model, headers and lists less, in line with English-language style control. But once length and linguistic features enter, most of that association collapses into a **single collinear "verbosity" dimension**: length, bold, and lists trade coefficient weight, each formatting coefficient roughly halves from the single-feature to the joint model, and lists is significant alone but not jointly. The honest summary is not "bold is worth +10%" but "longer, more heavily marked-up answers tend to win, and we cannot cleanly divide the credit."

Two signals stand apart from that bundle: **bold usage** and **length-independent lexical diversity** (MATTR). MATTR is the more interesting, being nearly uncorrelated with length, so at equal length higher lexical diversity is associated with winning, a correlate formatting-only style control misses. We stop short of calling it vocabulary "richness" (§5.5). Readability adds little once length is present.

The completed-conversation split (§4.6) produces a numerical contrast—formatting and length associations are smaller in eventual multi-turn conversations, while lexical diversity is comparatively unchanged—but the timing audit rules out the intended reading-depth interpretation. Later turns enter the stratum definition and sometimes the measured response itself. We therefore do not treat this result as evidence that engaged readers discount presentation.

The external comparison (§4.11) adds a second qualification. Formatting-only control produces the strongest correlation with each matched LMArena ranking, including LMArena's style-controlled French category, but the bootstrap uncertainty includes no improvement. Full joint control performs worse in three of four comparisons than leaving Compar:IA raw. The evidence therefore favours presenting narrow formatting control as a useful sensitivity analysis, not treating every measured text property as nuisance variation to remove.

### 5.2 Implications for Arena Design

1. **Ranking interpretation.** Heavy formatters (mistral-large-2512, gpt-oss-120b) fall under adjustment, but that does not establish that their raw ranks are inflated: presentation may be either a bias or part of answer quality. Arena leaderboards should consider publishing the raw ranking and pre-specified sensitivity analyses side by side.
2. **Interface experiments.** Formatting-normalised rendering or randomised re-rendering of identical content would test whether markup itself changes votes without assuming that a regression-adjusted ranking is ground truth.
3. **Model development incentives.** If arena rankings drive development priorities, a presentation premium may create incentives to optimize visible style. The observational results identify that risk but do not quantify how much substantive quality would change under a style intervention.

### 5.3 Endogeneity and Between-Model Composition

Our style features are covariates, not experimental manipulations, and two causal readings fit the data.

**Confounder (formatting as bias).** Users are partially swayed by presentation; formatting inflates win probability independent of content. Style control then removes a bias.

**Mediator (formatting as quality signal).** Better models produce better-structured output because they are more capable; formatting mediates capability and preference. Style control then removes legitimate signal.

Three descriptive tests probe these alternatives but do not identify either causal pathway.

**Test 1: Rating–formatting correlation.** Model formatting intensity (bold+lists+headers per response) correlates with standard BT rating at Pearson r = 0.60 (Spearman 0.66): models preferred by users also format more. After style control the correlation falls to r = 0.49. This is compatible with presentation carrying both shared preference signal and platform-specific advantage, but it is not evidence of mediation because the BT rating is itself estimated from the same votes.

**Test 2: Tier-stratified style effects.** Splitting models into tiers by standard rating and fitting within-tier:

**Table 7. Formatting effect by model-pair tier (odds change per SD).**

| Feature | Bottom | Middle | Top |
|---------|:------:|:------:|:---:|
| Bold | +18.6% | +8.9% | +10.3% |
| Lists | +14.0% | +7.0% | −6.0% |
| Headers | +8.7% | +2.3% | +15.1% |
| N battles | 32,414 | 14,214 | 15,698 |

The bold association is largest in the bottom tier (+18.6% vs +10.3% in the top tier). This heterogeneity is compatible with a stronger presentation premium among lower-rated models, but tiering on an outcome-derived rating and changing the set of model pairs prevents a causal interpretation.

**Test 3: Rating change vs formatting intensity.** Across all models, the rating change after style control correlates at r = −0.97 with formatting intensity (Pearson r = −0.967): the models that format most lose the most. This is largely a mechanical diagnostic that the adjustment operates in the intended direction; it does not settle whether the removed signal was bias or quality.

**Synthesis.** The data are compatible with formatting being both a quality signal and a source of presentation preference, but these analyses cannot estimate the share attributable to either. Per-model fixed effects separate within-model from between-model association; they do not turn the remaining coefficient into a causal effect. Because presentation and substantive quality are jointly determined, neither the standard nor the presentation-adjusted ranking is a definitive measure of quality. Only a controlled study that varies presentation while holding content fixed (§5.5) could separate the pathways.

### 5.4 Qualitative Analysis of Winner-Flipping Battles

A "winner-flipping" battle is one where the standard and style-controlled models disagree on which of its two models is stronger. Of 137,037 battles, **6,038 (4.4%)** flip. Within flips, the vote winner uses more total formatting (bold+lists+headers) in **51.6%** of cases versus **41.1%** for the loser, a modest asymmetry between vote outcome and formatting in closely contested battles. The models appearing most in flips are a mix of heavy formatters whose ratings deflate under control (mistral-large-2512, mistral-medium-2508) and frequent opponents (llama-3.1-405b, claude-4-6-sonnet).

This release carries no per-message reaction data, so the user-reported "clear formatting" attribute and the hand-picked response excerpts of an earlier analysis are not available here.

### 5.5 Limitations

**Conditional associations, not causal effects.** Our features are observed, not manipulated, so every coefficient is a conditional association, not the effect of changing a feature while holding substance fixed. Adjusting for model identity does not close this gap: within a model, formatting, length, and diversity co-vary with prompt difficulty, task type, correctness, refusals, and conversation history, any of which can make a feature a proxy for unobserved answer quality (a model may use bold precisely when it has produced a thorough answer). The between-model decomposition (§5.3) probes this but cannot resolve it; only a controlled study that renders identical content under randomly assigned presentation could. We use "associated with" deliberately throughout.

**Collinear presentation features.** Length, bold, and lists move together, and the joint model cannot cleanly identify their individual contributions. Claims should be read at the level of "verbosity" and of the two features that stand apart (bold, MATTR), not of every individual coefficient.

**MATTR interpretation.** §4.10 stress-tests MATTR: responses are long (median 736 tokens; 4.5% below the window), MATTR is uncorrelated with length within every length bin, its association holds in short and long strata, and it survives substitution by MTLD, content-word MATTR, and proper-noun-excluded MATTR. So the signal is robust to how diversity is measured and is not merely named entities. What we still cannot fully rule out is that high diversity proxies topical specificity, and we did not test HD-D. We therefore describe MATTR as our most stable diversity *correlate*, not as validated "richness."

**English-calibrated readability on French text.** Coleman-Liau and Flesch-Kincaid are English-calibrated; only REL is French-specific. We therefore do not interpret the readability coefficients individually.

**Task type controlled only with a rough proxy.** §4.7 controls for subject matter and §4.8 for task type, and the bold premium survives both while code blocks turn out task-specific. But the task classifier is a coarse keyword rule (about 56% agreement with LLM labels, recall-limited), so §4.8 is a robustness check rather than a clean control. A validated task taxonomy (an LLM classifier over all prompts) is the main remaining validity step.

**Post-vote look-ahead and completed conversation length.** The source repeats completed `full_conversation_*` fields on turn-level rows. The current pipeline therefore includes later turns in 11.2% of analysed battles, and 40.5% of the reported multi-turn stratum was single-turn when the retained vote occurred. This is more serious than ordinary endogeneity: temporal ordering is wrong for those cases. It directly invalidates an attentiveness interpretation of §4.6 and may affect the other feature coefficients and rankings. All primary analyses should be rerun on response prefixes and cumulative token totals truncated at the retained vote.

**Fluency not tested here.** The companion analysis's CamemBERT pseudo-perplexity, a fluency proxy, requires a GPU and is not recomputed on this dataset. On the earlier export it added nothing once length and readability were controlled, but we cannot confirm that null holds here; recomputing it on comparia-fr-arena is a natural next step.

**No independent replication.** This is a single corpus of votes from one platform. §4.11 provides a cross-platform ranking comparison, not a replication of the battle-level coefficient analysis. Generalisation of the feature associations still requires response-level data from another platform or population.

**External leaderboard comparison.** The LMArena analysis matches exact public model identifiers, but provider settings, system prompts, sampling parameters, and evaluation windows may differ even when names match. Its 40–49 matched models are a selected subset of the 116-model Compar:IA ranking, related model variants are dependent, and LMArena is another human-preference arena rather than a correctness benchmark. The correlations therefore measure convergent rank agreement on the overlap, not objective capability validity.

**Clustering and arena mode.** §4.9 shows the headline associations are stable on random pairs only, so mode selection is not the driver, though a fully random-only *primary* specification would be cleaner still. On clustering, our bootstrap resamples battles independently. Each battle is already a distinct conversation (we keep one vote per `comparison_id`), so within-conversation dependence is not double-counted. The remaining concern is that one user contributes many battles, but the comparia-fr-arena release carries no user, session, or visitor identifier (only conversation- and response-level ids, unlike the older export which had a session hash), so we cannot cluster by user or estimate how much it would widen the intervals. A user-level cluster bootstrap would require identifiers this release does not expose; as a partial substitute, the weekly block bootstrap in §4.9 widens the headline intervals only modestly, so temporal dependence is not inflating the significance.

**Platform-specific population.** Compar:IA is open without account creation, and the release contains no user demographics. Its self-selected user population may differ from other arenas or from the populations to which one might want to generalise.

---

## 6. Conclusion

In Compar:IA's French-language preference data, measured output characteristics are associated with pairwise votes beyond model identity, but not in the tidy, feature-by-feature way a formatting-only analysis suggests. Most presentation variables share substantial variance with response length, so their individual coefficients are specification-dependent. Two correlates stand apart in the joint model: bold usage and length-independent lexical diversity (MATTR), the latter associated with winning even at equal length. Readability adds little, and the pooled-versus-fixed-effect comparison shows that much apparent linguistic association is between models rather than within models. That decomposition does not identify whether presentation is bias, a quality signal, or both.

A completed-conversation split shows smaller formatting and length associations in eventual multi-turn conversations, but a timing audit finds post-vote turns in the feature text for 11.2% of battles and shows that 40.5% of the nominal multi-turn stratum was single-turn at the retained vote. That result cannot support a reading-depth mechanism, and the primary estimates should be treated as provisional until the pipeline is rerun on vote-truncated prefixes.

Across exact shared model identifiers, narrow formatting control has the strongest correlation with all four LMArena comparators, but the improvement over raw Compar:IA is not statistically decisive. Full joint control usually lowers external agreement. Because presentation and substantive quality are jointly determined, neither the raw nor an adjusted ranking is a pure measure of capability. The defensible option is to publish the raw ranking alongside clearly labelled, pre-specified sensitivity analyses. Two extensions would most sharpen the evidence: a validated task-type classifier to replace the rough proxy of §4.8, and a controlled study that varies presentation while holding content fixed, which is what would turn these associations into effects.

---

## References

*(The reading-comprehension references support general distinctions in text processing; they do not validate this paper's specific mapping from formatting, length, and lexical measures to levels of reading, which remains an interpretive hypothesis.)*

**Arenas, LLM evaluation, and style control**

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345. https://doi.org/10.2307/2334029
- Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., & Stoica, I. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. *Proceedings of ICML 2024*. https://arxiv.org/abs/2403.04132
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*. https://arxiv.org/abs/2306.05685
- Li, T., Angelopoulos, A. N., & Chiang, W.-L. (2024). Does style matter? Disentangling style and substance in Chatbot Arena. *LMSYS Org blog*. https://www.lmsys.org/blog/2024-08-28-style-control/
- Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-controlled AlpacaEval: A simple way to debias automatic evaluators. *arXiv:2404.04475*. https://arxiv.org/abs/2404.04475
- Wu, M., & Aji, A. F. (2025). Style over substance: Evaluation biases for large language models. *Proceedings of COLING 2025*, 297–312. https://aclanthology.org/2025.coling-main.21/
- Singhal, P., Goyal, T., Xu, J., & Durrett, G. (2024). A long way to go: Investigating length correlations in RLHF. *Proceedings of COLM 2024*. https://openreview.net/forum?id=G8LaO1P0xv
- Saito, K., Wachi, A., Wataoka, K., & Akimoto, Y. (2023). Verbosity bias in preference labeling by large language models. *arXiv:2310.10076*. https://arxiv.org/abs/2310.10076
- Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2023). Large language models are not fair evaluators. *arXiv:2305.17926*. https://arxiv.org/abs/2305.17926
- Arena Team. (2026). *Arena Leaderboard Dataset*. https://arena.ai/blog/arena-leaderboard-dataset/ Dataset revision `afed939e10281b660a4369206ca505b2bf5e0208`, leaderboard date 16 July 2026.
- compar:IA. (2026a). *Classement compar:IA : méthodologie et limites*. https://comparia.beta.gouv.fr/ranking
- compar:IA & Ministère de la Culture. (2026b). *comparia-fr-arena: A French human-preference arena dataset*. https://huggingface.co/datasets/ministere-culture/comparia-fr-arena Revision `8cd6488c5d0c3b8dfcb9339d11ae9624c84359be`, accessed 8 July 2026. Open Licence 2.0 (Etalab) and CC-BY-4.0.
- Ministère de la Culture. (2025). *ComparIA : améliorer les IA conversationnelles sur les usages francophones et former les citoyens*. https://www.culture.gouv.fr/thematiques/innovation-numerique/comparia-ameliorer-les-ia-conversationnelles-sur-les-usages-francophones-et-former-les-citoyens

**Methods and statistics**

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552

**Reading and text comprehension**

- Kintsch, W., & van Dijk, T. A. (1978). Toward a model of text comprehension and production. *Psychological Review*, 85(5), 363–394. https://doi.org/10.1037/0033-295X.85.5.363
- van Dijk, T. A., & Kintsch, W. (1983). *Strategies of discourse comprehension*. Academic Press.
- Just, M. A., & Carpenter, P. A. (1980). A theory of reading: From eye fixations to comprehension. *Psychological Review*, 87(4), 329–354. https://doi.org/10.1037/0033-295X.87.4.329
- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin*, 124(3), 372–422. https://doi.org/10.1037/0033-2909.124.3.372

**Readability and lexical diversity**

- Kandel, L., & Moles, A. (1958). Application de l'indice de Flesch à la langue française. *Cahiers d'Études de Radio-Télévision*, 19, 253–274.
- Coleman, M., & Liau, T. L. (1975). A computer readability formula designed for machine scoring. *Journal of Applied Psychology*, 60(2), 283–284.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of new readability formulas for Navy enlisted personnel*. Research Branch Report 8-75, Naval Air Station Memphis.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type–token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100. https://doi.org/10.1080/09296171003643098

---

## Appendix A: Analysis Pipeline

The core results are reproducible from the battle table `data/fr_battles.parquet`, built from `comparia-fr-arena` by `src/build_fr_arena.py`. `run.py` executes the core pipeline; scripts that require the raw release, auxiliary data, or an external leaderboard snapshot are run separately as indicated in the repository README (scripts live in `src/`, outputs in `results/`).

| Script | Section | Output |
|--------|---------|--------|
| `analyze_core.py` | §4.1–4.4 | formatting BT, rank changes, position bias |
| `linguistic_analysis.py` | §4.5 | joint formatting+length+linguistic model |
| `leaderboard_shift.py` | §4.5 | standard vs formatting vs joint ranking shift |
| `turn_depth_analysis.py` | §4.6 | formatting × completed-conversation-length interactions |
| `topic_analysis.py` | §4.7 | within-topic fits + topic × style controls |
| `extract_prompts.py` + `task_classify.py` + `task_analysis.py` | §4.8 | task proxy and within-task fits |
| `robustness_random.py`; `time_block_bootstrap.py` | §4.9 | random-only coefficients; weekly block bootstrap |
| `mattr_stress.py` + `mattr_alt.py` | §4.10 | MATTR length-independence, strata, and MTLD/content-word/no-proper-noun variants |
| `external_leaderboard_analysis.py` | §4.11 | exact-match correlations with pinned LMArena raw/style-controlled rankings |
| `audit_vote_timing.py` | §2.4, §4.6 | post-vote turns and completed-vs-visible turn classification |
| `endogeneity_analysis.py` | §5.3 | between-model composition and tier heterogeneity |
| `qualitative_analysis.py` | §5.4 | winner-flip prevalence and asymmetry |
