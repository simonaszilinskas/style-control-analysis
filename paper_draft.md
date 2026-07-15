# What Wins a Vote? Formatting, Length, and Lexical Diversity in the French Compar:IA LLM Arena

---

## Abstract

LLM evaluation arenas, where users compare two model outputs side-by-side, have become a primary source of model rankings, and a standing worry is that these rankings reward *presentation* over substance. We study this on Compar:IA, a French government-backed arena, using its consolidated `comparia-fr-arena` release (about 138,000 decisive French votes across 116 models). We decompose preference into model skill and presentation with a style-controlled Bradley-Terry model, and go beyond markdown formatting to a full presentation account: **formatting** (bold, lists, headers, code, emoji), **length**, and **linguistic** properties of the text (readability, lexical diversity, sentence structure). Four findings emerge. First, presentation genuinely moves votes, but most of it is a single collinear "verbosity" dimension: length, bold, and lists rise and fall together, and their individual attributions are unstable, one absorbs another as the model changes. Second, two signals stand out from that bundle and survive the full joint control: **bold formatting** (about +10% win odds per standard deviation) and **length-independent lexical diversity** (MATTR, about +18%), so at equal length, more varied vocabulary wins. Readability metrics are collinear and mostly weak, and raw type-token ratio is just length in disguise. Third, much of the *apparent* linguistic effect is model skill: coefficients roughly halve or more when per-model strengths are included, the difference between a confounder and a mediator. Fourth, presentation acts through the depth of reading: proxying reading depth by conversation length, the pull of formatting *and* of length fades sharply once a conversation runs several turns (bold falls from +38% to +6% win odds per SD), while length-independent lexical diversity is unchanged, so of the surface cues, only vocabulary richness survives an attentive read. Controlling for the full presentation bundle reshuffles the leaderboard (rating correlation 0.94; 37 of 116 models move by ≥10 ranks; heavy formatters such as gpt-oss-120b and mistral-large-2512 fall ~30 places while concise strong models rise). To our knowledge this is the first style-control analysis of a non-English arena, and the formatting effect matches English-language findings. The practical message for arena operators is that "quality" rankings partly measure verbosity, and that reporting both raw and presentation-controlled leaderboards is the honest option.

---

## 1. Introduction

The rise of LLM evaluation arenas, platforms where users interact with two anonymous models and select a preferred response, has established a new paradigm for model comparison. The LMSYS Chatbot Arena pioneered this approach, and its Elo-based rankings are widely cited as measures of model quality. The methodology has since been adopted by multiple platforms, including Compar:IA, a French government-backed arena launched in October 2024.

A key concern with arena-based evaluation is the extent to which user preferences reflect genuine content quality versus superficial presentation. Zheng et al. (2023) first noted that LLM judges exhibit a preference for longer, more verbosely formatted outputs, and subsequent work quantified a systematic length and verbosity bias in both human and automatic preference judgments (Singhal et al., 2023; Saito et al., 2023; Dubois et al., 2024). The LMSYS team subsequently introduced "style control" (Li et al., 2024), a methodology that decomposes win probability into model skill and formatting effects using a modified Bradley-Terry model (Bradley & Terry, 1952). Their analysis of English-language data found that controlling for response length, markdown formatting, and list usage modestly reshuffled rankings.

Most style-control work stops at markdown and length. But "presentation" is broader: how readable the prose is, how varied its vocabulary, how long its sentences, all things a user can respond to without them tracking correctness. These linguistic features are correlated with formatting and length, so studying any one in isolation risks mis-attributing a shared effect. We therefore analyze all three families together, **formatting** (bold, lists, headers, code, emoji), **length**, and **linguistic** properties (readability, lexical diversity, sentence structure), in a single style-controlled model, and ask which presentation features carry an *independent* effect once the others, and model identity, are held fixed.

It helps to organise these features by the level of reading they act on. Text-comprehension research distinguishes a surface reading of the words (Rayner, 1998) from deeper processing of the text's structure and meaning (Kintsch & van Dijk, 1978; van Dijk & Kintsch, 1983; Just & Carpenter, 1980). Adapting that idea to an arena vote, an evaluation can engage at three levels: (a) a glance at the *shape* of the answer, whether it looks complete and well organised; (b) a reading of its *argument*, whether the content is relevant and thorough; and (c) the *words themselves*, at the surface and semantic level. Our feature families map roughly onto these levels: **formatting** is what a level-(a) glance registers, **length** is the crudest proxy for the level-(b) size of the answer, and **lexical diversity and readability** are level-(c) properties of the language. This framing, which we owe to the project's supervision, predicts that the features should not matter equally to every vote: a reader who only glances should be swayed most by surface cues, and a reader who engages should weight substance more. We test this in §4.6, where reading depth is proxied by how many turns a conversation ran before the vote, and find a partial confirmation with an instructive twist.

We apply this analysis to the Compar:IA data, which provides a strong setting:

1. **Scale:** about 138,000 cleaned decisive battles across 116 models, from organic user interactions.
2. **Language:** French-language evaluation, enabling the first cross-linguistic test of whether presentation preferences are culturally invariant.
3. **Native metadata:** each battle carries an LLM-assigned topic taxonomy and the full multi-turn conversation, which let us control for subject matter and measure reading depth directly.
4. **Reasoning models:** the roster includes reasoning models whose chain-of-thought outputs produce systematically different presentation profiles.

Our central question: **which presentation features independently change which models rank highest, and which apparent effects are really model skill?** Our contributions are: (i) a joint presentation-controlled ranking that separates formatting, length, and linguistic effects from each other and from model skill; (ii) the finding that presentation is largely one collinear verbosity dimension, with only bold formatting and length-independent lexical diversity standing apart as stable signals; (iii) a confounder-versus-mediator analysis showing much apparent linguistic effect is model skill; and (iv) a reading-depth test showing that the pull of formatting and length fades once readers engage over several turns, while lexical diversity does not, which gives the feature families a reading-level interpretation.

---

## 2. Data

### 2.1 The Compar:IA Platform

Compar:IA is an LLM evaluation arena operated by the French government's Direction interministérielle du numérique (DINUM). Users submit prompts and receive responses from two anonymous models side-by-side, may continue the conversation over several turns, and then vote for a winner or declare a tie. Models are identified only after voting.

The platform offers several arena modes; in our data the decisive votes come from **random** (72%, random model pairs), **custom** (19%, user-selected pairs), **big-vs-small** (8%, deliberately pairing large and small models), and **small-models** (2%).

### 2.2 Dataset

We use **`ministere-culture/comparia-fr-arena`** (Ministère de la Culture, 2024), the consolidated Compar:IA release published on HuggingFace. It is organised by turn: one row per conversation turn, with the vote (`choice`) recorded on the single turn at which the user decided. From its 641,277 turns we keep the rows carrying a decisive French vote, take the full conversation the voter had seen at that point, and treat each as one battle.

| | Raw turns | Decisive French battles | Models (≥100 battles) |
|---|---:|---:|---:|
| comparia-fr-arena | 641,277 | 137,217 | 116 |

Ties and no-vote turns are dropped (decisive votes only, matching prior style-control work). Winner distribution is balanced (model A wins 49.9%, see §4.4).

### 2.3 Feature extraction and data quality

For each battle we concatenate the assistant messages of the conversation the voter saw and compute all features on that text. Formatting features use the markdown regexes below; length is the response's cumulative output-token count from the dataset metadata (not a whitespace word count, which is unreliable for French); the linguistic features are the set designed by Maayeesha Farzana (PSL) in a companion internship analysis, recomputed here on the comparia-fr-arena text using her exact formulas. Battles whose visible response text is empty (for example when only a model's hidden reasoning was recorded, or content contained `<think>` leakage) are dropped, since their style features would be spuriously zero. Topic and turn count come directly from the dataset (§2.4), so no cross-dataset joins are needed. Every feature is present for essentially all battles (topic 100%, length 99.9%, linguistic 97.5%).

### 2.4 Native topic and reading-depth signals

Two things this release provides directly are central to the analysis. First, each conversation carries an LLM-assigned **topic taxonomy** (`categories`, about 18 subject classes), which we use to control for subject matter (§4.7). Second, the full multi-turn conversation is retained, so **reading depth** is measurable as the number of user turns before the vote (§4.6). Both are properties of the vote as recorded, not reconstructions.

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

Our primary formatting analysis (§4.1–§4.4) uses the five markdown features only, so it is directly comparable to prior English-language style control. We hold length and the linguistic family back to §4.5 for a specific reason: **length confounds with completeness**, a genuine quality dimension. Users may prefer more thorough answers, so controlling for length risks removing legitimate quality signal rather than bias. §4.5 confronts that trade-off head-on by adding length and all linguistic features to the same model.

### 3.2 Bradley-Terry Model

We rank models using a Bradley-Terry model estimated via logistic regression, following the methodology of the LMSYS Chatbot Arena.

**Standard model.** For each decisive battle (A wins or B wins), we construct a feature vector with +1 for the winning model's index and -1 for the losing model's index, then fit a logistic regression without regularization. Ratings are computed as:

$$\text{Rating}_i = 1000 + \frac{400 \cdot \beta_i}{\ln(10)}$$

**Style-controlled model.** We augment the model indicator features with style difference features. For each feature $f$, we compute $\Delta f = f_A - f_B$ (standardized to zero mean and unit variance), then fit:

$$P(\text{A wins}) = \sigma\left(\sum_i \beta_i \cdot \mathbb{1}_i + \sum_f \gamma_f \cdot \Delta f\right)$$

where $\beta_i$ are model skill parameters and $\gamma_f$ are style coefficients. For the joint model (§4.5) the $\Delta f$ contrasts are additionally winsorized at the 1/99th percentile before standardizing, because several linguistic features are heavy-tailed and a handful of extreme battles would otherwise drive the coefficient.

### 3.3 Bootstrap Inference

We computed 95% confidence intervals via nonparametric bootstrap (Efron, 1979), with 1,000 iterations for both style coefficients and BT ratings. Each bootstrap sample drew battles with replacement and re-estimated the model. Two-sided bootstrap p-values are $p = 2 \cdot \min(\hat{F}(0), 1 - \hat{F}(0))$, where $\hat{F}(0)$ is the proportion of replicates with the statistic $\leq 0$, with a floor of $1/(B+1)$.

### 3.4 Multiple Comparison Correction

We applied the Benjamini-Hochberg (BH) procedure (Benjamini & Hochberg, 1995) to control the false discovery rate (FDR) at 0.05, separately for the style-coefficient tests and the per-model rank-change tests.

---

## 4. Results

### 4.1 Style Coefficients

Table 1 shows the estimated effect of each formatting feature on win probability, after controlling for model identity.

**Table 1. Style coefficients from the Bradley-Terry model (137,037 battles, 116 models). p-values are BH-adjusted across 5 tests.**

| Feature | % Odds Change | 95% CI | p (BH) | Significant? |
|---------|--------------:|--------|--------|-------------|
| **Bold** | **+14.8%** | [+11.8%, +18.0%] | 0.002 | Yes |
| **Headers** | +9.2% | [+3.5%, +12.5%] | 0.002 | Yes |
| **Lists** | +6.8% | [+4.7%, +9.3%] | 0.002 | Yes |
| **Emoji** | +3.5% | [+1.4%, +6.0%] | 0.002 | Yes |
| Code blocks | −0.2% | [−1.6%, +3.5%] | 0.504 | No |

**Interpretation.** A one-standard-deviation increase in bold formatting raises a response's win odds by 14.8%, independent of which model produced it. Headers, lists, and emoji have smaller but significant effects; code blocks have no detectable effect. Bold is the single strongest formatting cue, consistent with it being the most visually salient. These effects are more modest than the +16–19% reported on the older, reaction-augmented Compar:IA export, and we take the cleaner votes-only numbers here as the more reliable estimate.

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

**Table 3. Largest rank changes after formatting control (BH-adjusted).**

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

Heavy formatters (mistral-large-2512, mistral-small-2603, gpt-oss-120b) fall; leaner models rise, including reasoning models such as o4-mini, consistent with reasoning models trading visible formatting for reasoning depth.

### 4.4 Position Bias and Arena Modes

Position bias is negligible and not significant: model A wins 49.94% of decisive battles (binomial p = 0.67). Win rates are close to 50% within every arena mode (random 49.9%, custom 50.1%, big-vs-small 49.6%, small-models 50.7%). This is cleaner than the older export, which carried a small but significant A-side bias, and reflects the balanced construction of the consolidated release.

### 4.5 Length and Linguistic Features

The formatting model of §4.1–§4.4 excludes length and considers only markdown. Here we add, on the same battles, the answer's **length** (output-token count) and the **linguistic** features: readability (REL, Coleman-Liau, Flesch-Kincaid), diversity (TTR and length-robust MATTR), and sentence structure (mean sentence length, long-sentence ratio). The common support with every feature and ≥100 battles per model is **127,893 battles across 116 models**. Coefficients use the same standardized A−B contrast as §3.2, winsorized at 1/99%; bootstrap and BH follow §3.3–§3.4.

**Length is large, and it is partly what "formatting" was measuring.** Adding length shrinks the formatting coefficients, unsurprising given how correlated length is with markdown density: longer answers carry more markup. The team's original argument for excluding length (it proxies completeness) still stands; the point here is quantitative, that a share of the headline formatting effect travels with length.

**Formatting nonetheless survives the joint model.** With length and all linguistic features held fixed, the markdown effects remain: bold **+10.4%** [+7.9, +12.7], headers +6.6%, code blocks +2.2%, emoji +3.5% (all significant after BH); lists falls to +1.7% and is no longer significant, absorbed by its correlates.

**The one clean, robust linguistic signal is length-robust lexical diversity.** MATTR carries **+17.7%** [+15.5, +19.5] and is essentially uncorrelated with length, so it is not a verbosity proxy: at equal length, more varied vocabulary wins. Raw TTR (−27.5%) is the opposite face of the same coin, mechanically a decreasing function of length, and should not be read as an independent diversity effect. Length itself is +6.5% [+4.4, +8.9]. The honest reading is that length, markdown density, and raw diversity form one collinear "verbosity" bundle whose individual attributions shift as the specification changes (lists is significant alone but not jointly), while MATTR stands apart from the bundle.

**Readability adds little.** REL (+0.9%) and Flesch-Kincaid (−2.8%) are not significant; only Coleman-Liau (+4.2%) reaches significance, and the three are mutually collinear, so we do not interpret them individually. Mean sentence length is not significant.

![Figure 1. Left: joint style coefficients with 95% bootstrap CIs, coloured by feature family (grey = not significant after BH). Length dominates alone but is absorbed in the joint model; bold survives; MATTR is the one clean length-independent linguistic effect. Right: formatting coefficients shrink as length and then the linguistic features are added.](figures/fig9_linguistic.png)

**Adding these features improves fit**: battle-prediction accuracy rises from 0.634 (formatting) to 0.637 (+length) to **0.642** (joint), a modest gain consistent with style being a real but secondary driver of the vote.

**The joint control reshuffles the leaderboard more than formatting alone.** On the common support, the standard ranking correlates with the formatting-only controlled ranking at r = 0.973 but with the *joint* controlled ranking at only **0.938** (Spearman 0.920); **37 of 116 models move by ≥10 ranks**. Heavy formatters fall (gpt-oss-120b −44, mistral-small-2603 −37, mistral-large-2512 −31, glm-4.5 −30), while concise strong models rise (gpt-5.3 +40, claude-3-5-sonnet-v2 +34, claude-3-7-sonnet +29, claude-4-sonnet +29).

**Much of the apparent effect is model skill.** Moving from a reduced-form logit (no per-model strengths) to the full Bradley-Terry roughly halves bold (+28.2%→+10.4%), collapses mean sentence length (+25.4%→+2.9%), and cuts MATTR (+33.1%→+17.7%). A large part of what looks like a presentation effect is really the strength of the models that happen to present that way.

### 4.6 Reading Depth: Does Presentation Matter Less When the Answer Is Read More Carefully?

The three-level framing of §1 predicts that surface cues should fade once a reader engages more deeply. We proxy reading depth with **conversation length**: a single-turn battle (the user asked once, then voted) invites a quick surface judgement, whereas a multi-turn battle (≥2 user turns before the vote) signals commitment and a more attentive read. Of 137,037 battles, 80.9% are single-turn and 19.1% multi-turn.

We fit a single pooled Bradley-Terry model with per-model strengths, the standardized A−B contrasts, and each contrast **interacted with a multi-turn indicator**. The interaction is the object of interest: negative means the feature moves a vote less when the conversation ran long. Contrasts are standardized once on the full sample.

**Formatting fades sharply with reading depth.** Every markdown effect is smaller in multi-turn battles, and four of five interactions are significant after BH:

**Table 4. Formatting effect by reading depth (odds change per SD; interaction is the multi-turn slope shift).**

| Feature | Single-turn | Multi-turn | Interaction | Sig? |
|---------|:-----------:|:----------:|:-----------:|------|
| **Bold** | **+37.8%** | **+6.8%** | −0.219 | Yes |
| Headers | +14.4% | +4.3% | −0.044 | Yes |
| Lists | +11.6% | +3.7% | −0.032 | Yes |
| Code blocks | +7.8% | +0.5% | −0.039 | Yes |
| Emoji | +3.9% | +2.3% | +0.021 | No |

Bold's advantage falls from +38% win odds per SD in single-turn battles to +7% in multi-turn ones, roughly an 82% reduction. The visual-shape advantage is largest exactly when the reader is least engaged.

![Figure 2. Reading depth. Left: win-odds change per SD for each feature in single-turn (circle) vs multi-turn (diamond) battles; formatting effects and length both shrink toward zero with reading depth, and only length-independent diversity MATTR (orange) is unchanged. Right: the multi-turn interaction (slope shift) with 95% bootstrap CIs; grey is not significant after BH.](figures/fig10_reading_depth.png)

**Length also fades, and only lexical diversity persists.** Adding length and the linguistic features (joint model), length's effect *falls* with reading depth as well: +14.1% single-turn to +6.0% multi-turn, interaction −0.055 (significant). This is where the data qualify the three-level prediction. We had expected length, as a proxy for the depth of the answer, to matter *more* to an engaged reader; instead it behaves like the other surface cues and fades. The one exception is **length-independent lexical diversity: MATTR is unchanged across reading depth** (+15.8% single-turn, +16.1% multi-turn; interaction −0.011, not significant). So the honest revision of the framing is: quick readers are swayed by size and formatting; attentive readers discount those size heuristics, and the only presentation signal that survives an attentive read is the richness of the vocabulary itself.

**Caveats.** Reading depth is proxied, not measured: multi-turn conversations may also differ in task type, difficulty, or user disposition, and which conversations become multi-turn is not random. §4.7 shows the formatting result is not merely a topic-composition effect. We hold model strengths common across depth.

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

![Figure 3. Topic controls. Left: the bold effect (win-odds change per SD) estimated within each topic, with 95% bootstrap CIs; positive in every subject. Right: the multi-turn interactions of §4.6 (circle) versus the same model with topic × formatting interactions added (diamond); the two nearly coincide, so the reading-depth effect is not a topic-composition artefact.](figures/fig11_topic_controls.png)

**The reading-depth result survives topic controls.** Re-fitting the §4.6 formatting × multi-turn interaction model with topic × formatting interactions added, the multi-turn interactions are essentially unchanged: bold −0.214, headers −0.044, lists −0.031, code blocks −0.035 (all significant), emoji +0.017 (n.s.). The formatting-fades-with-engagement pattern is not a topic-mix artefact.

Topic here is subject matter, not task type (summarise, translate, write code, give advice); a task-type control would be a useful further step but is not cleanly available in the metadata, so we leave it to future work.

---

## 5. Discussion

### 5.1 Presentation Bias Is Real, but It Is Mostly One Dimension

Three things hold together. First, presentation genuinely influences French arena votes: bold raises win odds by about 15% per standard deviation in the formatting-only model, headers and lists less, and the effect matches English-language style control, so this is not a culture-specific quirk. Second, once length and linguistic features enter, most of that effect is a **single collinear "verbosity" dimension**: length, bold, and lists are correlated, they trade coefficient weight, and their individual attributions shift with the specification (each formatting coefficient roughly halves from the single-feature to the joint model, and lists is significant alone but not jointly). The honest summary is not "bold is worth exactly +10%" but "answers that are longer and more heavily marked up win, and we cannot cleanly divide the credit."

Third, and more usefully, two signals stand apart from that bundle: **bold formatting** and **length-independent lexical diversity** (MATTR). MATTR is the more interesting. It is essentially uncorrelated with length, so it is not verbosity in disguise: holding length fixed, answers that use a more varied vocabulary win. This is a genuinely new result relative to formatting-only style control, and it points at a preference for richer language rather than merely more of it. Readability adds little once length is present, so "fluency" as generic readability metrics measure it does not explain votes.

The reading-depth test (§4.6) gives the bundle a mechanism. The features do not act on the vote in the same way. Formatting behaves like a *glance* cue: its effect is large when the reader engages least and collapses (for bold, by ~82%) once the conversation runs several turns. Length behaves the same way, contrary to our initial expectation that it would signal argument depth: it too fades with engagement, which suggests users read length as a size heuristic rather than as evidence of substance. Length-independent lexical diversity is the exception, unchanged whether the answer is skimmed or read closely. So the sharper statement is: what a pooled model sees as one verbosity dimension is, under the reading-depth lens, a bundle of surface size-and-shape heuristics that attentive readers discount, plus one genuine language property (vocabulary richness) that they do not. It also sharpens the practical worry: the formatting premium is concentrated in precisely the low-engagement votes where it is least likely to reflect a considered judgement of quality.

### 5.2 Implications for Arena Design

1. **Ranking interpretation.** Heavy formatters (mistral-large-2512, gpt-oss-120b) may be overranked due to presentation rather than content. Arena leaderboards should consider publishing both standard and style-controlled rankings.
2. **Reasoning model evaluation.** The current format may undervalue reasoning models, which sacrifice formatting for reasoning depth (o3-mini and o4-mini rise under style control). Formatting-agnostic rendering could address this.
3. **Model development incentives.** If arena rankings drive development priorities, presentation bias creates perverse incentives to optimize for visually appealing output over substance.

### 5.3 Endogeneity: Confounder or Mediator?

Our style features are covariates, not experimental manipulations, and two causal readings fit the data.

**Confounder (formatting as bias).** Users are partially swayed by presentation; formatting inflates win probability independent of content. Style control then removes a bias.

**Mediator (formatting as quality signal).** Better models produce better-structured output because they are more capable; formatting mediates capability and preference. Style control then removes legitimate signal.

Three tests shed light without fully resolving it.

**Test 1: Quality–formatting correlation.** Model formatting intensity (bold+lists+headers per response) correlates with standard BT rating at Pearson r = 0.60 (Spearman 0.66): higher-rated models format more, consistent with mediation. After style control the correlation falls to r = 0.49, so the association is partly genuine and partly artefactual.

**Test 2: Tier-stratified style effects.** Splitting models into tiers by standard rating and fitting within-tier:

**Table 6. Formatting effect by model-pair tier (odds change per SD).**

| Feature | Bottom | Middle | Top |
|---------|:------:|:------:|:---:|
| Bold | +18.6% | +8.9% | +10.3% |
| Lists | +14.0% | +7.0% | −6.0% |
| Headers | +8.7% | +2.3% | +15.1% |
| N battles | 32,414 | 14,214 | 15,698 |

The bold effect is largest among weaker models (+18.6% bottom vs +10.3% top), more consistent with the confounder reading: presentation cues weigh more when content quality differences are smaller.

**Test 3: Rating change vs formatting intensity.** Across all models, the rating change after style control correlates at r = −0.97 with formatting intensity (Pearson r = −0.967): the models that format most lose the most. This confirms style control operates as intended but does not settle whether the removed signal was bias or quality.

**Synthesis.** Formatting is *both* a partial mediator and a partial confounder. Better models do format more (mediator), but formatting also exerts an independent pull, larger among weaker models (confounder). Neither standard nor style-controlled rankings is a definitive measure of quality, so operators should report both.

### 5.4 Qualitative Analysis of Winner-Flipping Battles

A "winner-flipping" battle is one where the standard and style-controlled models disagree on which of its two models is stronger. Of 137,037 battles, **6,038 (4.4%)** flip. Within flips, the vote winner uses more total formatting (bold+lists+headers) in **51.6%** of cases versus **41.1%** for the loser, a modest but consistent asymmetry: formatting provides a marginal edge in closely contested battles. The models appearing most in flips are a mix of heavy formatters whose ratings deflate under control (mistral-large-2512, mistral-medium-2508) and frequent opponents (llama-3.1-405b, claude-4-6-sonnet).

This release carries no per-message reaction data, so the user-reported "clear formatting" attribute and the hand-picked response excerpts of an earlier analysis are not available here.

### 5.5 Limitations

**Collinear presentation features.** Length, bold, and lists move together, and the joint model cannot cleanly identify their individual contributions. Claims should be read at the level of "verbosity" and of the two features that stand apart (bold, MATTR), not of every individual coefficient.

**English-calibrated readability on French text.** Coleman-Liau and Flesch-Kincaid are English-calibrated; only REL is French-specific. We therefore do not interpret the readability coefficients individually.

**Topic controlled, task type not.** §4.7 controls for subject matter, and the formatting premium survives. What remains uncontrolled is task type (summarise vs translate vs write code), which the metadata does not cleanly encode; a coding task mechanically invites code blocks. Building a task-type signal and repeating §4.7 is the main remaining validity step.

**Reading depth is a proxy.** §4.6 uses turn count as a stand-in for attentiveness; it also correlates with task type and difficulty, and which conversations become multi-turn is not random.

**Fluency not tested here.** The companion analysis's CamemBERT pseudo-perplexity, a fluency proxy, requires a GPU and is not recomputed on this dataset. On the earlier export it added nothing once length and readability were controlled, but we cannot confirm that null holds here; recomputing it on comparia-fr-arena is a natural next step.

**No independent replication.** This is a single corpus of votes from one platform. The results are internally robust (to topic, to reading depth, to per-model control), but generalisation would need arena data from another platform or population.

**Observational.** Style features are observed, not manipulated; the confounder-vs-mediator question (§5.3) cannot be fully resolved without a controlled study that varies presentation directly.

**Platform-specific population.** Compar:IA users are predominantly French civil servants and technology-interested citizens; results may not generalise to other populations.

---

## 6. Conclusion

Presentation shapes preference votes in the French Compar:IA arena, but not in the tidy, feature-by-feature way a formatting-only analysis suggests. In a formatting-only model, bold raises win odds by about 15% per standard deviation, headers and lists less, 84 of 116 models shift significantly after correction, and heavy formatters drop sharply, a clean transfer of English-language style control to a non-English arena for the first time. But when we add length and a family of linguistic features, most of that effect resolves into one collinear "verbosity" dimension whose individual coefficients are not stable across specifications. What stands apart is narrower and more interpretable: bold formatting (~+10% per SD) and length-independent lexical diversity (MATTR, ~+18%), the latter a signal that richer vocabulary wins even at equal length. Readability adds little, and a confounder-versus-mediator analysis shows much of the apparent linguistic effect is model skill: coefficients roughly halve once per-model strengths are included.

Reading depth ties the threads together. Splitting votes by how many turns the conversation ran, the pull of formatting and of length is concentrated in quick single-turn votes and fades once readers engage, while lexical diversity does not move. What reads as one verbosity dimension is better understood as a set of surface size-and-shape heuristics that attentive readers discount, plus one genuine language property they keep. The formatting premium is weakest exactly where the vote reflects the most considered reading.

For arena operators the message is concrete: "quality" leaderboards partly rank verbosity, controlling for the full presentation bundle rearranges the leaderboard (37 of 116 models move ≥10 ranks) more than controlling for markdown alone, and the only defensible option is to publish both raw and presentation-controlled rankings. Two extensions would sharpen the analysis: a task-type control to complement the topic control of §4.7, and a controlled study that manipulates presentation directly rather than observing it.

---

## References

*(Preliminary. The reading-comprehension references anchor the three-level framing of §1 and §5; they are our reading of Christophe Benavent's proposal and should be confirmed or replaced by the authors.)*

**Arenas, LLM evaluation, and style control**

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.
- Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., & Stoica, I. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. *ICML 2024*.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*.
- Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). Does style matter? Disentangling style and substance in Chatbot Arena. *LMSYS Org blog* (style-control methodology).
- Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-controlled AlpacaEval: A simple way to debias automatic evaluators. *arXiv:2404.04475*.
- Wu, M., & Aji, A. F. (2023). Style over substance: Evaluation biases for large language models. *arXiv:2307.03025*.
- Singhal, P., Goyal, T., Xu, J., & Durrett, G. (2023). A long way to go: Investigating length correlations in RLHF. *arXiv:2310.03716*.
- Saito, K., Wachi, A., Wataoka, K., & Akimoto, Y. (2023). Verbosity bias in preference labeling by large language models. *arXiv:2310.10076*.
- Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2023). Large language models are not fair evaluators. *arXiv:2305.17926*.
- Ministère de la Culture (2024). *Compar:IA: French LLM evaluation arena datasets* (`comparia-fr-arena`). HuggingFace. https://huggingface.co/ministere-culture

**Methods and statistics**

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1–26.

**Reading and text comprehension**

- Kintsch, W., & van Dijk, T. A. (1978). Toward a model of text comprehension and production. *Psychological Review*, 85(5), 363–394.
- van Dijk, T. A., & Kintsch, W. (1983). *Strategies of discourse comprehension*. Academic Press.
- Just, M. A., & Carpenter, P. A. (1980). A theory of reading: From eye fixations to comprehension. *Psychological Review*, 87(4), 329–354.
- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin*, 124(3), 372–422.

**Readability and lexical diversity**

- Kandel, L., & Moles, A. (1958). Application de l'indice de Flesch à la langue française. *Cahiers d'Études de Radio-Télévision*, 19, 253–274.
- Coleman, M., & Liau, T. L. (1975). A computer readability formula designed for machine scoring. *Journal of Applied Psychology*, 60(2), 283–284.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of new readability formulas for Navy enlisted personnel*. Research Branch Report 8-75, Naval Air Station Memphis.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type–token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100.

---

## Appendix A: Analysis Pipeline

All results are reproducible from the single battle table `data/fr_battles.parquet`, built from `comparia-fr-arena` by `src/build_fr_arena.py`; `run.py` executes the steps below in order (scripts live in `src/`, outputs in `results/`).

| Script | Section | Output |
|--------|---------|--------|
| `analyze_core.py` | §4.1–4.4 | formatting BT, rank changes, position bias |
| `linguistic_analysis.py` | §4.5 | joint formatting+length+linguistic model |
| `leaderboard_shift.py` | §4.5 | standard vs formatting vs joint ranking shift |
| `turn_depth_analysis.py` | §4.6 | formatting × reading-depth interactions |
| `topic_analysis.py` | §4.7 | within-topic fits + topic × style controls |
| `endogeneity_analysis.py` | §5.3 | confounder-vs-mediator tests |
| `qualitative_analysis.py` | §5.4 | winner-flip prevalence and asymmetry |
