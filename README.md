# Hawaii Business Insights
### Does Engaging with Customer Reviews Actually Help a Business's Rating?

**by Nancy Le** | DSC 80: Practice and Application of Data Science, UC San Diego

---

## Introduction

Online reviews have become one of the most powerful signals in consumer decision-making — and businesses know it. Many actively respond to customer reviews, hoping that visible engagement will improve their reputation. But does it actually work? Is there a measurable association between how often a business responds to reviews and the ratings it receives?

This project investigates that question using Google Maps review data from Hawaii. The dataset contains **21,507 businesses** and over **1.5 million reviews**, offering a rich window into how businesses manage their online presence across the islands. For each business, I engineered engagement metrics from the raw review data — including how often they respond, how quickly, and how much text-based interaction their reviewers provide — and combined these with business metadata such as category, price level, and listing completeness.

**Central Question: Is business engagement with customer reviews associated with higher average business ratings?**

This matters for any business owner, marketer, or platform designer wondering whether active reputation management pays off — or whether ratings are simply a product of service quality alone.

The relevant columns used throughout this analysis are:

| Column | Type | Description |
|---|---|---|
| `avg_rating` | Quantitative | Average star rating (1–5) from Google Maps metadata |
| `response_rate` | Quantitative | Proportion of reviews the business responded to |
| `total_reviews` | Quantitative | Number of reviews in the dataset for this business |
| `rating_std` | Quantitative | Standard deviation of review ratings |
| `text_rate` | Quantitative | Proportion of reviews containing written text |
| `review_span_days` | Quantitative | Days between earliest and latest review |
| `price_level` | Ordinal | Price tier ($=1, $$=2, $$$=3, $$$$=4) |
| `primary_category` | Nominal | Primary business category (e.g. Restaurant, Hotel) |
| `has_hours` | Nominal | Whether business hours are listed (0 or 1) |
| `has_description` | Nominal | Whether a business description exists (0 or 1) |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The raw data came in two files: business metadata (`meta-Hawaii.json`) and review-level data (`review-Hawaii_10.json`, a 10-core subset containing only businesses with at least 10 reviews).

**Review-level cleaning:** The `resp` column — indicating whether a business responded to a review — was ~93% null, reflecting that most reviews never receive a response. For the minority that did, `resp` was stored as a stringified dictionary with `time` and `text` keys, which I parsed with `ast.literal_eval`. I dropped rows with null `rating` values rather than imputing, since ratings feed directly into `mean_review_rating` and imputation would corrupt that aggregate. Negative `response_delay_days` values (where the response timestamp preceded the review timestamp) were treated as data artifacts and set to `NaN`.

**Business-level aggregation:** I collapsed the review data to one row per business, computing engagement metrics including `response_rate`, `avg_response_delay`, `text_rate`, `pics_rate`, and a `review_span_days` measure capturing how long a business had been actively reviewed.

**Metadata cleaning:** I encoded `price` as an ordinal integer (`$=1`, `$$=2`, `$$$=3`, `$$$$=4`), extracted the first entry from each business's category list as `primary_category`, flagged permanently closed businesses, and created binary proxy features `has_hours` and `has_description` for listing completeness.

**Merging:** I joined the metadata and aggregated review data on `gmap_id`. Because the reviews file is a 10-core subset, 9,789 out of 21,507 businesses (45%) had no matching reviews and therefore have `NaN` for all engagement features. These rows are retained in the EDA but excluded during modeling.

Here are the first few rows of the cleaned DataFrame:

| name | primary_category | price_level | avg_rating | has_hours | total_reviews | response_rate | text_rate | review_span_days |
|---|---|---|---|---|---|---|---|---|
| Hale Pops | Restaurant | NaN | 4.4 | 1 | NaN | NaN | NaN | NaN |
| SMP - Single Marine Program | Recreation center | NaN | 4.1 | 1 | 20.0 | 0.0 | 0.1 | 378.0 |
| 2 Cheesy Guys | Food court | NaN | 5.0 | 1 | NaN | NaN | NaN | NaN |
| Kraken Coffee Kahului | Coffee shop | 1.0 | 4.8 | 1 | NaN | NaN | NaN | NaN |
| Akasatana Ramen Kyoto | Ramen restaurant | NaN | 5.0 | 1 | NaN | NaN | NaN | NaN |

### Univariate Analysis

<iframe
  src="assets/response_rate_dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Response rate is heavily right-skewed — the majority of businesses respond to few or no reviews, with a smaller cluster that responds to nearly all of them. This bimodal tendency suggests two distinct engagement strategies among Hawaii businesses: those that almost never engage, and those that actively manage nearly every review.

<iframe
  src="assets/avg_rating_dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The distribution of average ratings is left-skewed, with the bulk of businesses rated between 4.0 and 5.0. This "positivity bias" is a well-known pattern on review platforms: customers are more likely to leave reviews for places they liked. The spike at 5.0 reflects businesses with very few reviews where a single 5-star rating becomes the entire average.

### Bivariate Analysis

<iframe
  src="assets/response_rate_vs_rating.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The OLS trendline suggests a slight positive association between response rate and average rating, though variance is high. Many non-responding businesses still achieve high ratings, indicating that engagement alone does not determine quality.

<iframe
  src="assets/rating_by_response_bin.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Binning response rate into decile intervals reveals that mean rating generally increases with response rate, particularly at higher bins. Businesses that respond to over 80% of their reviews consistently show higher average ratings than those in the lower bins.

### Interesting Aggregates

The table below shows mean `avg_rating` and `response_rate` grouped by price level, restricted to businesses with non-null engagement features:

| Price Level | Mean Avg Rating | Mean Response Rate | Business Count |
|---|---|---|---|
| $ (1) | 4.24 | 0.18 | 312 |
| $$ (2) | 4.31 | 0.21 | 418 |
| $$$ (3) | 4.38 | 0.27 | 189 |
| $$$$ (4) | 4.44 | 0.31 | 47 |

Both average rating and response rate tend to increase with price tier. Higher-priced businesses may have more professional management infrastructure, leading them to both respond more and earn higher ratings — though this relationship may also reflect selection effects (customers who patronize expensive venues may give more generous reviews).

---

## Assessment of Missingness

### NMAR Analysis

One column that may be **NMAR (Not Missing At Random)** is `response_rate`. Businesses that never engage with customer reviews likely have missing `response_rate` precisely *because* they do not engage — meaning the missingness depends on the true underlying value of the variable itself. A business with zero engagement would have a `response_rate` of 0, but may also be absent from the 10-core reviews dataset entirely, making the feature uncomputable. The fact that non-engagement causes the feature to be unobservable is the hallmark of NMAR.

To convert this to MAR, one would need additional data on business owner activity — such as Google Business Profile login frequency, whether an owner has claimed their listing, or platform-side records of account engagement — that could explain the missingness without referencing the unobserved values themselves.

### Missingness Dependency

I analyzed the missingness of `avg_response_delay`, which is missing for ~84.8% of businesses — specifically those that never responded to any review and therefore have no delay to measure.

**Does `avg_response_delay` missingness depend on `total_reviews`?**

<iframe
  src="assets/missingness_permutation.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Permutation test result: observed mean difference in `total_reviews` (missing − present) = **−28.09**, p-value = **0.0000**.

We reject the null hypothesis. Businesses where `avg_response_delay` is missing have significantly fewer total reviews than businesses where it is present. This makes intuitive sense: businesses with more reviews are more active on Google Maps and more likely to be managed by an owner who monitors and responds to feedback. The missingness of `avg_response_delay` is therefore **MAR** with respect to `total_reviews`.

**Does `avg_response_delay` missingness depend on `latitude`?**

Observed latitude difference = very small, p-value = **0.9410** (0.9390 in an alternate run).

We fail to reject the null hypothesis. Geographic location within Hawaii does not predict whether a business responds to reviews, which makes sense — responsiveness is a management behavior, not a geographic one.

---

## Hypothesis Testing

### Do Businesses That Respond to Reviews Have Higher Ratings?

**Null Hypothesis:** The average rating of businesses that respond to at least one review is the same as those that don't. Any observed difference is due to random chance.

**Alternative Hypothesis:** Businesses that respond to at least one review have a *higher* average rating than those that don't.

**Test Statistic:** Difference in mean `avg_rating` (responders − non-responders). A mean difference is appropriate here because `avg_rating` is a continuous numeric variable and we have a directional hypothesis.

**Significance Level:** 0.05

The test was restricted to businesses with at least one review in the dataset (`total_reviews` is not null), ensuring the non-responder group only contains businesses that actively chose not to respond rather than businesses absent due to the 10-core filter.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

**Results:** Observed difference = **0.0697 stars** (responders: 4.3831, non-responders: 4.3134). P-value = **0.0000**.

**Conclusion:** At a significance level of 0.05, we reject the null hypothesis. The difference in mean ratings between responding and non-responding businesses is unlikely to be due to random chance. However, the effect size is small (0.07 stars on a 5-point scale), so while statistically significant, the result may not be practically meaningful. Importantly, we cannot conclude causation: it is plausible that higher-rated businesses are simply more motivated to engage with reviews, rather than engagement driving ratings upward.

---

## Framing a Prediction Problem

**Prediction Problem:** Predict the average rating (`avg_rating`) of a Hawaii business based on its review engagement metrics and business characteristics.

**Type:** Regression. `avg_rating` is a continuous variable on a 1–5 scale.

**Response Variable:** `avg_rating` — the most direct measure of customer-perceived quality, and the variable our hypothesis test showed to be associated with engagement behavior.

**Features available at time of prediction:** All features used are observable properties of the business prior to any new review — `response_rate`, `total_reviews`, `text_rate`, `pics_rate`, `review_span_days`, `price_level`, `primary_category`, `has_description`, and `has_hours`. Note that `avg_response_delay` was excluded: it is missing for ~85% of businesses (those that never responded), which would substantially shrink the modeling dataset. `response_rate` captures the same engagement signal with far less missingness.

**Evaluation Metric:** Root Mean Squared Error (RMSE). RMSE reports error in the same units as the rating scale (stars), making it directly interpretable. It also penalizes larger errors more heavily than smaller ones — desirable here because a 2-star prediction error is much more problematic than a 0.1-star one. RMSE is preferred over MAE for this reason.

---

## Baseline Model

The baseline model uses **Linear Regression** with four features:

- `response_rate` — quantitative; our key engagement variable
- `total_reviews` — quantitative; proxy for business popularity
- `price_level` — ordinal (treated as quantitative)
- `primary_category` — nominal; one-hot encoded to capture structural rating differences across business types

All steps — median imputation for quantitative features, most-frequent imputation + one-hot encoding for the categorical feature, and model training — are implemented in a single `sklearn` Pipeline.

**Results:**

| Split | RMSE |
|---|---|
| Train | 0.3250 |
| Test | 0.3510 |

The train and test RMSE are close, indicating the model is not overfitting. An RMSE of ~0.35 stars means our predictions are off by about a third of a star on average — reasonable for a simple 4-feature linear model, but with meaningful room to improve. The linear model is likely missing non-linear relationships and additional signal from engagement features not yet included.

---

## Final Model

### Feature Engineering

On top of the baseline features, I added four engineered features:

1. **`log_total_reviews`** — `total_reviews` is heavily right-skewed. Log-transforming it compresses the scale, makes the relationship with `avg_rating` more linear, reduces the outsized influence of extremely popular businesses, and captures the diminishing informational returns of very high review counts.

2. **`rating_std`** — measures consistency of customer reviews. A business with a 4.0 average from uniformly 4-star reviews is fundamentally different from one averaging 4.0 from a mix of 1s and 5s. This captures review polarization that `avg_rating` alone cannot represent.

3. **`text_rate`** — proportion of reviews with written text. Businesses that attract detailed written feedback may signal higher engagement quality beyond just response rate, capturing how thoroughly customers describe their experience.

4. **`review_span_days`** — measures business longevity and accumulated customer exposure. Longer-running businesses have had more time to stabilize their ratings and represent more reliable averages.

### Modeling Algorithm and Hyperparameter Search

I used a **Random Forest Regressor** instead of Linear Regression because it can capture non-linear relationships and feature interactions without requiring manual specification of interaction terms. A `GridSearchCV` search was performed over:

- `n_estimators`: [100, 200] — number of trees in the ensemble
- `max_depth`: [5, 10, None] — tree depth controlling model complexity
- `min_samples_split`: [2, 5] — minimum observations required to split a node

**Best hyperparameters:** `max_depth=10`, `min_samples_split=2`, `n_estimators=200`

### Results

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Baseline (Linear Regression) | 0.3250 | 0.3510 |
| Final (Random Forest) | 0.2088 | **0.2649** |

The final model reduced test RMSE by **0.086 stars** — a **24.5% improvement** over the baseline. The gap between train and test RMSE (0.2088 vs. 0.2649) reflects some overfitting, but the `max_depth=10` constraint already mitigates this relative to unconstrained trees. The engineered features `log_total_reviews` and `rating_std` likely contributed most to the improvement by enabling the model to capture non-linear review volume effects and review consistency, neither of which were accessible to the baseline linear model.

---

## Fairness Analysis

**Question:** Does the model perform worse for low-volume businesses than high-volume businesses?

**Groups:**
- **Group X (low-volume):** businesses with `total_reviews` below the median
- **Group Y (high-volume):** businesses with `total_reviews` at or above the median

This is a meaningful fairness concern because high-volume businesses provide more data for computing engagement features like `response_rate` and `text_rate`, potentially yielding more reliable estimates and easier-to-predict ratings.

**Evaluation Metric:** RMSE

**Null Hypothesis:** The model is fair with respect to business review volume. RMSE for low-volume and high-volume businesses is the same; any observed difference is due to random chance.

**Alternative Hypothesis:** The model is less accurate for low-volume businesses — RMSE for low-volume businesses is higher than for high-volume businesses.

**Significance Level:** 0.05

<iframe
  src="assets/fairness_permutation.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

**Results:**
- Low-volume RMSE: **0.3285**
- High-volume RMSE: **0.1808**
- Observed difference (Low − High): **0.1477**
- P-value: **0.0000**

**Conclusion:** At a significance level of 0.05, we reject the null hypothesis. The model performs significantly worse for low-volume businesses, and the observed RMSE gap of 0.1477 stars falls far outside the null distribution. This is a meaningful limitation: engagement features like `response_rate` and `text_rate` derived from fewer reviews are noisier estimates of true business behavior, making predictions less reliable. The model works best for established businesses with substantial review histories, and may underserve the many smaller businesses that constitute a large portion of the Hawaii dataset.
