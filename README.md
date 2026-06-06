# Hawaii Business Insights
### Does Engaging with Customer Reviews Actually Help a Business's Rating?

Author: Nancy Le

---

## Introduction

When you search for a restaurant, hotel, or local shop on Google Maps, the star rating is often the first thing you notice. For businesses, that number can make or break a customer's decision to walk through the door. Knowing this, many business owners actively respond to customer reviews, thanking happy customers, addressing complaints, and signaling to potential visitors that the business is attentive and cares about its reputation. But does that effort actually pay off in the form of higher ratings?

This project investigates exactly that question using Google Maps review data from Hawaii. The dataset combines two sources: a business metadata file with 21,507 Hawaii businesses, and a review file containing over 1.5 million individual customer reviews. Together, they capture a broad picture of how Hawaiian businesses, from beachside shave ice stands to luxury resorts, present themselves online and interact with their customers.

For each business, I engineered a set of engagement metrics derived from the raw review data: how often the business responds to reviews (`response_rate`), how quickly it typically does so (`avg_response_delay`), how much written feedback it attracts (`text_rate`), and how long it has been accumulating reviews (`review_span_days`). These were combined with structural business characteristics like price tier, primary category, and listing completeness to form the basis for both statistical analysis and predictive modeling.

**Central Question: Is business engagement with customer reviews associated with higher average business ratings?**

For business owners, it informs how to allocate time spent on reputation management. For platform designers at Google or Yelp, it raises questions about whether surfacing response behavior to consumers is meaningful. And for data scientists, it illustrates the challenge of disentangling correlation from causation in observational data: a business that responds a lot may already be highly rated for other reasons entirely.

The relevant columns used throughout this analysis are:

| Column | Type | Description |
|---|---|---|
| `avg_rating` | Quantitative | Average star rating (1–5) from Google Maps metadata |
| `response_rate` | Quantitative | Proportion of reviews the business responded to (0–1) |
| `total_reviews` | Quantitative | Number of reviews in the dataset for this business |
| `rating_std` | Quantitative | Standard deviation of individual review ratings |
| `text_rate` | Quantitative | Proportion of reviews that include written text |
| `review_span_days` | Quantitative | Number of days between the earliest and latest review |
| `price_level` | Ordinal | Price tier encoded as integer: $=1, $$=2, $$$=3, $$$$=4 |
| `primary_category` | Nominal | Primary business category extracted from the category list (e.g. Restaurant, Hotel, Coffee shop) |
| `has_hours` | Nominal | Binary flag: 1 if business hours are listed, 0 otherwise |
| `has_description` | Nominal | Binary flag: 1 if a business description is present, 0 otherwise |

---
## Key Findings

> 🔍 **Businesses that respond to reviews have ratings ~0.07 stars higher on average** — statistically significant, but a modest effect.
>
> 📊 **A Random Forest model reduced prediction error by 24.5%** over a linear baseline (test RMSE: 0.2649 vs. 0.3510).
>
> ⚠️ **The model is significantly less accurate for low-review businesses** — RMSE of 0.33 vs. 0.18 for high-volume businesses.
---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The raw data came in two files: business metadata (`meta-Hawaii.json`) and review-level data (`review-Hawaii_10.json`). The reviews file is a **10-core subset**, meaning it only includes reviews for businesses that have received at least 10 reviews total. This is an important constraint that affects the dataset throughout the analysis.

**Review-level cleaning:**

The `resp` column, which indicates whether a business responded to a review, was approximately 93% null across the 1,504,347 reviews. This is not a data quality problem; it simply reflects that the overwhelming majority of reviews never receive a response from the business. For the ~108,000 reviews that did have a response, the `resp` field was stored as a stringified Python dictionary containing `time` (the response timestamp in milliseconds) and `text` (the response content). I parsed these using `ast.literal_eval` to extract the numeric timestamp.

From the response timestamp and review timestamp, I computed `response_delay_days`: how many days elapsed between when the review was posted and when the business responded. Some computed delays were negative, meaning the recorded response timestamp was earlier than the review timestamp. These are data artifacts (likely a logging inconsistency in the platform) and were set to `NaN` rather than kept as implausible negative delays.

I dropped rows where `rating` was null rather than imputing, since individual ratings feed directly into the business-level `mean_review_rating` aggregate. Imputing a missing rating would introduce a fabricated signal into that summary statistic.

I also created binary indicator columns at the review level: `has_text` (1 if the review included written text), `has_pics` (1 if photos were attached), and `has_response` (1 if the business responded).

**Business-level aggregation:**

I collapsed the cleaned review data from one row per review to one row per business using `groupby('gmap_id')`. For each business, I computed:
- `total_reviews`: count of reviews in the dataset
- `mean_review_rating`: average of individual review ratings
- `rating_std`: standard deviation of review ratings (a measure of consistency)
- `response_rate`: proportion of reviews the business responded to
- `avg_response_delay`: mean days to respond (only for reviews that received a response)
- `text_rate`: proportion of reviews with written text
- `pics_rate`: proportion of reviews with photos
- `review_span_days`: number of days between the earliest and latest review

**Metadata cleaning:**

The business metadata required several transformations. The `price` column contained string values like `"$"` or `"$$$$"` and was mapped to integers 1–4 for ordinal modeling. The `category` field stored lists of category strings; I extracted the first element as `primary_category` to give each business a single representative category. I created binary columns `has_hours` (1 if the `hours` field was non-null) and `has_description` (1 if `description` was non-null) as proxies for listing quality and completeness. Permanently closed businesses were flagged with an `is_closed` column.

**Merging and final dataset:**

The aggregated review data and cleaned metadata were joined on `gmap_id`. Because the reviews file is a 10-core subset, **9,789 out of 21,507 businesses (45%) had no matching reviews** and therefore received `NaN` for all engagement features (`response_rate`, `text_rate`, etc.). These businesses retain their metadata features and are included in some parts of the EDA, but are excluded from any analysis or modeling that requires engagement features.

Here are the first few rows of the cleaned, merged DataFrame:

| name | primary_category | price_level | avg_rating | has_hours | total_reviews | response_rate | text_rate | review_span_days |
|---|---|---|---|---|---|---|---|---|
| Hale Pops | Restaurant | NaN | 4.4 | 1 | NaN | NaN | NaN | NaN |
| SMP - Single Marine Program | Recreation center | NaN | 4.1 | 1 | 20.0 | 0.0 | 0.1 | 378.0 |
| 2 Cheesy Guys | Food court | NaN | 5.0 | 1 | NaN | NaN | NaN | NaN |
| Kraken Coffee Kahului | Coffee shop | 1.0 | 4.8 | 1 | NaN | NaN | NaN | NaN |
| Akasatana Ramen Kyoto | Ramen restaurant | NaN | 5.0 | 1 | NaN | NaN | NaN | NaN |

The NaN values in `total_reviews`, `response_rate`, and `text_rate` for rows 0, 2, 3, and 4 reflect businesses that had fewer than 10 reviews in total and were therefore excluded from the 10-core reviews dataset entirely.

---

### Univariate Analysis

#### Distribution of Business Response Rates

<iframe
  src="assets/response_rate_dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The distribution of `response_rate` is heavily right-skewed with a notable concentration at 0. The majority of Hawaii businesses respond to very few or none of their reviews. Most of the distribution mass sits near zero. However, there is a secondary cluster near 1.0, representing businesses that respond to nearly every review they receive. This bimodal shape suggests two meaningfully different engagement strategies in the dataset: businesses that have opted out of review engagement almost entirely, and those that treat review responses as a near-systematic part of customer relations. The large middle ground is relatively sparse, indicating that partial engagement is less common than either extreme.

#### Distribution of Average Business Ratings

<iframe
  src="assets/avg_rating_dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Average business ratings are strongly left-skewed, with the bulk of the distribution concentrated between 4.0 and 5.0. This is a well-documented phenomenon on consumer review platforms sometimes called **positivity bias**: customers who had a bad or mediocre experience are less likely to bother leaving a review, while those who had a great experience are more motivated to share it. The pronounced spike at exactly 5.0 is particularly notable and likely reflects businesses with very few reviews: a single 5-star review produces an average of 5.0 with no variance. This makes `avg_rating` a somewhat noisy signal for low-review businesses, a limitation that becomes important later in the fairness analysis.

---

### Bivariate Analysis

#### Response Rate vs. Average Rating

<iframe
  src="assets/response_rate_vs_rating.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The scatter plot of `response_rate` against `avg_rating` shows considerable variance at every response rate level, but the OLS trendline reveals a modest positive association: businesses that respond to a higher proportion of reviews tend, on average, to have slightly higher ratings. Notably, there are many non-responding businesses (response_rate = 0) that achieve high ratings, confirming that engagement is not a prerequisite for success. The relationship is real but weak, which motivates the hypothesis test in the next section.

#### Mean Rating by Response Rate Bin

<iframe
  src="assets/rating_by_response_bin.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Binning response rate into decile intervals and computing the mean rating for each bin makes the trend easier to see. Average rating generally increases as response rate moves from the lowest bin (0–10%) toward the highest (90–100%). The jump is most pronounced at the upper end of the response rate spectrum, where businesses responding to nearly all their reviews show noticeably higher mean ratings than those in the middle bins. This pattern is consistent with the hypothesis that active review engagement is associated with better ratings, though the aggregate nature of the plot prevents us from drawing causal conclusions.

---

### Aggregates

The table below shows mean `avg_rating` and mean `response_rate` grouped by price level, restricted to businesses that appear in the 10-core reviews dataset (i.e., have non-null engagement features):

| Price Level | Mean Avg Rating | Mean Response Rate | Business Count |
|---|---|---|---|
| $ (1) | 4.24 | 0.18 | 312 |
| $$ (2) | 4.31 | 0.21 | 418 |
| $$$ (3) | 4.38 | 0.27 | 189 |
| $$$$ (4) | 4.44 | 0.31 | 47 |

Both average rating and response rate increase monotonically with price tier. This is an interesting pattern with multiple plausible interpretations. Higher-priced businesses may have dedicated marketing or operations staff who manage their Google presence and respond to reviews as a professional practice, which could explain both higher response rates and, independently, higher service quality and ratings. Alternatively, customers visiting higher-priced establishments may have higher baseline satisfaction expectations that are more consistently met, or may be more inclined to leave generous reviews for premium experiences. The table highlights that price level, review engagement, and ratings are all correlated, disentangling their individual effects is part of what the predictive modeling aims to do.

---

## Assessment of Missingness

### NMAR Analysis

One column that is likely **NMAR (Not Missing At Random)** is `avg_response_delay`. This column is missing for approximately 84.8% of businesses, specifically, all businesses that never responded to a single review. The missingness is not random: it occurs precisely because those businesses chose not to engage with reviews at all. The value of the missing data (what their response delay *would have been*, had they responded) is not what's missing, what's missing is the behavior itself. The missingness is caused by the same underlying tendency that would produce a very high delay or zero responses: disengagement from review management.

To potentially make this MAR, one would need external data on business owner behavior. For example, whether the business owner has claimed their Google Business Profile listing, how frequently they log in to the platform, or whether the business has any staff dedicated to online reputation management. If such variables fully explained why some businesses never respond, the missingness could be attributed to those observable factors rather than to the unobserved delay itself.

---

### Missingness Dependency

I ran permutation tests to determine whether the missingness of `avg_response_delay` depends on other observable columns.

#### Does `avg_response_delay` missingness depend on `total_reviews`?

**Test statistic:** Difference in mean `total_reviews` between businesses where `avg_response_delay` is missing vs. not missing.

<iframe
  src="assets/missingness_permutation.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The observed difference in mean `total_reviews` (missing − present) was **−28.09**, meaning businesses where `avg_response_delay` is missing have on average 28 fewer reviews than businesses where it is present. The permutation test p-value was **0.0000**, under 1,000 random shuffles, none produced a difference as extreme as the one observed.

We reject the null hypothesis. The missingness of `avg_response_delay` is **MAR** with respect to `total_reviews`. This makes intuitive sense from the data generating process: businesses that accumulate more reviews are more visible, more actively managed, and more likely to have an owner who monitors and responds to customer feedback. Businesses with fewer reviews are more likely to be small, under-resourced, or simply unaware of their Google presence — and therefore more likely to never respond.

#### Does `avg_response_delay` missingness depend on `latitude`?

**Test statistic:** Difference in mean `latitude` between businesses where `avg_response_delay` is missing vs. not missing.

The observed latitude difference was negligible, and the permutation test p-value was **0.9400**.

We fail to reject the null hypothesis. Geographic position within Hawaii (as captured by latitude) has no meaningful relationship to whether a business responds to reviews. This makes sense: being located in northern vs. southern Hawaii does not change whether a business owner is likely to engage with their Google profile. The missingness here is effectively **MCAR with respect to latitude**.

---

## Hypothesis Testing

### Do Businesses That Respond to Reviews Have Higher Ratings?

The EDA suggested a positive association between `response_rate` and `avg_rating`, but visualizations alone cannot tell us whether the difference we observe is larger than what we'd expect from random chance. To test this formally, I conducted a permutation test comparing the mean ratings of responding vs. non-responding businesses.

**Null Hypothesis:** The average rating of businesses that respond to at least one review is the same as those that do not respond to any reviews. Any observed difference in mean ratings is due to random chance.

**Alternative Hypothesis:** Businesses that respond to at least one review have a *higher* average rating than businesses that respond to none.

**Test Statistic:** Difference in mean `avg_rating` between the responding group and the non-responding group (responders − non-responders). A mean difference is the natural choice here because `avg_rating` is a continuous numeric variable and we have a directional hypothesis (we expect responders to have *higher* ratings, not just different ones).

**Significance Level:** α = 0.05

**Important note on sample construction:** The test is restricted to businesses that appear in the 10-core reviews dataset (i.e., `total_reviews` is not null). This ensures the non-responder group contains only businesses that genuinely chose not to respond to any review, not businesses that were simply excluded from the reviews dataset due to having fewer than 10 reviews. Including the latter would conflate "no reviews captured" with "chose not to respond," which would corrupt the comparison.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

**Results:** The observed difference in mean ratings was **0.0697 stars** (responders: 4.3831, non-responders: 4.3134). Across 1,000 permutation iterations, zero permuted differences were as large as the observed value, giving a p-value of **0.0000**.

**Conclusion:** At α = 0.05, we reject the null hypothesis. The difference in mean ratings between responding and non-responding businesses is statistically unlikely to have arisen by chance alone. That said, the effect size is small — 0.07 stars on a 5-point scale is a real but modest difference. We should be cautious about overstating the practical significance. More importantly, this is an observational study: we cannot conclude that responding to reviews *causes* higher ratings. It is equally plausible that businesses already earning high ratings from excellent service are simply more motivated (and have more capacity) to engage with the positive feedback they receive. Correlation does not imply causation, and the directional relationship could run either way.

---

## Framing a Prediction Problem

Having established that engagement metrics are associated with ratings, the next question is: **how well can we actually predict a business's average rating from observable features?**

**Prediction Problem:** Predict the average rating (`avg_rating`) of a Hawaii business based on its review engagement behavior and structural business characteristics.

**Type:** This is a **regression problem**. The response variable `avg_rating` is continuous, ranging from 1.0 to 5.0 in the dataset.

**Response Variable:** `avg_rating`. It is the most direct, interpretable measure of customer-perceived quality available in the dataset, and our hypothesis test confirmed it is meaningfully associated with engagement features. Predicting it allows us to quantify how much of a business's rating can be explained by engagement and structural factors alone.

**Features used at time of prediction:** A key constraint in building a prediction model is ensuring that every feature used would actually be available at the time a prediction is made. All features in this model satisfy that condition. They are properties of the business's historical review behavior and metadata, not outcomes that occur after the rating is established:
- `response_rate`, `total_reviews`, `text_rate`, `pics_rate`, `review_span_days` — derived from accumulated review history
- `price_level`, `primary_category`, `has_description`, `has_hours` — business metadata present in the listing

One feature that was considered but excluded is `avg_response_delay`. While conceptually relevant, it is missing for ~85% of businesses (those that never responded at all), which would dramatically reduce the modeling dataset if included. Since `response_rate` captures essentially the same engagement signal with far less missingness, `avg_response_delay` was dropped.

**Evaluation Metric:** **Root Mean Squared Error (RMSE)**. RMSE is reported in the same units as `avg_rating` (stars), making it directly interpretable. An RMSE of 0.30 means our predictions are typically off by 0.30 stars. RMSE penalizes larger errors more than smaller ones, which is appropriate here: predicting a 3-star business as 5 stars is a far more consequential mistake than being off by 0.1 stars. This asymmetry in error severity makes RMSE preferable to Mean Absolute Error (MAE) for this task.

---

## Baseline Model

The baseline model establishes a simple but meaningful performance floor. It uses **Linear Regression** with four features (two quantitative, one ordinal, and one nominal) implemented as a single `sklearn` Pipeline.

**Features:**

- `response_rate` *(quantitative)* — the core engagement variable motivating the project; proportion of reviews responded to
- `total_reviews` *(quantitative)* — a proxy for business visibility and popularity; businesses with more reviews may be better-established and more reliably rated
- `price_level` *(ordinal, treated as quantitative)* — a structural characteristic of the business; encoded as 1–4 and passed in directly as a numeric feature
- `primary_category` *(nominal)* — one-hot encoded using `OneHotEncoder(handle_unknown='ignore')`; captures the fact that, for example, Hotels and Restaurants may have systematically different average rating distributions

**Pipeline steps:**
1. Median imputation for quantitative features (`response_rate`, `total_reviews`, `price_level`)
2. Most-frequent imputation + one-hot encoding for `primary_category`
3. Linear Regression

**Results:**

| Split | RMSE |
|---|---|
| Train | 0.3250 |
| Test | 0.3510 |

The close train and test RMSE values indicate the baseline model is not overfitting. Linear regression with a small feature set naturally has low variance. An RMSE of ~0.35 stars is a reasonable starting point: it means the model's predictions are typically within about a third of a star of the true rating. However, there is meaningful room to improve. Linear regression assumes a linear relationship between features and `avg_rating`, which may not hold. For instance, the relationship between `total_reviews` and `avg_rating` is likely non-linear given the right-skew in review counts. The final model addresses these limitations.

---

## Final Model

### Feature Engineering

Building on the baseline, I engineered four additional features designed to capture aspects of business behavior and rating stability that the raw variables could not represent:

**1. `log_total_reviews`**
`total_reviews` is heavily right-skewed: most businesses have a modest number of reviews, but a small number of very popular businesses have thousands. A raw linear treatment of this variable gives disproportionate weight to outliers and misrepresents the relationship with ratings. Log-transforming compresses the scale, makes the distribution more symmetric, and better captures the intuition that the difference between 10 and 20 reviews matters much more than the difference between 1,000 and 1,010. This transformation is justified by the data generating process: each additional review contributes diminishing new information about a business's true quality.

**2. `rating_std`**
Two businesses can have identical `avg_rating` values but very different review profiles. A restaurant averaging 4.0 from a consistent stream of 4-star reviews signals reliable quality; one averaging 4.0 from a mix of 1- and 5-star reviews signals polarizing, inconsistent experiences. `rating_std` captures this consistency dimension, which `avg_rating` alone cannot. From a prediction standpoint, a business with low rating variance has a more "stable" true rating that should be easier to predict, while high variance may indicate a business in flux.

**3. `text_rate`**
The proportion of reviews with written text indicates how engaged and motivated the business's customer base is. Businesses that inspire customers to write detailed reviews (rather than just click a star) likely deliver experiences worth describing, whether exceptionally good or bad. This is a signal of review quality and reviewer investment that complements `response_rate`, which measures engagement from the business side.

**4. `review_span_days`**
A business that has been receiving reviews for 1,000 days has had its rating shaped by far more customer interactions over a longer time horizon than one with 30 days of reviews. Longer review spans tend to produce more stable, regressed-toward-the-mean ratings. This feature captures business maturity and longevity in the review ecosystem, which is relevant to how reliable and stable the `avg_rating` signal is.

### Modeling Algorithm and Hyperparameter Search

I chose a **Random Forest Regressor** as the final modeling algorithm. Unlike Linear Regression, Random Forests can capture non-linear relationships (e.g., the curved relationship between `log_total_reviews` and rating) and feature interactions (e.g., a high `response_rate` might matter more for businesses in some categories than others) without requiring these to be manually specified. Random Forests are also robust to the scale of input features and handle imputed missing values gracefully.

Before tuning, I identified three hyperparameters most likely to affect model performance:

- **`n_estimators`**: The number of trees in the ensemble. More trees reduce variance at the cost of computation time. I tested [100, 200].
- **`max_depth`**: How deep each tree is allowed to grow. Deeper trees fit training data more precisely but risk overfitting. I tested [5, 10, 20, None (unlimited)].
- **`min_samples_split`**: The minimum number of samples required to split an internal node. Larger values prevent the model from making very specific rules on small subsets of data. I tested [2, 5].

`GridSearchCV` with 5-fold cross-validation was used to evaluate all combinations. The best configuration found was:

**`max_depth=10`, `min_samples_split=2`, `n_estimators=200`**

<iframe
  src="assets/actual_vs_predicted.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

<iframe
  src="assets/feature_importance.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Results

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Baseline (Linear Regression) | 0.3250 | 0.3510 |
| Final (Random Forest) | 0.2088 | **0.2649** |

The final model reduced test RMSE from 0.3510 to 0.2649 — an improvement of **0.086 stars**, or a **24.5% reduction** in prediction error. The gap between train RMSE (0.2088) and test RMSE (0.2649) indicates some overfitting, which is expected with Random Forests. The `max_depth=10` constraint meaningfully controls this: without it, the unconstrained trees overfit substantially more.

The engineered features `log_total_reviews` and `rating_std` likely drove most of the improvement. `log_total_reviews` fixed the non-linearity that the baseline's linear treatment of `total_reviews` could not capture, while `rating_std` gave the model a direct signal about review consistency that explained variance in `avg_rating` that engagement metrics alone could not.

---

## Fairness Analysis

A model that performs well on average can still be unfair if it systematically makes worse predictions for certain groups of businesses. Here, I ask: **does the model perform worse for low-volume businesses than high-volume businesses?**

This is a meaningful fairness question because the engagement features that drive our model (`response_rate`, `text_rate`, `rating_std`) are all computed from review counts. For a business with only 10 reviews, a `response_rate` of 0.5 means it responded to 5 reviews; for a business with 200 reviews, it means 100 responses. The former is a much noisier estimate of true engagement behavior. If our model relies heavily on these noisy estimates for low-volume businesses, predictions for that group will suffer.

**Groups:**
- **Group X (low-volume):** businesses with `total_reviews` strictly below the median
- **Group Y (high-volume):** businesses with `total_reviews` at or above the median

**Evaluation Metric:** RMSE — appropriate for a regression model. We measure whether RMSE is higher for Group X than Group Y.

**Null Hypothesis:** Our model is fair with respect to review volume. The RMSE for low-volume businesses and the RMSE for high-volume businesses are equal; any observed difference is due to random chance.

**Alternative Hypothesis:** Our model is less accurate for low-volume businesses. The RMSE for low-volume businesses is strictly greater than the RMSE for high-volume businesses.

**Test Statistic:** Difference in RMSE (low-volume RMSE − high-volume RMSE). A positive value means the model is worse for low-volume businesses.

**Significance Level:** α = 0.05

<iframe
  src="assets/fairness_permutation.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

**Results:**
- Low-volume RMSE: **0.3285**
- High-volume RMSE: **0.1808**
- Observed RMSE difference (Low − High): **0.1477**
- P-value: **0.0000**

**Conclusion:** At α = 0.05, we reject the null hypothesis. The model is significantly less accurate for low-volume businesses, and this disparity is not attributable to random chance. The observed difference of 0.1477 stars falls far outside the distribution of permuted differences, which cluster tightly around zero.

This is a substantive limitation worth acknowledging. The model's predictions are nearly twice as error-prone for low-volume businesses (RMSE = 0.33) as for high-volume ones (RMSE = 0.18). When a business has only 10–15 reviews, its `response_rate`, `text_rate`, and `rating_std` are inherently unstable estimates, one unusual review can swing them significantly. The model was essentially trained on more reliable signals for high-volume businesses and generalizes less well when those signals are noisy. Any deployment of this model should be accompanied by a clear caveat that predictions for businesses with few reviews carry substantially higher uncertainty, and that smaller businesses, which make up a large share of Hawaii's business landscape, are most affected.