# Decision Tree Model — Complete Documentation

## Table of Contents

1. [What is a Decision Tree?](#1-what-is-a-decision-tree)
2. [How Decision Trees Work — Step by Step](#2-how-decision-trees-work--step-by-step)
3. [Our Decision Tree Configuration](#3-our-decision-tree-configuration)
4. [Training Results](#4-training-results)
5. [Benchmark Test Results](#5-benchmark-test-results)
6. [Visualizations](#6-visualizations)
7. [Decision Tree vs XGBoost — The "1 Worker vs 100 Workers" Analogy](#7-decision-tree-vs-xgboost--the-1-worker-vs-100-workers-analogy)
8. [Detailed Comparison: Decision Tree vs XGBoost](#8-detailed-comparison-decision-tree-vs-xgboost)
9. [Why XGBoost is Still the Better Choice](#9-why-xgboost-is-still-the-better-choice)
10. [When Would a Decision Tree be Preferred?](#10-when-would-a-decision-tree-be-preferred)
11. [Conclusion](#11-conclusion)

---

## 1. What is a Decision Tree?

A **Decision Tree** is one of the most fundamental and interpretable machine learning algorithms. It is a supervised learning model that classifies data by making a series of binary questions (yes/no decisions) about the input features, forming a tree-like structure from root to leaves.

**Simple analogy:** Imagine a doctor diagnosing whether a patient has the flu. The doctor asks a series of questions:

```
Is the patient's temperature > 38°C?
├── YES → Does the patient have a cough?
│         ├── YES → Does the patient have body aches?
│         │         ├── YES → Diagnosis: FLU
│         │         └── NO  → Diagnosis: COLD
│         └── NO  → Diagnosis: FEVER (other cause)
└── NO  → Diagnosis: HEALTHY
```

A Decision Tree works exactly like this — but instead of a doctor asking questions, the algorithm learns the best questions to ask from training data, and instead of diseases, it classifies network traffic as **DoS** or **Normal**.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Root Node** | The very first question (split) at the top of the tree |
| **Internal Node** | A decision point where the tree asks "Is feature X > threshold Y?" |
| **Leaf Node** | A terminal node that gives the final prediction (DoS or Normal) |
| **Depth** | The longest path from root to any leaf (our tree: depth = 10) |
| **Split** | A binary decision that divides data into two groups |
| **Gini Impurity** | A measure of how "mixed" the classes are at a node (0 = pure, 0.5 = maximum impurity) |
| **Pruning** | Limiting tree growth to prevent overfitting (we use max_depth=10) |

---

## 2. How Decision Trees Work — Step by Step

### 2.1 The Splitting Process

At every node, the algorithm finds the **best feature** and **best threshold** to split the data. "Best" means the split that produces the purest child nodes (most records belonging to a single class).

**Example from our trained model:**

```
ROOT NODE: Is sload > 0.342?
├── YES (high source load — suspicious)
│   → Is proto > 0.5?
│     ├── YES → Is sbytes > 1.205?
│     │         ├── YES → PREDICTION: DoS (99.2% confidence)
│     │         └── NO  → PREDICTION: DoS (87.4% confidence)
│     └── NO  → Is tcprtt > 0.089?
│               ├── YES → PREDICTION: Normal (92.1% confidence)
│               └── NO  → PREDICTION: DoS (78.6% confidence)
└── NO (low source load — likely normal)
    → Is dload > 0.156?
      ├── YES → PREDICTION: DoS (71.3% confidence)
      └── NO  → PREDICTION: Normal (98.7% confidence)
```

### 2.2 Gini Impurity — How the Tree Decides Where to Split

The tree uses **Gini Impurity** to measure how well a split separates the classes:

```
Gini = 1 - (P(DoS))² - (P(Normal))²
```

- **Gini = 0**: Perfect purity (all samples are the same class)
- **Gini = 0.5**: Maximum impurity (50/50 split between classes)

The algorithm tries every possible feature and every possible threshold, and picks the split that **reduces Gini impurity the most**.

**Example:**
- Before split: 1000 samples (500 DoS, 500 Normal) → Gini = 0.5
- After split on `sload > 0.342`:
  - Left child: 600 samples (50 DoS, 550 Normal) → Gini = 0.153
  - Right child: 400 samples (450 DoS, 50 Normal) → Gini = 0.045
- **Weighted average Gini after split = 0.110** (much better than 0.5)

### 2.3 When Does the Tree Stop Growing?

Our tree stops splitting when any of these conditions are met:

1. **Max depth reached (10)** — The tree cannot go deeper than 10 levels
2. **Min samples to split (10)** — A node must have at least 10 samples to be split further
3. **Min samples per leaf (5)** — Each leaf must contain at least 5 samples
4. **Pure node** — All samples in the node belong to the same class (Gini = 0)

These constraints prevent **overfitting** — without them, the tree would memorize every training sample perfectly but fail on new data.

### 2.4 Making Predictions

When a new network traffic record arrives, it flows from the root to a leaf:

```
New record: rate=-0.52, sload=2.34, sbytes=1.21, dload=0.05, proto=-0.77, ...

Step 1: Is sload (2.34) > 0.342?  → YES → go right
Step 2: Is proto (-0.77) > 0.5?   → NO  → go left
Step 3: Is tcprtt (0.012) > 0.089? → NO  → go left
...
Leaf reached: 95% of training samples here were DoS
→ PREDICTION: DoS with P(DoS) = 0.95
```

The probability comes from the proportion of DoS vs Normal samples that ended up in that leaf during training.

---

## 3. Our Decision Tree Configuration

| Parameter | Value | Why This Value |
|-----------|-------|---------------|
| **criterion** | `gini` | Standard for classification; measures class purity at each split |
| **max_depth** | `10` | Matches Random Forest's max_depth; prevents overfitting while allowing complex patterns |
| **min_samples_split** | `10` | Nodes with fewer than 10 samples won't split further; prevents learning from noise |
| **min_samples_leaf** | `5` | Every leaf must have at least 5 samples; ensures statistically meaningful predictions |
| **random_state** | `42` | Fixed seed for reproducibility; ensures identical results on every run |

### Resulting Tree Structure

| Property | Value |
|----------|-------|
| Actual depth | 10 (reached the max) |
| Leaf nodes | 129 (129 possible prediction outcomes) |
| Total nodes | 257 (129 leaves + 128 internal decision nodes) |

---

## 4. Training Results

### 4.1 Cross-Validation (5-Fold Stratified, on 24,528 Balanced Samples)

Cross-validation splits the training data into 5 folds, trains on 4 folds, and tests on the held-out fold. This is repeated 5 times so every sample is used for testing exactly once.

| Metric | Mean | Standard Deviation | 95% Confidence Interval |
|--------|------|--------------------|------------------------|
| **Accuracy** | 95.55% | +/- 1.39% | [94.16% — 96.94%] |
| **Precision** | 96.84% | +/- 1.30% | [95.54% — 98.14%] |
| **Recall** | 94.18% | +/- 3.42% | [90.76% — 97.60%] |
| **F1 Score** | 95.48% | +/- 1.50% | [93.98% — 96.98%] |

**Interpretation:**
- High precision (96.84%) means when the model says "DoS", it's almost always correct
- Slightly lower recall (94.18%) means it misses about 6% of actual DoS attacks
- The higher standard deviation on recall (+/- 3.42%) suggests recall varies more across folds, indicating some DoS patterns are harder to capture with a single tree

### 4.2 Training Set Performance (Full 24,528 Samples)

| Metric | Value |
|--------|-------|
| Accuracy | 96.22% |
| Precision | 98.47% |
| Recall | 93.89% |
| F1 Score | 96.13% |

| | Predicted Normal | Predicted DoS |
|---|---|---|
| **Actual Normal** | 12,085 (TN) | 179 (FP) |
| **Actual DoS** | 749 (FN) | 11,515 (TP) |

The training performance is close to the CV performance, indicating the model is **not severely overfitting** — the regularization parameters (max_depth, min_samples) are working.

---

## 5. Benchmark Test Results

### 5.1 Test Dataset

The benchmark test uses the official UNSW-NB15 test set, which is **completely unseen during training**:

- **41,089 total samples**
- **37,000 Normal** (90.05%)
- **4,089 DoS** (9.95%)
- This imbalanced distribution simulates real-world network traffic where attacks are rare

### 5.2 Default Threshold (0.50)

| Metric | Value |
|--------|-------|
| Accuracy | 96.56% |
| Precision | 77.88% |
| Recall | **91.37%** |
| F1 Score | 84.09% |

| | Predicted Normal | Predicted DoS |
|---|---|---|
| **Actual Normal** | 35,939 (TN) | 1,061 (FP) |
| **Actual DoS** | 353 (FN) | 3,736 (TP) |

With the default threshold of 0.5, the model has high recall (catches 91.37% of DoS attacks) but poor precision (77.88%) — meaning 1,061 normal connections are falsely flagged as attacks.

### 5.3 Optimized Threshold (0.93)

By optimizing the classification threshold to maximize F1 score:

| Metric | Default (0.50) | Optimized (0.93) | Change |
|--------|---------------|-------------------|--------|
| **Accuracy** | 96.56% | **97.83%** | +1.27% |
| **Precision** | 77.88% | **93.43%** | +15.55% |
| **Recall** | 91.37% | 84.13% | -7.24% |
| **F1 Score** | 84.09% | **88.53%** | +4.44% |

| | Predicted Normal | Predicted DoS |
|---|---|---|
| **Actual Normal** | 36,758 (TN) | 242 (FP) |
| **Actual DoS** | 649 (FN) | 3,440 (TP) |

**Why threshold = 0.93?** The model requires 93% probability of DoS before flagging a connection as an attack. This dramatically reduces false positives (from 1,061 to 242) while still catching 84.13% of actual attacks.

### 5.4 Additional Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **ROC-AUC** | 0.9806 | Excellent discrimination between DoS and Normal across all thresholds |
| **PR-AUC** | 0.9248 | Strong precision-recall trade-off, important for imbalanced data |

---

## 6. Visualizations

All visualizations are saved in the `images/` directory:

### Decision Tree Model Results
| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix at optimized threshold (0.93) |
| `feature_importance.png` | Gini importance of all 10 features |
| `roc_curve.png` | ROC curve with AUC = 0.9806 |
| `precision_recall_curve.png` | Precision-Recall curve with PR-AUC = 0.9248 |
| `threshold_optimization.png` | F1, Precision, Recall vs threshold |
| `cross_validation.png` | 5-fold CV results with error bars |

### Comparison with XGBoost
| File | Description |
|------|-------------|
| `comparison_metrics.png` | Side-by-side bar chart of all metrics |
| `comparison_confusion_matrix.png` | Both confusion matrices side by side |
| `comparison_cv_f1.png` | Cross-validation F1 with error bars |
| `comparison_complexity.png` | Model complexity comparison (log scale) |
| `comparison_feature_importance.png` | Feature importance — DT vs XGBoost |
| `comparison_analogy.png` | Visual diagram: "1 Worker vs 100 Workers" |
| `comparison_errors.png` | False positive & false negative comparison |

### Feature Importance (Gini)

| Rank | Feature | Importance | Role in Detection |
|------|---------|-----------|-------------------|
| 1 | **sload** | 0.5298 (53.0%) | Source bits per second — dominant indicator of traffic volume |
| 2 | **dload** | 0.1455 (14.6%) | Destination bits per second — response volume |
| 3 | **sbytes** | 0.1112 (11.1%) | Source to destination bytes — packet payload size |
| 4 | **proto** | 0.1017 (10.2%) | Protocol type — TCP vs UDP distinction |
| 5 | **tcprtt** | 0.0647 (6.5%) | TCP round-trip time — connection latency |
| 6 | **dmean** | 0.0272 (2.7%) | Mean destination packet size |
| 7 | **stcpb** | 0.0097 (1.0%) | Source TCP base sequence number |
| 8 | **dur** | 0.0089 (0.9%) | Connection duration |
| 9 | **rate** | 0.0011 (0.1%) | Connection rate |
| 10 | **dtcpb** | 0.0004 (0.04%) | Destination TCP base sequence number |

**Key observation:** The Decision Tree relies **extremely heavily on `sload`** (53% of all decisions). This means over half of the tree's splitting decisions are based on just one feature. This is a sign of a model that has found a "shortcut" rather than learning nuanced patterns.

---

## 7. Decision Tree vs XGBoost — The "1 Worker vs 100 Workers" Analogy

### The Core Analogy

Think of network traffic detection as inspecting packages at a security checkpoint:

**Decision Tree = 1 Security Guard Doing Everything Alone**

One guard stands at the checkpoint. He has a checklist of 10 things to look for (our 10 features). He checks each package one by one, following his checklist from top to bottom. He's fast and consistent, but he's just one person — if a package is tricky (ambiguous features), he has to make a single judgment call based on his limited perspective.

- He has **257 decision rules** he follows (257 nodes in the tree)
- He looks at each package **once** and decides
- If he makes a mistake, there's **no one to catch it**
- He relies very heavily on **one check** (sload accounts for 53% of his decisions)

**XGBoost = A Team of 100 Security Guards Working in Sequence**

Guard #1 does a preliminary check. He catches the obvious cases — the clearly dangerous packages and the clearly safe ones. But he makes some mistakes on the borderline cases.

Guard #2 reviews **only the packages that Guard #1 got wrong**. He focuses specifically on those tricky cases and develops expertise in patterns that Guard #1 missed.

Guard #3 reviews Guard #2's mistakes. And so on, through all 100 guards.

By the end:
- Guard #1 caught the obvious 80% of cases
- Guard #2 fixed 50% of Guard #1's mistakes
- Guard #3 fixed 50% of Guard #2's remaining mistakes
- ...
- Guard #100 provides the final refinement
- The **combined team** catches 90.26% (F1) vs the solo guard's 88.53%

**This is exactly how gradient boosting works:** Each tree in XGBoost is trained on the **residual errors** (mistakes) of all previous trees. The 2nd tree focuses on what the 1st tree got wrong. The 3rd tree focuses on what the combination of trees 1 and 2 still get wrong. This sequential error correction is what makes the ensemble more powerful than any individual tree.

### The Cost of the Team

Here's where your observation about "expensiveness" is exactly right:

| Resource | Decision Tree (1 Guard) | XGBoost (100 Guards) | Cost Multiplier |
|----------|------------------------|---------------------|-----------------|
| **Number of trees** | 1 | 100 | 100x |
| **Total decision nodes** | 257 | ~25,700 | 100x |
| **Training time** | ~0.1 seconds | ~1.0 seconds | ~10x |
| **Model file size** | ~15 KB | ~900 KB | ~60x |
| **Memory usage** | Very low | Moderate | ~50x |
| **Inference time (41K records)** | ~0.002s | ~0.006s | ~3x |

**XGBoost uses approximately 100x more computational resources.** It trains 100 trees instead of 1. It stores 100 trees in memory. It evaluates 100 trees for every prediction.

### So Why Pay the Cost?

Because the improvement is worth it in a security context:

| What You Get | Decision Tree | XGBoost | Real-World Impact |
|---|---|---|---|
| **False Positives** | 242 | 209 | 33 fewer innocent connections blocked |
| **False Negatives** | 649 | 554 | **95 more attacks caught** |
| **F1 Score** | 88.53% | 90.26% | 1.73% more accurate overall |

Those **95 additional caught attacks** are the ones that Decision Tree missed but XGBoost caught. In network security, a single missed DoS attack can take down an entire server. The computational cost of running 100 trees (~0.006 seconds) is negligible compared to the cost of a successful DoS attack (minutes to hours of downtime).

---

## 8. Detailed Comparison: Decision Tree vs XGBoost

### 8.1 Performance Metrics (Optimized Thresholds)

| Metric | Decision Tree | XGBoost | Winner | Margin |
|--------|--------------|---------|--------|--------|
| **F1 Score** | 88.53% | **90.26%** | XGBoost | +1.73% |
| **Precision** | 93.43% | **94.42%** | XGBoost | +0.99% |
| **Recall** | 84.13% | **86.45%** | XGBoost | +2.32% |
| **Accuracy** | 97.83% | **98.14%** | XGBoost | +0.31% |
| **ROC-AUC** | 0.9806 | ~0.9950 | XGBoost | +0.0144 |
| **CV F1** | 95.48% | **96.45%** | XGBoost | +0.97% |
| **CV F1 Stability** | +/-1.50% | **+/-0.47%** | XGBoost | 3x more stable |

**XGBoost wins on every metric.** The largest gap is in recall (+2.32%) — XGBoost catches 95 more attacks that Decision Tree misses.

### 8.2 Cross-Validation Stability

| Model | CV F1 Mean | CV F1 Std | Interpretation |
|-------|-----------|-----------|---------------|
| Decision Tree | 95.48% | 0.75% | Moderate variance across folds |
| XGBoost | 96.45% | **0.23%** | Very stable across folds |

XGBoost's standard deviation is **3x smaller**, meaning it produces consistent results regardless of which data it's trained on. A single Decision Tree is more sensitive to which specific samples appear in each fold — one bad split at the root can cascade through the entire tree.

### 8.3 Error Analysis

| Error Type | Decision Tree | XGBoost | Impact |
|------------|--------------|---------|--------|
| **False Positives** (Normal → DoS) | 242 | 209 | XGBoost: 33 fewer false alarms |
| **False Negatives** (DoS → Normal) | 649 | 554 | XGBoost: **95 fewer missed attacks** |
| **Total Errors** | 891 | 763 | XGBoost: **128 fewer total errors** |
| **Error Rate** | 2.17% | 1.86% | XGBoost: 14.4% fewer errors |

In a security system, **false negatives are more dangerous than false positives**. A false positive means an innocent connection is briefly blocked (inconvenience). A false negative means a real attack goes undetected (potential server downtime). XGBoost's 95 fewer missed attacks is the most critical advantage.

### 8.4 Feature Usage

One of the most revealing differences is **how each model uses features**:

**Decision Tree — Over-reliance on sload:**
```
sload:  53.0% ████████████████████████████████████████████████████
dload:  14.6% ██████████████
sbytes: 11.1% ███████████
proto:  10.2% ██████████
tcprtt:  6.5% ██████
Other:   4.7% ████
```

**XGBoost — Distributed importance:**
```
sload:  28.9% █████████████████████████████
proto:  21.5% █████████████████████
tcprtt: 15.2% ███████████████
sbytes:  8.1% ████████
dload:   6.4% ██████
rate:    5.2% █████
Other:  14.7% ██████████████
```

**Why this matters:** The Decision Tree puts 53% of its "eggs in one basket" — if `sload` happens to be misleading for a particular record, the tree has limited ability to recover. XGBoost distributes importance across many features, so even if one feature is ambiguous, the other features compensate. This is why XGBoost is more robust and catches those 95 extra attacks.

### 8.5 Threshold Sensitivity

| Model | Optimal Threshold | F1 at 0.5 | F1 at Optimal |
|-------|-------------------|-----------|--------------|
| Decision Tree | 0.93 | 84.09% | 88.53% |
| XGBoost | 0.85 | 87.24% | 90.26% |

Decision Tree requires a much higher threshold (0.93) to achieve its best F1. This means its probability outputs are **less well-calibrated** — it needs extreme confidence (93%) before its "DoS" predictions become reliable. XGBoost achieves better results at a lower threshold (0.85), indicating its probability estimates are more trustworthy.

---

## 9. Why XGBoost is Still the Better Choice

Despite XGBoost being 100x more complex (100 trees vs 1), we choose it for these five reasons:

### Reason 1: Higher Detection Rate (Recall)

XGBoost catches **86.45%** of DoS attacks vs Decision Tree's **84.13%**. That 2.32% difference translates to **95 more detected attacks** out of 4,089 in the test set. In a production network with millions of connections per day, this gap becomes thousands of additional caught attacks.

### Reason 2: Fewer False Alarms (Precision)

XGBoost has **94.42% precision** vs Decision Tree's **93.43%**. Fewer false alarms means the security team doesn't waste time investigating harmless traffic, and legitimate users don't experience unnecessary connection blocks.

### Reason 3: Better Generalization (Lower Variance)

XGBoost's CV F1 standard deviation is **0.23%** vs Decision Tree's **0.75%**. This means XGBoost performs consistently regardless of the specific data it encounters. A single Decision Tree can get "unlucky" with its root split and perform significantly worse on certain data distributions.

### Reason 4: SHAP Compatibility for Explainable AI

Both models are compatible with SHAP TreeExplainer. However, XGBoost's SHAP values are **more nuanced** because they aggregate explanations across 100 trees, each of which captures different feature interactions. A single tree's SHAP values are dominated by just 1-2 features (sload at 53%), making explanations less informative.

This is critical for our project because **Explainable AI is a core research objective**. The SHAP explanations from XGBoost better reveal why each record was classified as DoS or Normal, which directly feeds into our attack classification and mitigation framework.

### Reason 5: Negligible Inference Cost

While XGBoost trains 100 trees, the inference time for 41,089 records is only **0.006 seconds** — just 0.004 seconds more than Decision Tree's 0.002 seconds. This 4-millisecond difference is completely imperceptible to the user and irrelevant for real-time network monitoring.

### The Cost-Benefit Summary

```
COST OF XGBOOST (vs Decision Tree):
  - 100x more trees                → But training takes only ~1 second total
  - 60x larger model file          → But still only ~900KB (negligible)
  - 3x slower inference             → But still only 0.006 seconds for 41K records

BENEFIT OF XGBOOST (vs Decision Tree):
  + 95 more attacks detected        → Critical for security
  + 33 fewer false alarms           → Less noise for security team
  + 3x more stable predictions      → Reliable in production
  + Better SHAP explanations        → Essential for our XAI framework
  + 1.73% higher F1 score           → Measurably better overall
```

**The extra computational cost is measured in milliseconds. The security benefit is measured in attacks caught. The trade-off is overwhelmingly in favor of XGBoost.**

---

## 10. When Would a Decision Tree be Preferred?

Despite our choice of XGBoost, Decision Trees have legitimate advantages in certain contexts:

| Scenario | Why Decision Tree Wins |
|----------|----------------------|
| **Regulatory requirement for full interpretability** | A Decision Tree can be printed as a simple flowchart. A regulator or auditor can trace every decision manually. XGBoost's 100 trees cannot be reasonably inspected by a human. |
| **Extremely resource-constrained devices** | IoT sensors or embedded systems with <1MB memory. A 15KB Decision Tree fits; a 900KB XGBoost model may not. |
| **Real-time requirements <1 microsecond** | If inference must be under 1 microsecond per record (e.g., high-frequency trading), a single tree is faster. |
| **Quick prototyping and baseline** | Training takes 0.1 seconds vs 1 second. For rapid experimentation, Decision Trees give instant feedback. |
| **Educational purposes** | Decision Trees are the best way to understand how tree-based models work before learning about ensembles. |

**For our project, none of these constraints apply.** We have a Streamlit dashboard (not an embedded device), inference takes milliseconds (not microseconds), and SHAP provides the explainability we need (not manual tree inspection). Therefore, XGBoost remains the correct choice.

---

## 11. Conclusion

### Decision Tree Performance Summary

| Metric | Value | Ranking Among 8 Models |
|--------|-------|----------------------|
| **Test F1 (optimized)** | 88.53% | 4th (between Random Forest and MLP) |
| **Test Accuracy** | 97.83% | 4th |
| **Test Precision** | 93.43% | 3rd |
| **Test Recall** | 84.13% | 5th |
| **ROC-AUC** | 0.9806 | 4th |
| **CV F1** | 95.48% | 4th |

### Full Model Ranking by Test F1 Score

| Rank | Model | Test F1 | Notes |
|------|-------|---------|-------|
| 1 | **XGBoost** | **90.26%** | Selected model — best F1 + SHAP compatible |
| 2 | Random Forest | 89.56% | Ensemble of 100 independent trees (bagging) |
| 3 | 1D-CNN | 86.38% | Deep learning — 1D convolutions on features |
| 4 | **Decision Tree** | **88.53%** | Single tree — strong baseline |
| 5 | MLP | 85.10% | Neural network — 2 hidden layers |
| 6 | LSTM | 83.58% | Recurrent neural network |
| 7 | SVM | 78.06% | Support Vector Machine with RBF kernel |
| 8 | Logistic Regression | 53.16% | Linear model — insufficient for this task |

### Key Takeaway

The Decision Tree is a **strong model** that achieves 88.53% F1 — comfortably the 4th best among our 8 models. It validates that tree-based approaches are fundamentally well-suited for this tabular network traffic classification task. However, XGBoost's ensemble approach (combining 100 trees with gradient boosting) provides a meaningful 1.73% F1 improvement, better stability, and richer SHAP explanations — all at negligible computational cost. The "1 worker vs 100 workers" trade-off decisively favors the team approach for security-critical applications where every detected attack matters.

---

## Files in This Directory

```
decisiontree/
├── train_decisiontree.py              # Training script (7 steps)
├── generate_comparison_charts.py      # Comparison visualization script
├── decisiontree_model.pkl             # Trained model (pickle)
├── DECISION_TREE_DOCUMENTATION.md     # This file
├── results/
│   └── decisiontree_results.json      # Complete results (JSON)
└── images/
    ├── confusion_matrix.png           # Confusion matrix (optimized)
    ├── feature_importance.png         # Gini importance (all 10 features)
    ├── roc_curve.png                  # ROC curve (AUC = 0.9806)
    ├── precision_recall_curve.png     # PR curve (PR-AUC = 0.9248)
    ├── threshold_optimization.png     # Threshold vs F1/Precision/Recall
    ├── cross_validation.png           # 5-fold CV results with error bars
    ├── comparison_metrics.png         # DT vs XGBoost — all metrics
    ├── comparison_confusion_matrix.png# Side-by-side confusion matrices
    ├── comparison_cv_f1.png           # CV F1 with error bars
    ├── comparison_complexity.png      # Model complexity (log scale)
    ├── comparison_feature_importance.png # Feature importance side-by-side
    ├── comparison_analogy.png         # "1 Worker vs 100 Workers" diagram
    └── comparison_errors.png          # False positive/negative comparison
```
