# Review Notes — Sequential Pattern Mining & AdaBoost

## 1. Sequential Pattern Mining (SPM)

### 1.1 What is Sequential Pattern Mining?

Sequential Pattern Mining (SPM) is a data mining technique that discovers **statistically relevant patterns in ordered sequences of events**. Unlike traditional classification where each sample is independent, SPM specifically looks for **temporal relationships** — patterns where the order in which events occur matters.

**Formal definition:** Given a database of sequences, SPM finds all subsequences that appear with a frequency above a user-defined minimum support threshold.

**Example in network security:**
```
Packet sequence: SYN → SYN → SYN → SYN → RST → RST
Pattern detected: "4+ SYN packets followed by RST within 2 seconds" → SYN flood attack
```

**Common SPM algorithms:**
- **AprioriAll** — Extension of Apriori for sequential data
- **GSP (Generalized Sequential Patterns)** — Handles time constraints and taxonomies
- **PrefixSpan** — Prefix-projected sequential pattern mining (most efficient)
- **SPADE** — Sequential Pattern Discovery using Equivalence classes

**Typical applications:**
- Web clickstream analysis (user navigated Page A → B → C → purchase)
- Market basket analysis over time (customer bought milk on Monday, bread on Tuesday)
- DNA sequence analysis (gene subsequence discovery)
- Network intrusion detection at the **packet level** (raw packet sequences)

---

### 1.2 Why SPM Does Not Apply to Our System

Our system operates on the **UNSW-NB15 dataset**, which is a **connection-level feature dataset**, not a raw packet capture. This distinction is critical:

#### What our data looks like (connection-level):

| Record | rate | sload | sbytes | dload | proto | ... | Label |
|--------|------|-------|--------|-------|-------|-----|-------|
| 0 | -0.526 | -0.429 | -0.044 | -0.349 | -0.77 | ... | Normal |
| 1 | 2.341 | 3.108 | 1.205 | 0.052 | -0.77 | ... | DoS |

Each row represents **one completed network connection** described by 10 pre-computed aggregate features. Record #0 has **no temporal relationship** to Record #1 — they are independent observations.

#### What SPM would need (packet-level):

```
Time 0.001s: SRC→DST  SYN     (64 bytes)
Time 0.002s: SRC→DST  SYN     (64 bytes)
Time 0.003s: SRC→DST  SYN     (64 bytes)
Time 0.004s: DST→SRC  RST     (40 bytes)
Time 0.005s: SRC→DST  SYN     (64 bytes)
...
```

SPM needs the **raw ordered sequence** of individual packets or events to find temporal patterns.

#### The fundamental mismatch:

| Aspect | Our System (XGBoost) | SPM Requirement |
|--------|---------------------|-----------------|
| **Data unit** | One complete connection (1 row) | Ordered sequence of events |
| **Temporal order** | Not preserved — features are aggregates | Essential — order defines the pattern |
| **Independence** | Each record is classified independently | Records must be analyzed as a sequence |
| **Features** | Pre-computed statistics (rate, sload, duration) | Raw events (packet type, timestamp, size) |
| **Input format** | Fixed-size feature vector (10 values) | Variable-length sequence of events |

#### The sequential information is already captured in the features:

The UNSW-NB15 dataset's feature engineering has already **summarized the sequential behavior** into single numeric values:

- **`rate`** (packets/second) — Already encodes "how many packets occurred in what time span." SPM would discover "many packets in short time = flood" — but `rate` already captures this.
- **`sload`** (source bits/second) — Already encodes the bandwidth pattern over the connection's lifetime.
- **`dur`** (connection duration) — Already encodes how long the sequence lasted.
- **`sbytes`** (total source bytes) — Already encodes the cumulative volume.

SPM would attempt to rediscover patterns that are **already embedded in these aggregate features**. This makes it redundant.

---

### 1.3 Why XGBoost is the Correct Choice

| Criterion | XGBoost | SPM |
|-----------|---------|-----|
| **Data type** | Tabular features (our data) | Sequential events (not our data) |
| **Performance on tabular data** | State-of-the-art | Not designed for tabular data |
| **Explainability** | SHAP TreeExplainer provides exact Shapley values | No established XAI framework |
| **Inference speed** | ~0.01s for 41,000 records | Depends on sequence length; typically slower |
| **Our F1 score** | 90.26% on imbalanced benchmark | N/A — would need different data format |
| **Integration with mitigation** | SHAP top features → attack type → iptables commands | No direct mapping to mitigation |

**Conclusion:** SPM is a valid technique for network intrusion detection **when working with raw packet captures or event logs**. However, our project works with the UNSW-NB15 dataset which provides **pre-computed connection-level features**. The temporal information that SPM would discover is already captured within features like `rate`, `sload`, `dur`, and `sbytes`. XGBoost is the appropriate model for this data format, and it provides the additional benefit of compatibility with SHAP for explainable AI — a core requirement of our project.

---

## 2. AdaBoost (Adaptive Boosting)

### 2.1 What is AdaBoost?

AdaBoost (Adaptive Boosting) is an **ensemble learning algorithm** proposed by Yoav Freund and Robert Schapire in 1997. It was one of the first practical boosting algorithms and won the Godel Prize in 2003 for its theoretical significance.

**Core idea:** Combine many "weak learners" (classifiers that are only slightly better than random guessing) into a single "strong learner" by focusing on the mistakes of previous learners.

---

### 2.2 How AdaBoost Works (Step-by-Step)

**Setup:** Given a training dataset of N samples, each sample starts with equal weight: `w_i = 1/N`

**Round 1:**
1. Train a weak classifier (typically a decision stump — a tree with just 1 split) on the weighted data
2. Calculate its weighted error rate: `err = sum of weights of misclassified samples`
3. Calculate the classifier's importance: `alpha = 0.5 * ln((1 - err) / err)`
   - High accuracy → high alpha (this classifier gets more voting power)
   - Low accuracy → low alpha
4. **Update sample weights:**
   - Misclassified samples: weight **increases** → `w_i * e^(+alpha)`
   - Correctly classified samples: weight **decreases** → `w_i * e^(-alpha)`
5. Normalize all weights so they sum to 1

**Round 2:**
1. Train a new weak classifier on the **reweighted** data
   - This classifier naturally focuses on the previously misclassified samples (they now have higher weights)
2. Repeat steps 2-5

**After T rounds:**
- Final prediction = **weighted majority vote** of all T classifiers:
  ```
  H(x) = sign( alpha_1 * h_1(x) + alpha_2 * h_2(x) + ... + alpha_T * h_T(x) )
  ```
  where each `h_t` is a weak classifier and `alpha_t` is its importance weight

**Visual illustration of the process:**

```
Round 1:  All samples equal weight
          Train stump → misclassifies some samples
          ● ● ● ○ ○ ○ ● ○    (● = correct, ○ = wrong)

Round 2:  Wrong samples get HIGHER weight (bigger)
          Train new stump → focuses on previously hard cases
          ● ● ○ ● ● ● ○ ●    (fixes some, but new errors)

Round 3:  Again, wrong samples get HIGHER weight
          Train new stump → focuses on remaining hard cases
          ● ● ● ● ● ○ ● ●

Final:    Combine all 3 stumps with weighted voting
          → Much better than any single stump alone
```

---

### 2.3 The Evolution: AdaBoost → Gradient Boosting → XGBoost

AdaBoost was the starting point. Each subsequent algorithm improved on the previous one:

#### AdaBoost (1997) — Freund & Schapire
- **Mechanism:** Reweights misclassified samples
- **Loss function:** Exponential loss only
- **Weakness:** Sensitive to noisy data and outliers (because it keeps increasing weights on hard-to-classify samples, which may just be noise)

#### Gradient Boosting (2001) — Jerome Friedman
- **Key insight:** Instead of reweighting samples, fit each new tree to the **residual errors** (gradients) of the previous ensemble
- **Mechanism:** Uses gradient descent in function space
- **Loss function:** Any differentiable loss function (log loss, squared error, huber loss, etc.)
- **Improvement over AdaBoost:** More flexible, better theoretical framework, handles different loss functions

#### XGBoost (2016) — Tianqi Chen
- **Key insight:** Add **regularization** to prevent overfitting + engineering optimizations for speed
- **Improvements over Gradient Boosting:**
  - **L1 and L2 regularization** on leaf weights (prevents overfitting)
  - **Tree pruning** using max_depth (stops growing before overfitting)
  - **Column subsampling** (like Random Forest — each tree sees only a subset of features)
  - **Built-in handling of missing values** (learns optimal default direction)
  - **Parallel tree construction** (splits can be evaluated in parallel)
  - **Cache-aware access patterns** (optimized for CPU cache efficiency)
  - **Approximate split finding** for large datasets (weighted quantile sketch)

**Comparison Table:**

| Feature | AdaBoost | Gradient Boosting | XGBoost |
|---------|----------|-------------------|---------|
| **Year** | 1997 | 2001 | 2016 |
| **Learning mechanism** | Reweight misclassified samples | Fit trees to residual errors (gradients) | Gradient descent + regularization |
| **Loss function** | Exponential only | Any differentiable | Any differentiable + regularization terms |
| **Overfitting control** | Limited (early stopping only) | Learning rate, subsampling | L1/L2 regularization, pruning, column sampling |
| **Missing values** | Cannot handle | Cannot handle | Handles natively (learns default direction) |
| **Outlier sensitivity** | High (keeps boosting weight on outliers) | Moderate | Low (regularization dampens extreme weights) |
| **Parallelization** | Sequential only | Sequential only | Parallel split evaluation |
| **Speed** | Slow | Moderate | Fast |
| **Typical weak learner** | Decision stump (depth 1) | Shallow tree (depth 3-6) | Shallow tree (depth 3-8) |

---

### 2.4 Why XGBoost Over AdaBoost for Our Project

We chose XGBoost over AdaBoost for several specific reasons:

**1. Regularization prevents overfitting on imbalanced data**

Our test set is highly imbalanced (37,000 Normal vs. 4,089 DoS — roughly 90/10). AdaBoost would keep increasing weights on the minority DoS samples that get misclassified, potentially overfitting to noise in those samples. XGBoost's L1/L2 regularization prevents this.

**2. Better threshold optimization**

XGBoost outputs calibrated probabilities via `predict_proba()`, which allowed us to optimize the classification threshold to 0.8517 using our validation set. AdaBoost's exponential loss can produce less well-calibrated probabilities.

**3. SHAP TreeExplainer compatibility**

SHAP's TreeExplainer provides **exact** (not approximate) Shapley values for tree-based models. Both AdaBoost with decision trees and XGBoost are compatible, but XGBoost's deeper trees produce more nuanced SHAP explanations because each tree captures more complex feature interactions.

**4. Superior performance on tabular data**

XGBoost consistently outperforms AdaBoost on structured/tabular datasets in benchmarks (including Kaggle competitions). For our UNSW-NB15 dataset:
- XGBoost achieved **90.26% F1 score** on the imbalanced benchmark test
- The regularization and gradient-based optimization give it an edge on datasets with noise

**5. Inference speed**

XGBoost processes 41,089 records in ~0.006 seconds. While AdaBoost with stumps might be faster per tree, it typically needs more trees (hundreds to thousands) to match XGBoost's accuracy, making total inference time comparable or slower.

---

### 2.5 How to Explain This in the Review

**If asked "Why didn't you use AdaBoost?":**

> "AdaBoost is the original boosting algorithm that introduced the concept of combining weak learners by focusing on misclassified samples. XGBoost is its modern successor — it uses the same boosting philosophy but replaces sample reweighting with gradient descent and adds regularization to prevent overfitting. For our imbalanced UNSW-NB15 dataset (90% Normal, 10% DoS), XGBoost's regularization is critical — AdaBoost would risk overfitting to noise in the minority class. Additionally, XGBoost's calibrated probability outputs allowed us to optimize our detection threshold at 0.8517, and its compatibility with SHAP TreeExplainer gives us exact feature explanations for every prediction."

**If asked "What is the relationship between AdaBoost and XGBoost?":**

> "XGBoost is an evolution of AdaBoost. Both are boosting algorithms — they sequentially build an ensemble of weak learners where each new learner corrects the mistakes of the previous ones. AdaBoost (1997) does this by reweighting misclassified samples. Gradient Boosting (2001) improved on this by fitting new trees to residual errors using gradient descent. XGBoost (2016) further improved by adding L1/L2 regularization, tree pruning, and parallel computation. The core idea — 'focus on what you got wrong' — remains the same across all three."

**If asked "Could you add AdaBoost to your model comparison?":**

> "Yes, AdaBoost could be added as an 8th model in our comparison. However, based on existing literature and benchmarks on the UNSW-NB15 dataset, AdaBoost typically achieves lower F1 scores than XGBoost due to its lack of regularization and its sensitivity to the class imbalance in the test set. Our current 7-model comparison already includes a spectrum from simple models (Logistic Regression) to complex ones (LSTM, CNN), with XGBoost as the best performer."

---

### 2.6 Key Definitions for Quick Reference

| Term | Definition |
|------|-----------|
| **Weak learner** | A classifier that performs only slightly better than random guessing (e.g., a decision stump with ~55% accuracy) |
| **Strong learner** | A classifier with high accuracy, created by combining many weak learners |
| **Boosting** | Ensemble technique that trains models sequentially, with each new model focusing on the errors of the previous ones |
| **Bagging** | Ensemble technique that trains models in parallel on random subsets (e.g., Random Forest). Unlike boosting, models are independent |
| **Decision stump** | A decision tree with only one split (depth = 1). The most common weak learner in AdaBoost |
| **Ensemble** | A collection of models whose predictions are combined (by voting or averaging) to produce a final prediction |
| **Regularization** | Techniques that penalize model complexity to prevent overfitting (L1 = sparsity, L2 = small weights) |
| **Gradient descent** | Optimization technique that iteratively adjusts parameters in the direction that reduces the loss function |
| **Shapley values** | From cooperative game theory — the fair contribution of each feature to a prediction, considering all possible feature combinations |
