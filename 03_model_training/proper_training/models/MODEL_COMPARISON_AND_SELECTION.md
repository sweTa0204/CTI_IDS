# Model Comparison & Selection: Why XGBoost Over LSTM and 1D-CNN

## Overview

This document explains why we trained three different model architectures for DoS attack detection, what each model does, how they performed, and why XGBoost remains the recommended model for deployment.

---

## The Three Models We Trained

### 1. XGBoost (Extreme Gradient Boosting) — Tree-Based

**How it works — Think of it like 100 doctors:**

```
Doctor 1 checks the patient:
  Is packet rate > 5000?  → Yes
  Is source bytes > 10000? → Yes
  → "Probably an attack" (70% confident)

Doctor 2 looks at what Doctor 1 got WRONG:
  Is duration < 0.1s? → Yes
  Is destination load low? → Yes
  → "Definitely an attack" (fixes Doctor 1's mistake)

Doctor 3 looks at what Doctor 2 got WRONG:
  ... and so on for 100 doctors

FINAL ANSWER = Combined opinion of all 100 doctors
```

Each "doctor" is a Decision Tree. Each new tree focuses on **correcting the mistakes** of the previous trees. This is called **Gradient Boosting**.

**Key Property:** Each network flow is checked **independently**. Row 1 has no connection to Row 2. The model sees one sample at a time.

---

### 2. LSTM (Long Short-Term Memory) — Recurrent Neural Network

**How it works — Think of it like reading a story:**

```
Reading a book, page by page:

Page 1: "A man entered the bank"
  → Brain remembers: bank

Page 2: "He was wearing a mask"
  → Brain remembers: bank + mask

Page 3: "He reached into his pocket"
  → Brain predicts: ROBBERY! (because it remembers the context)
```

LSTM has a **memory cell** that carries information forward. What it saw in step 1 influences its decision in step 3.

**For network traffic (ideal scenario):**
```
Packet 1: Normal HTTP request       → Memory: nothing unusual
Packet 2: Another request, no wait  → Memory: getting frequent
Packet 3: Flood of requests         → Memory: THIS IS AN ATTACK!
```

The model recognizes the pattern **across time** — not just from a single packet, but from the **sequence of events**.

**Key Property:** Designed for **sequential/time-series data** where order and history matter.

---

### 3. 1D-CNN (1D Convolutional Neural Network) — Pattern Detector

**How it works — Think of it like a magnifying glass:**

```
Our 10 features laid out in a row:
[rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur]

The CNN slides a small window (size 3) across the features:

Step 1: Looks at [rate, sload, sbytes]    → "High traffic pattern detected!"
Step 2: Looks at [sload, sbytes, dload]   → "Asymmetric load pattern!"
Step 3: Looks at [sbytes, dload, proto]   → "Suspicious protocol usage!"
...and so on

Each window looks for LOCAL PATTERNS — specific combinations
of adjacent features that signal an attack.
```

**Real-world analogy:**
Like airport security scanning luggage through an X-ray machine. The scanner doesn't look at the entire bag at once. It slides across the bag looking for **suspicious shapes/patterns** in each small section.

**Key Property:** Finds **local feature patterns** (signatures) without needing memory or history.

---

## Performance Results

### The Final Numbers

| Metric | XGBoost | 1D-CNN | LSTM |
|--------|---------|--------|------|
| **F1 Score** | **90.57%** | 86.38% | 83.58% |
| **Accuracy** | **97.76%** | 97.42% | 96.89% |
| **Precision** | **94.41%** | 90.92% | 88.12% |
| **Recall** | **87.09%** | 82.27% | 79.48% |
| **AUC** | **0.9915** | 0.9780 | 0.9683 |
| **Optimal Threshold** | 0.8517 | 0.8700 | 0.7900 |

### Visual Ranking

```
F1 Score:

  XGBoost  ██████████████████████████████████████████████████  90.57%  ← BEST
  1D-CNN   ████████████████████████████████████████████        86.38%
  LSTM     ██████████████████████████████████████████          83.58%
           |---------|---------|---------|---------|---------|
           0%       20%       40%       60%       80%      100%
```

### What These Numbers Mean in Practice

**Out of 41,089 test samples (37,000 Normal + 4,089 DoS):**

```
XGBoost:
  ✓ Caught 3,561 out of 4,089 attacks (87.09%)
  ✗ Missed 528 attacks
  ✗ 209 false alarms (flagged normal traffic as attacks)

1D-CNN:
  ✓ Caught 3,364 out of 4,089 attacks (82.27%)
  ✗ Missed 725 attacks
  ✗ 335 false alarms

LSTM:
  ✓ Caught 3,250 out of 4,089 attacks (79.48%)
  ✗ Missed 839 attacks
  ✗ 438 false alarms
```

**XGBoost catches 311 MORE attacks** than LSTM while raising **229 FEWER false alarms**.

---

## Why XGBoost Wins — The Core Reason

### It Comes Down to Data Format

The UNSW-NB15 dataset provides **aggregated flow-level features**:

```
What our data looks like:
┌──────────────────────────────────────────────────────────────┐
│ Each row = One COMPLETE network flow (already summarized)    │
│                                                              │
│ Row 1: rate=5000, bytes=12000, dur=0.3s  → DoS              │
│ Row 2: rate=50,   bytes=500,   dur=2.1s  → Normal           │
│ Row 3: rate=8000, bytes=15000, dur=0.1s  → DoS              │
│                                                              │
│ These rows are INDEPENDENT. Row 1 has no relation to Row 2. │
│ Each row is a standalone summary of a complete flow.         │
└──────────────────────────────────────────────────────────────┘
```

This is **tabular data** — rows and columns, like an Excel spreadsheet. And XGBoost is literally the **king of tabular data**.

### What Each Model Was Built For

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Data Type               Best Model         Our Data?      │
│   ─────────               ──────────         ─────────      │
│   Tabular (rows/columns)  XGBoost            ✓ YES          │
│   Time-series (sequences) LSTM               ✗ NO           │
│   Signal patterns         1D-CNN             ✗ Partial      │
│   Images                  2D-CNN             ✗ NO           │
│   Text                    Transformers       ✗ NO           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Using LSTM on tabular data is like using a hammer to turn a screw. It works, but a screwdriver (XGBoost) does it better.**

---

## Detailed Reason Breakdown

### Reason 1: No Time Sequences in Our Data

**LSTM's superpower is MEMORY.** It remembers what it saw before.

```
What LSTM needs (packet-by-packet data):

  Time 0s: [SYN packet]          → Memory stores: connection started
  Time 1s: [Partial header]      → Memory stores: incomplete request
  Time 2s: [Partial header]      → Memory stores: still incomplete...
  Time 3s: [Partial header]      → Memory decides: SLOWLORIS ATTACK!

  The SEQUENCE over time is what reveals the attack.
```

```
What our data gives LSTM:

  Row 1: [5000, 800, 12000, ...]  → Memory stores: ... nothing useful
  Row 2: [50, 200, 500, ...]      → Memory stores: ... completely unrelated flow
  Row 3: [8000, 900, 15000, ...]  → Memory stores: ... another unrelated flow

  These are DIFFERENT flows from DIFFERENT connections.
  There is NO sequence to remember.
```

**Result:** LSTM's memory goes unused. It degrades into a basic neural network, which is always weaker than XGBoost on tabular data.

---

### Reason 2: Too Few Features for CNN to Shine

**1D-CNN's superpower is finding LOCAL PATTERNS** in feature sequences.

```
CNN with 100+ features (ideal):
  [..., rate, sload, sbytes, dload, proto, ...]
   ↑────────────────↑
   The sliding window has plenty of room to find
   complex multi-feature patterns (attack signatures)

CNN with only 10 features (our case):
  [rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur]
   ↑──────────↑
   Only 10 features total. Window of size 3 can only form
   8 different combinations. Not enough for CNN advantage.
```

**XGBoost doesn't have this limitation.** It can combine ANY features regardless of their position — feature 1 with feature 8, feature 3 with feature 10. CNN can only combine **adjacent** features.

---

### Reason 3: Small Dataset Favors Tree Models

```
Dataset Size and Model Performance:

  Samples        Best Model Type
  ──────         ───────────────
  < 1,000        Simple models (Logistic Regression, SVM)
  1,000-50,000   Tree-based (XGBoost, Random Forest)  ← OUR CASE (24,528)
  50,000-500K    Trees or Neural Networks (either can win)
  > 500,000      Neural Networks (LSTM, CNN, Transformers)
```

Neural networks are **data hungry**. They have thousands of parameters (LSTM: 22,209, CNN: 30,000+) that all need to be tuned. With only 24,528 training samples, they can't fully learn.

XGBoost with 100 trees is far more **sample-efficient** — it learns faster from less data.

---

### Reason 4: XGBoost Handles Feature Interactions Naturally

```
XGBoost automatically learns:

  Decision Tree Branch:
  ├── IF rate > 5000
  │   ├── AND sbytes > 10000
  │   │   ├── AND dur < 0.5
  │   │   │   → DoS Attack (98% confidence)

  It naturally combines ANY features at ANY level.
  No preprocessing, no reshaping, no special architecture needed.
```

```
LSTM/CNN need to LEARN these interactions:

  1. Reshape data into special format
  2. Feed through multiple layers
  3. Hope the network discovers the same relationships
  4. Requires more data and more training time

  They CAN learn it, but they need more data and time.
```

---

## Speed and Resource Comparison

### Training Time

| Model | Training Time | Why |
|-------|--------------|-----|
| **XGBoost** | ~2-3 seconds | Simple split-finding operations |
| **1D-CNN** | ~1-2 minutes | Matrix multiplications, 64 epochs |
| **LSTM** | ~2-3 minutes | Sequential processing, 100 epochs |

**XGBoost trains 40-60x faster than neural networks.**

### Prediction Speed (How Fast It Classifies New Traffic)

| Model | Time per Sample | Samples per Second | Operations |
|-------|-----------------|-------------------|------------|
| **XGBoost** | ~0.1 ms | ~10,000/sec | Simple if-else comparisons across 100 trees |
| **1D-CNN** | ~0.5-1 ms | ~1,000-2,000/sec | Convolution + matrix multiplication |
| **LSTM** | ~1-2 ms | ~500-1,000/sec | Sequential gate computations |

**For a real-time Intrusion Detection System, speed matters.**

A busy network might process 5,000+ flows per second. Only XGBoost can keep up without dedicated GPU hardware.

```
Network Traffic Rate: 5,000 flows/second

  XGBoost: 10,000/sec capacity  → ✓ Handles easily (50% utilization)
  1D-CNN:   1,500/sec capacity  → ✗ Falls behind (needs 3.3x more power)
  LSTM:       750/sec capacity  → ✗ Falls far behind (needs 6.7x more power)
```

### Resource Requirements

| Resource | XGBoost | 1D-CNN | LSTM |
|----------|---------|--------|------|
| **Model File Size** | ~150 KB | ~200 KB | ~318 KB |
| **RAM Usage** | ~50 MB | ~500 MB (TensorFlow) | ~500 MB (TensorFlow) |
| **GPU Required** | No | Helpful | Helpful |
| **Dependencies** | xgboost, sklearn | TensorFlow (~500 MB install) | TensorFlow (~500 MB install) |

XGBoost is **lightweight and portable**. It can run on any machine without heavy frameworks.

---

## Explainability (XAI) Comparison

Our project focuses on **Explainable AI**. This is another area where XGBoost excels.

| XAI Aspect | XGBoost | 1D-CNN | LSTM |
|------------|---------|--------|------|
| **SHAP Method** | TreeExplainer (exact) | DeepExplainer (approximate) | DeepExplainer (approximate) |
| **SHAP Speed** | Fast (~seconds) | Slow (~minutes) | Slow (~minutes) |
| **Feature Importance** | Clear and direct | Indirect (through gradients) | Indirect (through gradients) |
| **Human Understanding** | "rate was the top factor" | Hard to interpret | Hard to interpret |
| **Trust Building** | Easy to explain to non-experts | Requires ML expertise | Requires ML expertise |

```
XGBoost + SHAP Explanation (Easy to understand):
─────────────────────────────────────────────────
"This traffic was classified as DoS because:
  1. Packet rate was 8000/sec (normally < 100)     → +0.35 toward DoS
  2. Source bytes were 15000 (normally < 1000)      → +0.28 toward DoS
  3. Duration was 0.05s (normally > 1s)             → +0.15 toward DoS
  Total: 0.78 → Above threshold 0.85 → DoS Attack"

LSTM Explanation (Hard to understand):
──────────────────────────────────────
"The hidden state activations in layer 1 neurons 23, 47, and 61
showed high activation patterns correlated with the output neuron..."

→ This means nothing to a network administrator.
```

---

## When Would LSTM or CNN Beat XGBoost?

It's important to note that LSTM and CNN are not bad models. They simply need different data.

### LSTM Would Win With:

| Scenario | Example | Why LSTM Excels |
|----------|---------|-----------------|
| **Packet-level data** | Individual packets captured over time | Can track how an attack develops |
| **Session monitoring** | Watching one connection evolve | Remembers connection history |
| **Slow-rate attacks** | Slowloris (sends data very slowly over minutes) | Only detectable by looking at patterns over time |
| **Network traffic streams** | Continuous packet captures (PCAP files) | Natural sequential format |

### 1D-CNN Would Win With:

| Scenario | Example | Why CNN Excels |
|----------|---------|----------------|
| **Payload analysis** | Raw bytes of network packets | Finds malicious byte patterns |
| **Many features (50+)** | Large feature vectors with local structure | More room for pattern detection |
| **Signal processing** | Network bandwidth measurements over time | Detects frequency patterns |
| **Large datasets (100K+)** | Massive network captures | CNN learns better with more data |

---

## Summary Table

| Criteria | XGBoost | 1D-CNN | LSTM |
|----------|---------|--------|------|
| **F1 Score** | 90.57% | 86.38% | 83.58% |
| **Training Speed** | Fastest (seconds) | Medium (minutes) | Slowest (minutes) |
| **Prediction Speed** | Fastest (0.1ms) | Medium (0.5-1ms) | Slowest (1-2ms) |
| **Resource Usage** | Lowest | High | High |
| **Explainability** | Best (SHAP) | Poor | Poor |
| **Best Data Type** | Tabular (our data) | Sequential signals | Time-series |
| **Our Ranking** | 1st | 2nd | 3rd |

---

## Final Conclusion

### Why XGBoost is Our Recommended Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  RECOMMENDED: XGBoost (F1: 90.57%, Threshold: 0.8517)               │
│                                                                      │
│  1. BEST PERFORMANCE    → Highest F1, Precision, Recall, AUC        │
│  2. FASTEST PREDICTION  → 10,000 samples/sec (real-time capable)    │
│  3. BEST EXPLAINABILITY → SHAP TreeExplainer (exact values)         │
│  4. LOWEST RESOURCES    → No GPU, no TensorFlow, small model        │
│  5. RIGHT FIT           → Tabular flow data = XGBoost's strength    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Why We Still Implemented LSTM and 1D-CNN

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  VALUE OF MULTI-MODEL APPROACH:                                      │
│                                                                      │
│  1. ADDRESSES REVIEWER FEEDBACK                                      │
│     → "Why not try sequence models?"                                 │
│     → We did. XGBoost still wins for this data format.               │
│                                                                      │
│  2. COMPREHENSIVE RESEARCH                                           │
│     → Explored tree-based, recurrent, and convolutional approaches   │
│     → Data-driven model selection, not assumption-driven             │
│                                                                      │
│  3. JUSTIFIED SELECTION                                              │
│     → XGBoost wasn't chosen blindly                                  │
│     → We compared and proved it's the best for THIS dataset          │
│                                                                      │
│  4. FUTURE WORK FOUNDATION                                           │
│     → If packet-level data becomes available, LSTM is ready          │
│     → If raw payload analysis is needed, CNN is ready                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### For the Research Paper

**Methodology Section:**
> "We evaluated three distinct model architectures to ensure comprehensive analysis: XGBoost (tree-based ensemble), LSTM (recurrent neural network), and 1D-CNN (convolutional neural network). This multi-model approach was chosen to compare tree-based, sequential, and convolutional paradigms for DoS detection."

**Results Section:**
> "XGBoost achieved the highest F1 score (90.57%), followed by 1D-CNN (86.38%) and LSTM (83.58%). All models used optimized decision thresholds found via F1-score maximization."

**Discussion Section:**
> "The dominance of XGBoost is attributed to the tabular nature of UNSW-NB15's aggregated flow features. LSTM's memory mechanism provides no advantage when samples represent independent, pre-summarized network flows rather than temporal packet sequences. Similarly, 1D-CNN's local pattern detection is limited by the small feature space (10 features). For future deployments with packet-level or streaming data, sequence-based models may offer superior performance."

---

*Document Generated: 2026-02-03*
*Models Compared: XGBoost, LSTM, 1D-CNN*
*Dataset: UNSW-NB15 (24,528 training, 41,089 testing)*
