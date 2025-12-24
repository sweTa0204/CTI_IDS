# 🔄 Complete Project Flow: XAI-Powered DDoS Mitigation System

## 📖 Introduction

This document explains the complete flow of our DDoS Mitigation System from start to finish. It is designed for team members who are looking at this project for the first time.

**Team Name:** Threat Hunters  
**Members:** Shweta Sharma, Devika, Aakash  
**Challenge:** CSIC 1.0 - Systems & Software Security  
**Document Created:** December 22, 2025

---

## 🎯 What Does Our System Do?

Our system protects networks from **DDoS (Distributed Denial of Service) attacks** by:
1. **Detecting** malicious traffic using Machine Learning
2. **Explaining** why traffic was detected as malicious using XAI
3. **Blocking** attacks and preventing future similar attacks

---

## 🏗️ System Architecture Overview

```
                              COMPLETE SYSTEM FLOW
═══════════════════════════════════════════════════════════════════════════════

                           INTERNET / NETWORK TRAFFIC
                                      │
                                      │ Incoming packets
                                      ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │         PHASE 1: BPF FILTER         │
                    │         (The Gatekeeper)            │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                    KNOWN ATTACK               UNKNOWN TRAFFIC
                    (Pattern Match)            (No Match)
                         │                           │
                         ▼                           │
                    🚫 BLOCK                         │
                    (Instant!)                       │
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │    PHASE 2: FEATURE EXTRACTION      │
                    │    (The Translator)                 │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       │ 10 Features
                                       ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │    PHASE 3: ML MODEL                │
                    │    (The Brain)                      │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                      NORMAL                       DoS ATTACK
                      TRAFFIC                      DETECTED
                         │                           │
                         ▼                           │
                    ✅ ALLOW                         │
                    (Pass through)                   │
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │    PHASE 4: XAI EXPLANATION         │
                    │    (The Explainer)                  │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       │ Why it's DoS + Confidence
                                       ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │    PHASE 5: MITIGATION              │
                    │    (The Enforcer)                   │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                         ▼             ▼             ▼
                    🚫 BLOCK      📝 UPDATE     📢 ALERT
                    Traffic       BPF Rules     Security
                                     │           Team
                                     │
                                     └────────────────────┐
                                                          │
                    Back to Phase 1 ◄─────────────────────┘
                    (BPF now knows this pattern!)

═══════════════════════════════════════════════════════════════════════════════
```

---

## 📋 Detailed Step-by-Step Flow

---

### PHASE 1: BPF FILTER (The Gatekeeper)

#### What is BPF?
**BPF (Berkeley Packet Filter)** is a super-fast filtering system that works at the kernel level of the operating system. It can process millions of packets per second.

#### What does it do?
- Acts as the **first line of defense**
- Checks incoming traffic against **known attack patterns (signatures)**
- Blocks known attacks **instantly** (microseconds)
- Passes unknown traffic to the ML model for analysis

#### How does it work?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   INCOMING PACKET                                                           │
│   ───────────────                                                           │
│   Source IP: 192.168.1.100                                                  │
│   Bytes: 48,000                                                             │
│   Rate: 13,000 packets/sec                                                  │
│   Protocol: UDP                                                             │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   SIGNATURE DATABASE (Rules)                                                │
│   ──────────────────────────                                                │
│   Rule 1: IF sbytes > 45000 AND rate > 12000 AND proto = UDP → BLOCK       │
│   Rule 2: IF source_ip = 10.0.0.50 → BLOCK                                  │
│   Rule 3: IF packet_size = 64 AND rate > 50000 → BLOCK                     │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   CHECK: Does packet match any rule?                                        │
│                                                                             │
│          Rule 1: sbytes(48000) > 45000? ✓                                   │
│                  rate(13000) > 12000? ✓                                     │
│                  proto = UDP? ✓                                             │
│                                                                             │
│          MATCH FOUND!                                                       │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   ACTION: 🚫 BLOCK IMMEDIATELY                                              │
│   (Packet never reaches the server)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Important Note: First Time System Starts

```
WHEN SYSTEM FIRST STARTS:
═══════════════════════════════════════════════════════════════════

BPF Signature Database = EMPTY (no rules yet!)

Result: ALL traffic passes to ML Model for analysis

As ML detects attacks → Signatures are created → BPF learns!

After some time: BPF has learned many attack patterns
                 Most attacks blocked instantly by BPF
                 Only NEW unknown attacks go to ML

═══════════════════════════════════════════════════════════════════
```

#### Key Points:
- ⚡ **Speed**: Microseconds (0.001 ms) per packet
- 📍 **Location**: Runs in Linux kernel (very fast)
- 📚 **Rules**: Pattern-based (NOT hash-based like malware signatures)
- 🔄 **Updates**: Receives new rules from Phase 5 (Mitigation)

---

### PHASE 2: FEATURE EXTRACTION (The Translator)

#### What is Feature Extraction?
It converts **raw network traffic** into **10 parameters/features** that our ML model can understand.

#### Why is it needed?
- Raw network packets have hundreds of data points
- ML model needs specific, meaningful features
- We scientifically selected the 10 most important features

#### The 10 Features We Extract:

| # | Feature | What It Means | How to Calculate |
|---|---------|---------------|------------------|
| 1 | **rate** | Packets per second | count(packets) / time |
| 2 | **sload** | Source data rate | source_bytes / duration |
| 3 | **sbytes** | Total bytes from source | sum(source_bytes) |
| 4 | **dload** | Destination data rate | dest_bytes / duration |
| 5 | **proto** | Protocol type | TCP=0, UDP=1, etc. |
| 6 | **dtcpb** | Destination TCP bytes | TCP bytes to destination |
| 7 | **stcpb** | Source TCP bytes | TCP bytes from source |
| 8 | **dmean** | Mean packet delay | average(packet_delays) |
| 9 | **tcprtt** | TCP round-trip time | Time for TCP handshake |
| 10 | **dur** | Connection duration | end_time - start_time |

#### Feature Extraction Process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RAW NETWORK TRAFFIC                                                       │
│   ───────────────────                                                       │
│   • Packet 1: src=192.168.1.1, dst=10.0.0.1, bytes=1500, time=0.001s       │
│   • Packet 2: src=192.168.1.1, dst=10.0.0.1, bytes=1400, time=0.002s       │
│   • Packet 3: src=192.168.1.1, dst=10.0.0.1, bytes=1600, time=0.003s       │
│   • ... (many more packets)                                                 │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   FEATURE EXTRACTION ENGINE                                                 │
│   ─────────────────────────                                                 │
│   Calculates:                                                               │
│   • rate = 3 packets / 0.003s = 1000 pps                                   │
│   • sbytes = 1500 + 1400 + 1600 = 4500                                     │
│   • sload = 4500 / 0.003 = 1,500,000                                       │
│   • dur = 0.003 seconds                                                     │
│   • ... (calculate all 10)                                                  │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   SCALING (StandardScaler)                                                  │
│   ────────────────────────                                                  │
│   Converts to standard range (mean=0, std=1)                                │
│   So all features have equal importance                                     │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   OUTPUT: 10 SCALED FEATURES                                                │
│   ──────────────────────────                                                │
│   [0.85, -0.32, 1.45, 0.12, 0, -0.67, 0.89, -1.23, 0.45, 0.78]            │
│                                                                             │
│   Ready for ML Model!                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why These 10 Features?

We started with **48 features** from the UNSW-NB15 dataset and reduced to **10** through:

```
FEATURE ENGINEERING PIPELINE:
═══════════════════════════════════════════════════════════════════

48 features (original)
    │
    ▼ Data Cleanup (remove IDs, metadata)
48 features
    │
    ▼ Categorical Encoding (text → numbers)
48 features
    │
    ▼ Correlation Analysis (remove redundant)
34 features
    │
    ▼ Variance Analysis (remove low-variance)
18 features
    │
    ▼ Statistical Testing (keep significant only)
10 features ← FINAL

═══════════════════════════════════════════════════════════════════
```

#### Key Points:
- 📊 **Input**: Raw network packets
- 📤 **Output**: 10 scaled numbers
- 🔬 **Method**: Statistical analysis
- ✅ **Result**: 76% reduction in features, 100% significance

---

### PHASE 3: ML MODEL (The Brain)

#### What is the ML Model?
A **trained Machine Learning model** (XGBoost) that predicts whether traffic is **Normal** or a **DoS Attack**.

#### How was it trained?
- **Dataset**: UNSW-NB15 (8,178 samples, 50% Normal, 50% DoS)
- **Features**: 10 engineered features
- **Algorithm**: XGBoost (winner among 5 tested models)
- **Accuracy**: 95.54%

#### Model Comparison (Why XGBoost?):

| Model | Accuracy | F1-Score | ROC-AUC | Verdict |
|-------|----------|----------|---------|---------|
| **XGBoost** | **95.54%** | **95.47%** | **98.91%** | 🏆 Champion |
| Random Forest | 95.29% | 95.22% | 98.67% | Strong |
| MLP Neural Network | 92.48% | 92.16% | 97.35% | Good |
| SVM | 90.04% | 89.73% | 96.12% | Moderate |
| Logistic Regression | 78.18% | 76.89% | 84.52% | Baseline |

#### How Prediction Works:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   INPUT: 10 Scaled Features                                                 │
│   ─────────────────────────                                                 │
│   [0.85, -0.32, 1.45, 0.12, 0, -0.67, 0.89, -1.23, 0.45, 0.78]            │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                                                             │          │
│   │                    XGBoost Model                            │          │
│   │                                                             │          │
│   │   • 100+ decision trees                                     │          │
│   │   • Each tree votes: Normal or DoS?                        │          │
│   │   • Final decision = majority vote                          │          │
│   │                                                             │          │
│   │   Internal process:                                         │          │
│   │   Tree 1: "sbytes is high → DoS"                           │          │
│   │   Tree 2: "rate is high → DoS"                             │          │
│   │   Tree 3: "duration is short → DoS"                        │          │
│   │   ...                                                       │          │
│   │   Tree 100: "Overall pattern → DoS"                        │          │
│   │                                                             │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   OUTPUT:                                                                   │
│   ───────                                                                   │
│   Prediction: DoS Attack (1)                                                │
│   Confidence: 95.2%                                                         │
│   Probability: [0.048, 0.952]  (4.8% Normal, 95.2% DoS)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Decision Flow:

```
                         ML MODEL OUTPUT
                              │
                              ▼
                    ┌─────────────────┐
                    │   Prediction?   │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
          NORMAL (0)                  DoS ATTACK (1)
               │                           │
               ▼                           ▼
         ✅ ALLOW                    Continue to XAI
         Traffic passes              for explanation
         to destination              
```

#### Key Points:
- 🧠 **Type**: XGBoost (Gradient Boosting)
- 📈 **Accuracy**: 95.54%
- ⚡ **Speed**: Milliseconds per prediction
- 📊 **Output**: Prediction (0/1) + Confidence (0-100%)

---

### PHASE 4: XAI EXPLANATION (The Explainer)

#### What is XAI?
**XAI (Explainable AI)** tells us **WHY** the model made its decision. Instead of just saying "This is an attack", it explains which features caused the detection.

#### Why do we need XAI?
- **Trust**: Security teams can verify if the detection makes sense
- **Debugging**: If wrong, we can see which feature misled the model
- **Compliance**: Some regulations require explainable decisions
- **Learning**: Helps us understand attack patterns

#### Two XAI Methods We Use:

##### 1. SHAP (SHapley Additive exPlanations)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   SHAP EXPLANATION                                                          │
│   ────────────────                                                          │
│                                                                             │
│   Prediction: DoS Attack (95% confidence)                                   │
│                                                                             │
│   Base value (neutral): 50%                                                 │
│                                                                             │
│   Feature Contributions:                                                    │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │                                                                │       │
│   │   sbytes (45,000)  ████████████████████████  +25%  → DoS      │       │
│   │   rate (12,000)    ████████████████         +15%  → DoS      │       │
│   │   sload (8,500)    ████████████             +10%  → DoS      │       │
│   │   proto (UDP)      ██████                    +5%  → DoS      │       │
│   │   dmean (0.002)    ████                      -5%  → Normal   │       │
│   │                                                                │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   Calculation:                                                              │
│   50% (base) + 25% + 15% + 10% + 5% - 5% = 100% → Capped at 95%           │
│                                                                             │
│   INTERPRETATION:                                                           │
│   "This traffic was detected as DoS because:                                │
│    - Source bytes (45,000) is abnormally high (+25%)                       │
│    - Packet rate (12,000/s) indicates flooding (+15%)                      │
│    - Source load (8,500) shows high data transfer (+10%)"                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 2. LIME (Local Interpretable Model-agnostic Explanations)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LIME EXPLANATION                                                          │
│   ────────────────                                                          │
│                                                                             │
│   Creates simple IF-THEN rules for this specific prediction:                │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │                                                                │       │
│   │   IF sbytes > 40,000                                          │       │
│   │      → 70% likely DoS                                         │       │
│   │                                                                │       │
│   │   IF rate > 10,000                                            │       │
│   │      → 65% likely DoS                                         │       │
│   │                                                                │       │
│   │   IF proto = UDP AND sbytes > 40,000                          │       │
│   │      → 85% likely DoS                                         │       │
│   │                                                                │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   INTERPRETATION:                                                           │
│   "For this traffic, the simple rule is:                                    │
│    IF bytes > 40,000 AND rate > 10,000 THEN it's likely a DoS attack"      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### XAI Output Summary:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   XAI OUTPUT (Passed to Mitigation)                                         │
│   ─────────────────────────────────                                         │
│                                                                             │
│   {                                                                         │
│     "prediction": "DoS Attack",                                             │
│     "confidence": 95.2,                                                     │
│                                                                             │
│     "top_features": [                                                       │
│       {"feature": "sbytes", "value": 45000, "contribution": "+25%"},       │
│       {"feature": "rate", "value": 12000, "contribution": "+15%"},         │
│       {"feature": "sload", "value": 8500, "contribution": "+10%"}          │
│     ],                                                                      │
│                                                                             │
│     "simple_rule": "IF sbytes > 40000 AND rate > 10000 THEN DoS",          │
│                                                                             │
│     "human_explanation": "Traffic flagged due to high byte count            │
│                          (45,000) and high packet rate (12,000/s),         │
│                          indicating flooding behavior."                     │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Points:
- 🔍 **SHAP**: Shows exact contribution of each feature (mathematical)
- 📝 **LIME**: Creates simple rules (human-readable)
- 🎯 **Output**: Explanation + Confidence + Pattern
- 💡 **Use**: Passed to Phase 5 (Mitigation) for action decisions

---

### PHASE 5: MITIGATION (The Enforcer)

#### What is Mitigation?
Takes **action** based on the XAI explanation to:
1. Block the current attack
2. Prevent future similar attacks
3. Alert security team

#### Mitigation Decision Logic:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   XAI INPUT                                                                 │
│   ─────────                                                                 │
│   Prediction: DoS Attack                                                    │
│   Confidence: 95.2%                                                         │
│   Top Features: sbytes, rate, sload                                         │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│   DECISION ENGINE                                                           │
│   ───────────────                                                           │
│                                                                             │
│   Check confidence level:                                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                                                                 │      │
│   │   LOW (50-70%)           MEDIUM (70-90%)        HIGH (90%+)    │      │
│   │   ────────────           ──────────────         ───────────    │      │
│   │                                                                 │      │
│   │   • Log only             • Rate limit           • BLOCK NOW    │      │
│   │   • Monitor              • Temp block (5min)    • Update BPF   │      │
│   │   • Alert (low)          • Alert (medium)       • Block IP     │      │
│   │                                                                 │      │
│   │   "Not sure,             "Suspicious,           "Confirmed,    │      │
│   │    watch it"              slow it down"          stop it!"     │      │
│   │                                                                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Current confidence: 95.2% → HIGH CONFIDENCE                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Actions for HIGH Confidence (Our Example):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ACTION 1: BLOCK TRAFFIC                                                   │
│   ───────────────────────                                                   │
│   Immediately drop this connection                                          │
│   Status: ✅ EXECUTED                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ACTION 2: UPDATE BPF RULES (Auto-Signature Generation)                    │
│   ──────────────────────────────────────────────────────                    │
│                                                                             │
│   From XAI, we know:                                                        │
│   • sbytes = 45,000 (high contributor)                                      │
│   • rate = 12,000 (high contributor)                                        │
│   • proto = UDP                                                             │
│                                                                             │
│   Generate new BPF rule:                                                    │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                                                             │          │
│   │   NEW SIGNATURE RULE:                                       │          │
│   │                                                             │          │
│   │   IF  source_bytes > 40,000                                 │          │
│   │   AND packet_rate > 10,000                                  │          │
│   │   AND protocol == UDP                                       │          │
│   │   THEN → BLOCK                                              │          │
│   │                                                             │          │
│   │   Signature ID: SIG_20251222_001                           │          │
│   │   Created by: Auto-generation from ML+XAI                   │          │
│   │                                                             │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   Send to BPF Filter → BPF now knows this pattern!                         │
│   Status: ✅ EXECUTED                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ACTION 3: BLOCK SOURCE IP                                                 │
│   ─────────────────────────                                                 │
│   Add IP to blacklist: 192.168.1.100                                        │
│   Duration: Permanent (high confidence attack)                              │
│   Status: ✅ EXECUTED                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ACTION 4: SEND ALERT                                                      │
│   ────────────────────                                                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  🚨 SECURITY ALERT - HIGH PRIORITY                          │          │
│   │                                                             │          │
│   │  Time: 2025-12-22 14:30:05                                 │          │
│   │  Type: DoS Attack Detected                                  │          │
│   │  Source IP: 192.168.1.100                                  │          │
│   │  Confidence: 95.2%                                          │          │
│   │                                                             │          │
│   │  Why detected:                                              │          │
│   │  • High source bytes (45,000) - +25%                       │          │
│   │  • High packet rate (12,000/s) - +15%                      │          │
│   │  • High source load (8,500) - +10%                         │          │
│   │                                                             │          │
│   │  Actions taken:                                             │          │
│   │  ✓ Traffic blocked                                          │          │
│   │  ✓ BPF rule created                                         │          │
│   │  ✓ IP blacklisted                                           │          │
│   │                                                             │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   Status: ✅ SENT TO SECURITY TEAM                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ACTION 5: LOG ATTACK                                                      │
│   ────────────────────                                                      │
│   Store complete attack details for:                                        │
│   • Future analysis                                                         │
│   • Reporting                                                               │
│   • Compliance audit                                                        │
│   Status: ✅ LOGGED                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### The Learning Loop:

```
SYSTEM IMPROVEMENT OVER TIME:
═══════════════════════════════════════════════════════════════════════════════

Day 1: System starts
       BPF rules: 0
       All traffic → ML Model

Day 2: First attack detected
       ML detects → XAI explains → Mitigation creates Rule 1
       BPF rules: 1

Day 7: Multiple attacks detected
       BPF rules: 15
       Similar attacks now blocked instantly by BPF

Day 30: System has learned many patterns
        BPF rules: 100+
        80% of attacks blocked by BPF (instant)
        Only 20% need ML analysis (new attacks)

Result: System gets FASTER over time!
        Most attacks never reach the ML model

═══════════════════════════════════════════════════════════════════════════════
```

---

## 🔄 Complete Flow Example

Let's trace a **complete attack** through the system:

```
═══════════════════════════════════════════════════════════════════════════════
                         COMPLETE ATTACK TRACE
═══════════════════════════════════════════════════════════════════════════════

TIME: 14:30:05.001

STEP 1: Attack traffic arrives
────────────────────────────────────────────────────────────────────────────────
Attacker (192.168.1.100) sends UDP flood:
• 15,000 packets per second
• Each packet: 3,000 bytes
• Total: 45 Mbps of attack traffic

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.002 (1 millisecond later)

STEP 2: BPF Filter checks
────────────────────────────────────────────────────────────────────────────────
BPF checks signature database...
No matching rule found (this is a NEW attack pattern)
Decision: Pass to ML Model

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.003 (2 milliseconds later)

STEP 3: Feature Extraction
────────────────────────────────────────────────────────────────────────────────
Extract 10 features from traffic:
• rate = 15,000 pps
• sbytes = 45,000
• sload = 8,500
• dload = 200
• proto = 1 (UDP)
• dtcpb = 0
• stcpb = 0
• dmean = 0.002
• tcprtt = 0
• dur = 0.003

Scale features to standard range...

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.008 (7 milliseconds later)

STEP 4: ML Model Prediction
────────────────────────────────────────────────────────────────────────────────
XGBoost processes features...

Prediction: DoS Attack (1)
Confidence: 95.2%

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.012 (11 milliseconds later)

STEP 5: XAI Explanation
────────────────────────────────────────────────────────────────────────────────
SHAP analysis:
• sbytes (45,000) → +25% contribution
• rate (15,000) → +20% contribution
• sload (8,500) → +10% contribution

LIME rule:
"IF sbytes > 40000 AND rate > 12000 THEN DoS"

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.015 (14 milliseconds later)

STEP 6: Mitigation Actions
────────────────────────────────────────────────────────────────────────────────
Confidence: 95.2% → HIGH CONFIDENCE

✓ Action 1: Block traffic (DONE)
✓ Action 2: Generate BPF signature (DONE)
  New rule: IF sbytes > 40000 AND rate > 12000 AND proto = UDP → BLOCK
✓ Action 3: Blacklist IP 192.168.1.100 (DONE)
✓ Action 4: Send alert to security team (DONE)
✓ Action 5: Log attack details (DONE)

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:05.020 (19 milliseconds later)

RESULT: Attack blocked!
────────────────────────────────────────────────────────────────────────────────
Total detection time: 19 milliseconds
Server protected!

────────────────────────────────────────────────────────────────────────────────

TIME: 14:30:06.000 (NEXT SECOND - Same attack continues)

BPF Filter: "I know this pattern now!"
Action: BLOCK INSTANTLY (0.001 milliseconds)
ML Model: Never even sees it!

════════════════════════════════════════════════════════════════════════════════
```

---

## 📊 Summary Table

| Phase | Name | What It Does | Speed | Status |
|-------|------|--------------|-------|--------|
| **1** | BPF Filter | Blocks known attack patterns | Microseconds | 🔴 To Build |
| **2** | Feature Extraction | Converts traffic to 10 features | Milliseconds | ✅ Done |
| **3** | ML Model | Predicts Normal/DoS | Milliseconds | ✅ Done |
| **4** | XAI Explanation | Explains why detected | Milliseconds | 🔄 Ready |
| **5** | Mitigation | Takes action, updates BPF | Milliseconds | 🔴 To Build |

---

## 🎯 Key Takeaways

1. **Phase 1 (BPF) is the first line** - Fast but only knows what ML teaches it
2. **Phase 3 (ML) is the brain** - Smart but slower, handles new attacks
3. **Phase 4 (XAI) is the explainer** - Tells us WHY, builds trust
4. **Phase 5 (Mitigation) is the enforcer** - Takes action, teaches BPF
5. **The system learns** - Gets faster over time as Phase 1 (BPF) learns patterns from Phase 5

---

## 📚 Glossary

| Term | Meaning |
|------|---------|
| **BPF** | Berkeley Packet Filter - fast kernel-level filtering |
| **DoS** | Denial of Service - attack that floods systems |
| **DDoS** | Distributed DoS - attack from multiple sources |
| **XAI** | Explainable AI - makes ML decisions understandable |
| **SHAP** | Method to explain feature contributions |
| **LIME** | Method to create simple explanation rules |
| **Signature** | Pattern rule to identify known attacks |
| **Feature** | A measurable property of network traffic |

---

*Document Version: 1.0*  
*Last Updated: December 22, 2025*  
*For: Threat Hunters Team - CSIC 1.0 Challenge*
