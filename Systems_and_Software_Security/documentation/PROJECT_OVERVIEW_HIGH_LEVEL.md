# 🔄 Project Overview: XAI-Powered DDoS Mitigation System

## 📖 High-Level Overview

This document provides a simplified, high-level overview of our DDoS Mitigation System. It focuses on the 4 main phases of our project.

**Team Name:** Threat Hunters  
**Members:** Shweta Sharma, Devika, Akash  
**Challenge:** CSIC 1.0 - Systems & Software Security  
**Document Created:** December 22, 2025

---

## 🎯 What Does Our System Do?

Our system protects networks from **DDoS (Distributed Denial of Service) attacks** by:
1. **Filtering** known attack patterns instantly
2. **Detecting** new attacks using Machine Learning
3. **Explaining** why traffic was detected as malicious
4. **Blocking** attacks and learning for future protection

---

## 🏗️ System Architecture (4 Phases)

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
                    │    PHASE 2: ML MODEL                │
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
                    │    PHASE 3: XAI EXPLANATION         │
                    │    (The Explainer)                  │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       │ Why it's DoS + Confidence
                                       ▼
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │    PHASE 4: MITIGATION              │
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

## 📋 The 4 Phases Explained

---

### PHASE 1: BPF FILTER (The Gatekeeper)

#### What is it?
**BPF (Berkeley Packet Filter)** is a super-fast filtering system that blocks known attack patterns instantly.

#### What does it do?
- Acts as the **first line of defense**
- Checks incoming traffic against **known attack patterns (signatures)**
- Blocks known attacks **instantly** (microseconds)
- Passes unknown traffic to the ML model for analysis

#### How it works:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INCOMING TRAFFIC                                              │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────┐              │
│   │         SIGNATURE DATABASE                   │              │
│   │                                             │              │
│   │   Rule 1: IF sbytes > 45000 AND rate > 12000 → BLOCK      │
│   │   Rule 2: IF source_ip = 10.0.0.50 → BLOCK                │
│   │   Rule 3: IF packet_size = 64 AND rate > 50000 → BLOCK    │
│   │                                             │              │
│   └─────────────────────────────────────────────┘              │
│        │                                                        │
│        ▼                                                        │
│   MATCH FOUND? ──YES──→ 🚫 BLOCK (Instant!)                    │
│        │                                                        │
│       NO                                                        │
│        │                                                        │
│        ▼                                                        │
│   Pass to Phase 2 (ML Model)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Important: When System First Starts

```
SYSTEM START:
═══════════════════════════════════════════════════════

BPF Signature Database = EMPTY (no rules yet!)

↓

ALL traffic passes to ML Model for analysis

↓

As ML detects attacks → Phase 4 creates rules → BPF learns!

↓

Over time: Most attacks blocked by BPF instantly!

═══════════════════════════════════════════════════════
```

#### Key Points:
| Aspect | Detail |
|--------|--------|
| ⚡ Speed | Microseconds (0.001 ms) |
| 📍 Location | Linux kernel |
| 📚 Rules | Pattern-based signatures |
| 🔄 Updates | Receives new rules from Phase 4 |

---

### PHASE 2: ML MODEL (The Brain)

#### What is it?
A **trained Machine Learning model** (XGBoost) that predicts whether traffic is **Normal** or a **DoS Attack**.

#### What does it do?
- Analyzes traffic that BPF couldn't identify
- Uses 10 key features to make predictions
- Achieves **95.54% accuracy**
- Detects NEW attacks that BPF hasn't seen before

#### Our Model:

| Detail | Value |
|--------|-------|
| **Algorithm** | XGBoost |
| **Accuracy** | 95.54% |
| **Dataset** | UNSW-NB15 |
| **Features** | 10 key network features |

#### Model Comparison (Why XGBoost?):

| Model | Accuracy | Verdict |
|-------|----------|---------|
| **XGBoost** | **95.54%** | 🏆 Champion |
| Random Forest | 95.29% | Strong |
| MLP Neural Network | 92.48% | Good |
| SVM | 90.04% | Moderate |
| Logistic Regression | 78.18% | Baseline |

#### How it works:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   TRAFFIC FROM PHASE 1 (Unknown traffic)                       │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────┐              │
│   │              XGBoost Model                   │              │
│   │                                             │              │
│   │   • Analyzes 10 network features            │              │
│   │   • 100+ decision trees vote                │              │
│   │   • Returns: DoS or Normal                  │              │
│   │                                             │              │
│   └─────────────────────────────────────────────┘              │
│        │                                                        │
│        ▼                                                        │
│   PREDICTION:                                                   │
│        │                                                        │
│        ├──→ NORMAL (0) → ✅ ALLOW traffic                      │
│        │                                                        │
│        └──→ DoS ATTACK (1) → Continue to Phase 3               │
│             Confidence: 95.2%                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Points:
| Aspect | Detail |
|--------|--------|
| 🧠 Type | XGBoost (Gradient Boosting) |
| 📈 Accuracy | 95.54% |
| ⚡ Speed | Milliseconds per prediction |
| 📊 Output | Prediction + Confidence % |

---

### PHASE 3: XAI EXPLANATION (The Explainer)

#### What is it?
**XAI (Explainable AI)** tells us **WHY** the model detected an attack.

#### What does it do?
- Explains which features caused the detection
- Shows contribution of each feature
- Creates human-readable explanations
- Builds trust in the system

#### Why do we need XAI?

| Reason | Benefit |
|--------|---------|
| **Trust** | Security teams can verify detections |
| **Debugging** | Find which features misled the model |
| **Compliance** | Some regulations require explainable AI |
| **Learning** | Understand attack patterns |

#### Two XAI Methods:

##### SHAP (Feature Contributions)
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   SHAP EXPLANATION                                              │
│                                                                 │
│   Prediction: DoS Attack (95% confidence)                       │
│                                                                 │
│   Feature Contributions:                                        │
│   ┌───────────────────────────────────────────────────────┐    │
│   │   sbytes (45,000)  ████████████████████  +25% → DoS   │    │
│   │   rate (12,000)    ██████████████       +15% → DoS   │    │
│   │   sload (8,500)    ██████████           +10% → DoS   │    │
│   │   dmean (0.002)    ████                  -5% → Normal │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                 │
│   "Traffic detected as DoS because of high bytes and rate"      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

##### LIME (Simple Rules)
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   LIME EXPLANATION                                              │
│                                                                 │
│   Simple Rule Generated:                                        │
│                                                                 │
│   IF sbytes > 40,000 AND rate > 10,000                         │
│   THEN → DoS Attack (85% likely)                               │
│                                                                 │
│   Human-readable and easy to understand!                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Points:
| Aspect | Detail |
|--------|--------|
| 🔍 SHAP | Mathematical feature contributions |
| 📝 LIME | Simple IF-THEN rules |
| 🎯 Output | Explanation + Pattern |
| 💡 Use | Passed to Phase 4 for action |

---

### PHASE 4: MITIGATION (The Enforcer)

#### What is it?
Takes **action** based on the detection and explanation.

#### What does it do?
- Blocks the current attack
- Creates new BPF rules for future protection
- Alerts security team
- Logs attack details

#### Mitigation Actions (Based on Confidence):

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   CONFIDENCE LEVEL DETERMINES ACTIONS:                          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │   LOW (50-70%)       MEDIUM (70-90%)      HIGH (90%+)  │  │
│   │   ────────────       ──────────────       ───────────  │  │
│   │                                                         │  │
│   │   • Log only         • Rate limit         • BLOCK NOW  │  │
│   │   • Monitor          • Temp block         • Update BPF │  │
│   │   • Alert (low)      • Alert (medium)     • Block IP   │  │
│   │                                                         │  │
│   │   "Watch it"         "Slow it down"       "Stop it!"   │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### HIGH Confidence Actions (Example: 95%):

| Action | Description |
|--------|-------------|
| 🚫 **Block Traffic** | Drop the connection immediately |
| 📝 **Update BPF** | Create new signature rule for Phase 1 |
| 🔒 **Block IP** | Add attacker IP to blacklist |
| 📢 **Alert Team** | Notify security with XAI explanation |
| 📋 **Log Attack** | Store details for analysis |

#### The Learning Loop:

```
THE SYSTEM GETS SMARTER OVER TIME:
═══════════════════════════════════════════════════════════════════

Day 1:  BPF has 0 rules → All traffic goes to ML
Day 7:  BPF has 15 rules → Known attacks blocked instantly
Day 30: BPF has 100+ rules → 80% attacks blocked by BPF

Result: System becomes FASTER as it learns more patterns!

═══════════════════════════════════════════════════════════════════
```

#### Key Points:
| Aspect | Detail |
|--------|--------|
| 🎯 Actions | Based on confidence level |
| 🔄 Learning | Updates Phase 1 (BPF) with new rules |
| 📢 Alerts | Includes XAI explanation |
| 📋 Logging | Complete attack records |

---

## 🔄 The Complete Cycle

```
           ┌────────────────────────────────────────────────┐
           │                                                │
           ▼                                                │
    ┌─────────────┐                                         │
    │  PHASE 1    │                                         │
    │ BPF Filter  │ ──→ Known Attack ──→ 🚫 BLOCK          │
    └──────┬──────┘                                         │
           │ Unknown                                        │
           ▼                                                │
    ┌─────────────┐                                         │
    │  PHASE 2    │                                         │
    │  ML Model   │ ──→ Normal ──→ ✅ ALLOW                │
    └──────┬──────┘                                         │
           │ DoS Detected                                   │
           ▼                                                │
    ┌─────────────┐                                         │
    │  PHASE 3    │                                         │
    │ XAI Explain │                                         │
    └──────┬──────┘                                         │
           │ Explanation                                    │
           ▼                                                │
    ┌─────────────┐                                         │
    │  PHASE 4    │                                         │
    │ Mitigation  │ ──→ 🚫 BLOCK + 📢 ALERT                │
    └──────┬──────┘                                         │
           │                                                │
           │ New BPF Rule                                   │
           └────────────────────────────────────────────────┘
```

---

## 📊 Summary Table

| Phase | Name | What It Does | Speed | Status |
|-------|------|--------------|-------|--------|
| **1** | BPF Filter | Blocks known attack patterns | Microseconds | 🔴 To Build |
| **2** | ML Model | Predicts Normal/DoS | Milliseconds | ✅ Done |
| **3** | XAI Explanation | Explains why detected | Milliseconds | 🔄 Ready |
| **4** | Mitigation | Takes action, updates BPF | Milliseconds | 🔴 To Build |

---

## 🎯 Key Takeaways

1. **Phase 1 (BPF)** - Fast gatekeeper, blocks known attacks instantly
2. **Phase 2 (ML)** - Smart brain, detects new attacks with 95.54% accuracy
3. **Phase 3 (XAI)** - Explainer, tells us WHY it's an attack
4. **Phase 4 (Mitigation)** - Enforcer, takes action and teaches Phase 1

**The cycle continues:** Phase 4 teaches Phase 1 → System gets faster over time!

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
| **XGBoost** | Our ML algorithm with 95.54% accuracy |

---

*Document Version: 1.0*  
*Last Updated: December 22, 2025*  
*For: Threat Hunters Team - CSIC 1.0 Challenge*
