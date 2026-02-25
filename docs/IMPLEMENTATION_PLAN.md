# Implementation Plan: XAI + Mitigation Framework

## Document Purpose
This document outlines the step-by-step implementation plan for Objectives 3 and 4. Each step requires user approval before proceeding.

---

## Current Status

| Objective | Status | Details |
|-----------|--------|---------|
| Objective 1: Dataset + Features | ✅ DONE | 10 features selected, data preprocessed |
| Objective 2: Model Training | ✅ DONE | XGBoost selected (90.57% F1 on benchmark) |
| Objective 3: XAI Integration | ⏳ NEXT | SHAP TreeExplainer |
| Objective 4: Mitigation Framework | ⏳ PENDING | Feature-to-Action mapping |

---

## How XAI Works (Clarification)

### Per-Record Explanation (NOT batch)

```
Input: 100 network traffic records

Step 1: Model predicts each record
        Record 1  → Normal (0.12 probability)
        Record 2  → Normal (0.08 probability)
        Record 3  → DoS    (0.94 probability)  ← FLAGGED
        Record 4  → Normal (0.15 probability)
        ...
        Record 45 → DoS    (0.89 probability)  ← FLAGGED
        ...

Step 2: SHAP explains ONLY flagged records
        Record 3:  "DoS because rate=+0.35, sload=+0.28"
        Record 45: "DoS because dur=+0.40, sbytes=+0.25"

Step 3: Mitigation generated ONLY for flagged records
        Record 3:  → Volumetric attack → Rate limiting commands
        Record 45: → Slowloris attack  → Timeout reduction commands
```

**Key Point:** We don't explain all records. Only records detected as DoS get explanations and mitigations.

---

## Attack Type Classification

### 4 Attack Types Based on Feature Patterns

| Attack Type | Key Features | Detection Rule |
|-------------|--------------|----------------|
| **Volumetric Flood** | rate ↑, sload ↑, sbytes ↑ | High volume traffic |
| **Protocol Exploit** | proto abnormal, rate normal | Protocol-based attack |
| **Slowloris** | dur ↑, rate ↓, sbytes ↑ | Slow, persistent connection |
| **Amplification** | dload >> sload | Response larger than request |

### Classification Logic (Simple Rules)

```python
def classify_attack(shap_values, feature_values):
    # Get top contributing features
    top_features = get_top_features(shap_values)

    if 'rate' in top_features and 'sload' in top_features:
        return "Volumetric Flood"
    elif 'proto' in top_features:
        return "Protocol Exploit"
    elif 'dur' in top_features and feature_values['rate'] < threshold:
        return "Slowloris"
    elif feature_values['dload'] > feature_values['sload'] * 2:
        return "Amplification"
    else:
        return "Generic DoS"
```

---

## Implementation Steps (With Approval Gates)

### STEP 1: SHAP Integration
**Goal:** Add SHAP TreeExplainer to explain XGBoost predictions

**What will be created:**
```
04_xai_integration/
├── shap_explainer.py      # SHAP TreeExplainer code
├── README.md              # Documentation
└── test_shap.py           # Test script
```

**Output Example:**
```python
# For a single DoS detection:
{
    "prediction": "DoS",
    "confidence": 0.942,
    "feature_contributions": {
        "rate": +0.35,      # High packet rate (main cause)
        "sload": +0.28,     # High source load
        "sbytes": +0.15,    # High bytes transferred
        "proto": +0.05,     # Protocol contribution
        ...
    }
}
```

**⏸️ APPROVAL GATE 1:** Review SHAP output before proceeding

---

### STEP 2: Attack Classification
**Goal:** Classify attack type based on SHAP feature contributions

**What will be created:**
```
05_mitigation_framework/
├── attack_classifier.py   # Attack type classification
└── README.md
```

**Output Example:**
```python
{
    "attack_type": "Volumetric Flood",
    "confidence": 0.942,
    "primary_indicators": ["rate", "sload", "sbytes"],
    "description": "High-volume traffic flood detected"
}
```

**⏸️ APPROVAL GATE 2:** Review attack classification before proceeding

---

### STEP 3: Severity Calculator
**Goal:** Determine severity level (Low/Medium/High/Critical)

**What will be created:**
```
05_mitigation_framework/
├── severity_calculator.py  # Severity scoring
```

**Severity Rules:**
```
LOW (60-75% confidence):
  - Monitor only
  - No immediate action

MEDIUM (75-90% confidence):
  - Apply rate limiting
  - Increase logging

HIGH (90-95% confidence):
  - Immediate throttling
  - Alert security team

CRITICAL (95%+ confidence + multiple features):
  - Auto-block recommended
  - Escalate to SOC
```

**⏸️ APPROVAL GATE 3:** Review severity logic before proceeding

---

### STEP 4: Mitigation Generator
**Goal:** Generate specific commands based on attack type

**What will be created:**
```
05_mitigation_framework/
├── mitigation_generator.py  # Command generation
├── mappings/
│   └── feature_to_action.json  # Mapping rules
```

**Feature-to-Action Mapping:**

| Feature | If Abnormal | Mitigation Command |
|---------|-------------|-------------------|
| rate | High packet rate | `iptables -m limit --limit 100/s` |
| sload | High bandwidth | `tc qdisc ... rate 1mbit` |
| sbytes | High bytes | `iptables -m connlimit --connlimit-above 10` |
| dur | Long duration | `timeout reduction in server config` |
| proto | Abnormal protocol | `iptables -p [proto] -j DROP` |

**⏸️ APPROVAL GATE 4:** Review mitigation commands before proceeding

---

### STEP 5: Complete Alert System
**Goal:** Combine all components into final output

**What will be created:**
```
05_mitigation_framework/
├── alert_generator.py      # Complete alert output
├── main.py                 # Entry point
└── examples/
    └── sample_output.json  # Example outputs
```

**Final Output Format:**
```
═══════════════════════════════════════════════════════════════
DETECTION ALERT
═══════════════════════════════════════════════════════════════
Timestamp:    2026-01-29 14:32:15
Source IP:    192.168.1.105
Verdict:      DoS Attack
Confidence:   94.2%

───────────────────────────────────────────────────────────────
XAI EXPLANATION
───────────────────────────────────────────────────────────────
Top Contributing Features:
  1. rate:   +0.35 (1,200 pkt/s vs normal 80 pkt/s)
  2. sload:  +0.28 (850 KB/s vs normal 50 KB/s)
  3. sbytes: +0.15 (High total bytes transferred)

Human Explanation:
  "This traffic is flagged as DoS because the source is sending
   packets at 15x the normal rate with 17x normal bandwidth."

───────────────────────────────────────────────────────────────
ATTACK CLASSIFICATION
───────────────────────────────────────────────────────────────
Type:     Volumetric Flood
Severity: HIGH

───────────────────────────────────────────────────────────────
RECOMMENDED MITIGATIONS
───────────────────────────────────────────────────────────────
Immediate Actions:
  □ iptables -A INPUT -s 192.168.1.105 -m limit --limit 100/s -j ACCEPT
  □ iptables -A INPUT -s 192.168.1.105 -j DROP

Alternative (Less Aggressive):
  □ tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit

Monitoring:
  □ tcpdump -i eth0 src 192.168.1.105 -w capture.pcap

Escalation: YES - Notify SOC team
═══════════════════════════════════════════════════════════════
```

**⏸️ APPROVAL GATE 5:** Review complete system before finalizing

---

## What We Are NOT Doing (To Keep It Simple)

| Not Doing | Reason |
|-----------|--------|
| ❌ LIME | SHAP is sufficient, LIME adds complexity |
| ❌ Runtime images | No image generation during detection |
| ❌ Dashboard/UI | Focus on core functionality |
| ❌ Real-time streaming | Batch processing is simpler |
| ❌ Database storage | File-based output is sufficient |

---

## Final Directory Structure

```
CTI_IDS/
├── 03_model_training/proper_training/    ✅ DONE
│   ├── models/xgboost/
│   ├── data/
│   ├── results/
│   └── images/ (6 images)
│
├── 04_xai_integration/                   STEP 1
│   ├── shap_explainer.py
│   ├── test_shap.py
│   └── README.md
│
├── 05_mitigation_framework/              STEPS 2-5
│   ├── attack_classifier.py
│   ├── severity_calculator.py
│   ├── mitigation_generator.py
│   ├── alert_generator.py
│   ├── main.py
│   ├── mappings/
│   │   └── feature_to_action.json
│   ├── examples/
│   │   └── sample_output.json
│   └── README.md
│
├── IMPLEMENTATION_PLAN.md                THIS FILE
└── XAI_MITIGATION_FRAMEWORK_PROPOSAL.md  Original proposal
```

---

## Approval Process

```
Step 1: SHAP Integration
        ↓
   [YOUR APPROVAL]
        ↓
Step 2: Attack Classification
        ↓
   [YOUR APPROVAL]
        ↓
Step 3: Severity Calculator
        ↓
   [YOUR APPROVAL]
        ↓
Step 4: Mitigation Generator
        ↓
   [YOUR APPROVAL]
        ↓
Step 5: Complete Alert System
        ↓
   [YOUR APPROVAL]
        ↓
      DONE ✅
```

---

## Next Action

**Ready to start Step 1 (SHAP Integration)?**

When you approve, I will:
1. Create `04_xai_integration/` directory
2. Write `shap_explainer.py` with TreeExplainer
3. Write test script
4. Show you the output
5. Wait for your approval before Step 2

---

*Document Created: 2026-01-29*
*Status: AWAITING APPROVAL TO START STEP 1*
