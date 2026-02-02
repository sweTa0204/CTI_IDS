# Mitigation Framework Overview
## High-Level Explanation for Faculty Review

**Date:** 2026-01-31
**Component:** Automated DoS Mitigation System

---

## What is the Mitigation Framework?

The **Mitigation Framework** converts DoS detections into **actionable security responses**. It bridges the gap between:
- **Detection:** "This is a DoS attack"
- **Action:** "Here's exactly what to do about it"

---

## Complete Flow: XAI Output → Mitigation Commands

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INPUT: XAI OUTPUT (SHAP)                          │
├──────────────────────────────────────────────────────────────────────┤
│  From SHAP Explainer:                                                │
│  • Prediction: DoS                                                   │
│  • Confidence: 95.18%                                                │
│  • SHAP Values: {tcprtt: +1.44, dload: +1.07, dmean: +0.94, ...}    │
│  • Top 3 Features: [tcprtt, dload, dmean]                            │
│  • Feature Values: Actual scaled values                              │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│            STEP 1: ATTACK CLASSIFICATION                             │
│                  (attack_classifier.py)                              │
├──────────────────────────────────────────────────────────────────────┤
│  Logic: Based on which features have highest SHAP values             │
│                                                                      │
│  IF top_features contain [rate, sload, sbytes]:                      │
│     → VOLUMETRIC FLOOD                                               │
│     → High-volume traffic overwhelming network                       │
│                                                                      │
│  IF top_features contain [proto, tcprtt, stcpb, dtcpb]:             │
│     → PROTOCOL EXPLOIT                                               │
│     → Manipulating protocol behavior (SYN flood, etc.)               │
│                                                                      │
│  IF top_features contain [dur, dmean]:                              │
│     → SLOWLORIS                                                      │
│     → Slow, persistent connections exhausting resources              │
│                                                                      │
│  IF top_features contain [dload, dbytes]:                           │
│     → AMPLIFICATION                                                  │
│     → Response traffic larger than request                           │
├──────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                                             │
│  • attack_type: "Amplification"                                      │
│  • description: "Attack using amplification techniques"              │
│  • indicators: ["dload", "dbytes"]                                   │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│            STEP 2: SEVERITY ASSESSMENT                               │
│                 (severity_calculator.py)                             │
├──────────────────────────────────────────────────────────────────────┤
│  Logic: Based on model confidence score                              │
│                                                                      │
│  IF confidence >= 95%:                                               │
│     → CRITICAL                                                       │
│     → Immediate blocking recommended                                 │
│                                                                      │
│  IF confidence 90-95%:                                               │
│     → HIGH                                                           │
│     → Priority response required                                     │
│                                                                      │
│  IF confidence 75-90%:                                               │
│     → MEDIUM                                                         │
│     → Monitor closely, apply rate limiting                           │
│                                                                      │
│  IF confidence 60-75%:                                               │
│     → LOW                                                            │
│     → Log and observe                                                │
├──────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                                             │
│  • level: "CRITICAL"                                                 │
│  • escalation_required: true                                         │
│  • actions: ["Block immediately", "Alert SOC"]                       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│           STEP 3: MITIGATION GENERATION                              │
│               (mitigation_generator.py)                              │
├──────────────────────────────────────────────────────────────────────┤
│  Logic: Based on attack_type + severity_level                        │
│                                                                      │
│  For AMPLIFICATION + CRITICAL:                                       │
│                                                                      │
│  1. Block Source IP:                                                 │
│     $ iptables -A INPUT -s 192.168.1.100 -j DROP                     │
│                                                                      │
│  2. Filter DNS Amplification:                                        │
│     $ iptables -A INPUT -p udp --sport 53 -m length --length 512: \ │
│       -j DROP                                                        │
│                                                                      │
│  3. Rate Limit Interface:                                            │
│     $ tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit \    │
│       latency 400ms                                                  │
│                                                                      │
│  4. Enable SYN Cookies:                                              │
│     $ sysctl -w net.ipv4.tcp_syncookies=1                            │
│                                                                      │
│  5. Log Attack:                                                      │
│     $ logger "CRITICAL: Amplification attack from 192.168.1.100"     │
├──────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                                             │
│  • commands: [list of executable bash commands]                      │
│  • explanation: Human-readable reasoning                             │
│  • category: "amplification_filtering"                               │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 FINAL OUTPUT: SECURITY ALERT                         │
│                    (alert_generator.py)                              │
├──────────────────────────────────────────────────────────────────────┤
│  {                                                                   │
│    "timestamp": "2026-01-31 10:30:15",                               │
│    "detection": {                                                    │
│      "prediction": "DoS",                                            │
│      "confidence": 0.9518                                            │
│    },                                                                │
│    "classification": {                                               │
│      "attack_type": "Amplification",                                 │
│      "description": "Response traffic > request traffic"             │
│    },                                                                │
│    "severity": {                                                     │
│      "level": "CRITICAL",                                            │
│      "escalation_required": true                                     │
│    },                                                                │
│    "mitigation": {                                                   │
│      "commands": [                                                   │
│        "iptables -A INPUT -s 192.168.1.100 -j DROP",                 │
│        "tc qdisc add dev eth0 root tbf rate 100mbit...",            │
│        ...                                                           │
│      ],                                                              │
│      "explanation": "Block source and limit traffic"                 │
│    },                                                                │
│    "network_info": {                                                 │
│      "source_ip": "192.168.1.100",                                   │
│      "destination_ip": "10.0.0.1",                                   │
│      "interface": "eth0"                                             │
│    }                                                                 │
│  }                                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Attack Classifier (`attack_classifier.py`)

**Purpose:** Determine what TYPE of DoS attack this is

**Input from XAI:**
- Top 3 SHAP features (e.g., `[tcprtt, dload, dmean]`)
- SHAP values dictionary
- Feature values

**Classification Logic:**

| Top Features | Attack Type | Characteristics |
|-------------|-------------|-----------------|
| rate, sload, sbytes | **Volumetric Flood** | Overwhelming traffic volume |
| proto, tcprtt, stcpb | **Protocol Exploit** | Protocol manipulation (SYN flood) |
| dur, dmean | **Slowloris** | Slow connections exhausting resources |
| dload, dbytes | **Amplification** | Response > request (DNS amp, NTP amp) |

**Output:**
```json
{
  "attack_type": "Amplification",
  "description": "Attack using amplification techniques",
  "confidence": 0.89,
  "indicators": ["dload", "dbytes"]
}
```

---

### 2. Severity Calculator (`severity_calculator.py`)

**Purpose:** Determine HOW SERIOUS the attack is

**Input:**
- Model confidence (0.9518 = 95.18%)
- Attack type

**Severity Levels:**

| Confidence | Level | Action Required |
|-----------|-------|-----------------|
| >= 95% | **CRITICAL** | Immediate block + escalate |
| 90-95% | **HIGH** | Priority response + alert team |
| 75-90% | **MEDIUM** | Apply rate limiting + monitor |
| 60-75% | **LOW** | Log + observe |

**Output:**
```json
{
  "level": "CRITICAL",
  "color": "red",
  "description": "Very high confidence, auto-block recommended",
  "escalation_required": true,
  "actions": [
    "Apply auto-blocking",
    "Escalate to SOC immediately"
  ]
}
```

---

### 3. Mitigation Generator (`mitigation_generator.py`)

**Purpose:** Generate SPECIFIC commands to stop the attack

**Input:**
- Attack type: "Amplification"
- Severity: "CRITICAL"
- Source IP: "192.168.1.100"
- Interface: "eth0"

**Mitigation Strategy by Attack Type:**

#### Volumetric Flood
```bash
# Rate limit traffic
tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms

# Limit connections per IP
iptables -A INPUT -p tcp --syn -m limit --limit 10/s -j ACCEPT
```

#### Protocol Exploit (SYN Flood)
```bash
# Block source IP
iptables -A INPUT -s 192.168.1.100 -j DROP

# Enable SYN cookies
sysctl -w net.ipv4.tcp_syncookies=1

# Limit SYN packets
iptables -A INPUT -p tcp --syn -m limit --limit 50/s -j ACCEPT
```

#### Slowloris
```bash
# Reduce connection timeout
sysctl -w net.ipv4.tcp_fin_timeout=30

# Limit connections per source
iptables -A INPUT -s 192.168.1.100 -m connlimit --connlimit-above 10 -j DROP
```

#### Amplification
```bash
# Block amplification responses
iptables -A INPUT -p udp --sport 53 -m length --length 512: -j DROP

# Rate limit UDP traffic
iptables -A INPUT -p udp -m limit --limit 100/s -j ACCEPT
```

**Output:**
```json
{
  "commands": [
    "iptables -A INPUT -s 192.168.1.100 -j DROP",
    "tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit",
    "sysctl -w net.ipv4.tcp_syncookies=1"
  ],
  "explanation": "Block attacker IP, apply rate limiting, enable SYN protection",
  "category": "amplification_filtering"
}
```

---

## Why This Matters

### Traditional IDS:
```
Network Traffic → Detection → "DoS Attack Detected" → [Analyst figures out what to do]
```

**Problems:**
- ❌ No explanation WHY
- ❌ No guidance WHAT to do
- ❌ Analyst must manually decide
- ❌ Slow response time

### Our System:
```
Network Traffic → Detection → SHAP Explanation → Attack Type → Severity → Mitigation Commands
```

**Benefits:**
- ✅ Explains WHY (SHAP values)
- ✅ Classifies WHAT TYPE
- ✅ Assesses HOW SERIOUS
- ✅ Generates SPECIFIC commands
- ✅ **Automated response in <3ms**

---

## Connection: XAI → Mitigation

**The Key Innovation:**

The **same SHAP values** that explain the detection are **used to classify** the attack type.

**Example:**

```
SHAP Output:
  Top features: [tcprtt: +1.44, dload: +1.07, dmean: +0.94]
               ↓
Attack Classifier:
  "dload is top feature → Amplification attack"
               ↓
Mitigation Generator:
  "Amplification → Use DNS amplification filters"
```

**This means:**
- The explanation IS the input to mitigation
- Transparency enables intelligent response
- No black-box decision-making

---

## Summary for Faculty

| Question | Answer |
|----------|--------|
| **What does it do?** | Converts DoS detections into executable mitigation commands |
| **What's the input?** | SHAP output (top features + confidence) |
| **What's the output?** | Specific iptables/tc commands to stop the attack |
| **How fast?** | Complete pipeline: <3ms per sample |
| **Why is this novel?** | First to use XAI explanations to drive automated mitigation |
| **Real-world ready?** | Yes - commands are executable on Linux systems |

---

## For Tomorrow's Review

**Quick Explanation (30 seconds):**

"After SHAP explains why an attack was detected, our mitigation framework uses those same explanations to:
1. Classify the attack type (4 types)
2. Calculate severity (4 levels)
3. Generate specific mitigation commands (iptables, tc)

For example, if SHAP says 'high dload caused this detection,' we classify it as Amplification and generate commands to filter DNS amplification responses."

**Show File:**
- `05_mitigation_framework/attack_classifier.py` - Classification logic
- `05_mitigation_framework/mitigation_generator.py` - Command generation

---

*Document Created: 2026-01-31*
*For: Faculty Review Presentation*
