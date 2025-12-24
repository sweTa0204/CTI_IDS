# Systems & Software Security - Problem Statement

## Cluster Information
- **Cluster Name:** Systems & Software Security
- **Problem Statement Title:** DDoS Mitigation System

---

## 📋 Description

Develop a machine learning based DDoS mitigation system capable of distinguishing legitimate traffic spikes from malicious floods, providing adaptive and automated defenses against hyper volumetric attacks.

---

## 📦 Exact Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Traffic Shaping Proxy** | Implementation with fast BPF filters and auto signature generation |
| 2 | **ML-based Anomaly Detection Module** | Differentiate real traffic surges from attacks |
| 3 | **Simulation Framework** | Test floods and chart mitigation latency vs. packet-rate peaks |
| 4 | **Comparative Benchmarking Report** | Comparison vs. traditional appliance-based solutions |

---

## 🎯 Milestones & Evolution Parameters

### Phase 1: Baseline Traffic Anomaly Detection
- Implement baseline traffic anomaly detection
- Establish foundational detection capabilities

### Phase 2: ML-based Classification
- Add ML-based classification for attack vs. legitimate surges
- Train models to distinguish between normal traffic spikes and malicious floods

### Phase 3: Traffic Shaping Proxy Deployment
- Deploy traffic shaping proxy
- Validate on large, simulated floods
- Performance testing and optimization

---

## ℹ️ Additional Information & Requirements

| Requirement | Details |
|-------------|---------|
| **Focus Area** | Real-time mitigation capabilities |
| **Model Preference** | Lightweight ML models for high-speed packet filtering |
| **Benchmarking Target** | Attack scales relevant to India's internet exchanges |

---

## 🔗 Alignment with CTI_IDS Project

Our existing **CTI_IDS (Cyber Threat Intelligence - Intrusion Detection System)** project provides a strong foundation for this challenge:

### Current Project Assets
- ✅ ML models trained for DoS detection
- ✅ Feature engineering pipeline complete
- ✅ Multiple model comparison framework
- ✅ External benchmarking framework
- ✅ XAI (Explainable AI) implementation planned

### Gap Analysis - Components to Develop

| Component | Status | Priority |
|-----------|--------|----------|
| Traffic Shaping Proxy with BPF filters | 🔴 Not Started | High |
| Auto Signature Generation | 🔴 Not Started | High |
| Real-time Processing Pipeline | 🔴 Not Started | High |
| Simulation Framework | 🔴 Not Started | Medium |
| Latency vs Packet-rate Benchmarking | 🔴 Not Started | Medium |
| Comparative Report vs Appliances | 🟡 Partial (have benchmarking) | Medium |

---

## 📁 Project Structure

```
CSIC 1.0/
├── CSIC_1.0_SUMMARY.md
├── documents/
│   ├── CSIC Brocher.pdf
│   ├── Innovation-Challenge-Rulebook-v1.pdf
│   ├── Systems_Software_Security.pdf
│   └── WhatsApp Images...
└── Systems_and_Software_Security/
    ├── Problem_Statement.md (this file)
    └── [Additional project files to be added]
```

---

*Document Created: December 22, 2025*
*Challenge: CSIC 1.0 - Cyber Security Innovation Challenge*
*Track: Systems & Software Security*
