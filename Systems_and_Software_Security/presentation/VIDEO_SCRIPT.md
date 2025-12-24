# Video Script: Detection to Defense
## XAI-Powered DDoS Mitigation System

**Team:** Threat Hunters  
**Total Duration:** 3 minutes (60 seconds per speaker)

---

## PART 1: The Problem and Vision
**Speaker 1 | Duration: 60 seconds**

---

Hello, we are Team Threat Hunters, and today we present our solution for the Systems and Software Security challenge.

Here is the reality of DDoS attacks today. Organizations have dozens of detection tools, but when an attack hits, security teams still scramble to figure out what is actually happening and what to do about it.

Think about it. Your detection system says "attack detected." Great. But why was it detected? Which traffic patterns triggered the alert? And most importantly, what exact steps should you take right now to stop it?

Current solutions leave a dangerous gap. Detection happens here. Mitigation happens there. And in between? Manual investigation, guesswork, and precious time lost.

Our project, Detection to Defense, closes this gap. We are building a system where detection does not just raise an alarm. It explains exactly why traffic is malicious and automatically generates the specific rules needed to block it.

The key insight is simple. If we can explain what triggered a detection, we can automatically create targeted defenses against it.

---

## PART 2: How It Works
**Speaker 2 | Duration: 60 seconds**

---

Let me walk you through our four-phase architecture.

Phase One is the BPF Filter. This operates at the kernel level and blocks known attack patterns instantly, in microseconds. Think of it as the first line of defense that handles everything we have seen before.

Phase Two is the Machine Learning Model. Traffic that gets past the filter is analyzed by our ML classifier. We evaluate multiple algorithms to find the best balance of accuracy and speed for real-time detection.

Phase Three is where we differentiate ourselves. This is our Explainable AI Engine. When an attack is detected, we do not just flag it. We explain exactly which traffic features caused the detection. High packet rate? Unusual byte patterns? The system tells you precisely what triggered the alarm.

Phase Four is Mitigation. Here is where explanation becomes action. The system takes those XAI insights and automatically generates new BPF rules. These rules feed back to Phase One, so the next time this attack pattern appears, it gets blocked instantly.

The result? A system that learns and improves with every attack it encounters.

---

## PART 3: Impact and Roadmap
**Speaker 3 | Duration: 60 seconds**

---

So what makes our approach unique?

First, we treat explainability as infrastructure, not just a feature. The explanation is not just for humans to read. It is parsed programmatically to build actual filtering rules.

Second, our system creates a closed loop. Every detected attack makes the system faster and smarter. Over time, more attacks get blocked at the BPF layer in microseconds, without needing ML inference and saving time.

Third, we provide actionable output. Security teams do not get vague alerts. They get specific parameters and ready-to-deploy mitigation rules.

Our target users include enterprise security operations centers, internet service providers, and critical infrastructure protection agencies. Basically, anyone who needs to defend against DDoS attacks and actually understand what is happening.

For our development roadmap, we will complete prototype development by March 2026 and deliver a working MVP by May 2026.

We are not building another detection tool. We are building the bridge between knowing an attack is happening and knowing exactly what to do about it.

Thank you.

---

## Production Notes

**Visual Suggestions:**
- Part 1: Show problem statistics, current solution gaps
- Part 2: Display the four-phase architecture diagram
- Part 3: Show roadmap timeline and use case examples

**Tone:** Confident, clear, professional but not robotic

**Pace:** Approximately 150 words per minute for natural delivery

**Total Word Count:** ~540 words (180 per speaker)
