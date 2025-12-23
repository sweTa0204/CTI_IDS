Detection to Defense: An XAI-Powered DoS Detection System with Implementable Mitigation Protocols

Track: Systems and Software Security
Problem Statement: To Develop a machine learning based DDoS mitigation system.
1. Problem Understanding
Distributed Denial of Service attacks remain a persistent threat to organizations across banking, healthcare, and critical infrastructure. While numerous detection tools exist, most only tell administrators that an attack is happening without explaining why traffic was classified as malicious or what specific mitigation steps should be taken.
Current solutions treat detection and mitigation as separate concerns. Detection systems generate alerts while mitigation systems apply generic responses like rate limiting. The connection between "what was detected" and "what should be done" remains a manual process that consumes valuable time during active incidents and depends heavily on analyst expertise.
2. Proposed Solution and Technical Approach
We propose a four-phase system that closes the gap between detection and defense. The core innovation lies in treating explainability not as an afterthought, but as the bridge that connects machine learning classification to actionable mitigation protocols.
The central insight driving our approach: If we can explain precisely which traffic characteristics triggered a detection, we can automatically generate targeted filtering rules that address those specific characteristics.
Our system processes network traffic through four integrated phases:
Phase 1: BPF Filter (Kernel-Level Filtering)
Traffic first passes through a Berkeley Packet Filter operating at the kernel level. This filter maintains a database of known attack signatures derived from previous detections. Packets matching known patterns are blocked immediately, achieving microsecond-level response times. Unknown traffic passes to the machine learning model for analysis.
Phase 2: Machine Learning Model (Classification Engine)
Traffic that passes the initial filter undergoes classification using a machine learning model trained on network flow characteristics. We will evaluate multiple classification algorithms including ensemble methods, neural networks, and traditional classifiers to identify the model that offers the best balance of accuracy, speed, and interpretability for our use case. The selected model will analyze carefully engineered network features and output both a classification (normal or attack) and a confidence score.
Phase 3: Explainable AI Engine (The Differentiator)
This phase represents our primary technical contribution. When the model detects an attack, the XAI engine generates human-readable explanations using explainability techniques such as SHAP, LIME, or other suitable methods. Rather than simply flagging traffic as malicious, the system identifies which specific features drove the classification decision.
This explanation serves two purposes. It allows security analysts to verify the detection logic, building trust in the system. More importantly, it provides the exact parameters needed to construct targeted mitigation rules.
Phase 4: Mitigation Protocol Generation (Automated Response)
The final phase converts XAI explanations into actionable defense measures. Based on the identified attack characteristics and confidence level, the system:
•	Generates specific BPF rules targeting the identified traffic patterns
•	Updates the Phase 1 signature database for future instant blocking
•	Produces human-readable mitigation recommendations for security teams
•	Logs detailed attack information for forensic analysis
The four phases form a continuous feedback loop. Each detected attack improves the system's ability to block similar future attacks instantly, without requiring repeated ML classification.
3. System Architecture
The architecture follows a sequential processing pipeline with a feedback mechanism. Traffic flows from Phase 1 through Phase 4, with the mitigation phase feeding new signatures back to the initial filter.
Key architectural decisions:
•	Kernel-level initial filtering ensures minimal latency for known attack patterns.
•	Model inference runs in user space allowing for complex feature analysis without kernel modification.
•	Explanation generation occurs only for detected attacks minimizing computational overhead for normal traffic.
•	Signature updates are atomic preventing race conditions during rule deployment.
 
Figure 1: Four-Phase Architecture for Detection to Defense System

4. Expected Outcomes
Upon completion, our system will deliver:
Quantitative Outcomes:
•	Detection accuracy exceeding 95% on standard benchmark datasets.
•	Immediate blocking for known attack signatures via BPF.
•	Automated generation of mitigation rules.
•	Reduction in analyst investigation time through automated explanations.

Qualitative Outcomes:
•	Transparent decision-making that security teams can audit and verify.
•	Actionable intelligence rather than passive alerts.
•	Continuous improvement through the signature feedback loop.
•	Reduced dependency on manual rule creation by security analysts.
5. Benchmarking Against Existing Solutions
We evaluated our approach against three categories of existing solutions:
Aspect	Traditional IDS	Commercial DDoS Mitigation	Our Approach
Detection Method	Signature or anomaly-based	Rate limiting, behavioral	ML with explainability
Explanation Provided	Alert logs only	Traffic statistics	Feature-level attribution
Mitigation Approach	Manual rule creation	Generic rate limiting	Targeted automated rules
Learning Capability	Static signatures	Limited adaptation	Continuous signature generation
Response Time (Known)	Milliseconds	Milliseconds	Microseconds (BPF)

Key differentiators from existing approaches:
Traditional Intrusion Detection Systems like Snort and Suricata excel at signature matching but cannot adapt to new attack patterns without manual rule updates. They provide detection but leave mitigation entirely to administrators.
Commercial DDoS mitigation services from providers like Cloudflare and Akamai offer robust protection but operate as black boxes. Organizations cannot understand why specific traffic was blocked, making it difficult to tune policies or learn from incidents.
Academic research in ML-based detection has produced highly accurate models, but most work focuses on classification metrics rather than operational deployment. The question of how to translate a model prediction into a firewall rule remains largely unaddressed.
Our approach bridges these gaps by making explainability the mechanism through which detection connects to mitigation.
6. Differentiation and Unique Value Proposition
What makes our approach different is not the detection itself, but what happens after detection.
The security industry has largely solved the detection problem. Multiple commercial and open-source tools can identify DDoS attacks with reasonable accuracy. What remains unsolved is the translation problem: converting a detection event into an effective, targeted response.
Our unique contributions:
Explainability as Infrastructure: We treat XAI not as a compliance checkbox or debugging tool, but as core infrastructure that enables automated mitigation. The explanation is not generated for human consumption alone; it is parsed programmatically to construct filtering rules.
Closed-Loop Learning: Each attack detection improves future response capability. The signature database grows automatically, shifting workload from the computationally expensive ML model to the lightweight BPF filter over time.
Actionable Output: Security teams receive specific, implementable guidance rather than generic alerts. When our system reports an attack, it simultaneously provides the exact filtering parameters needed to block it.
Transparency by Design: Organizations can inspect why any traffic was blocked, enabling policy refinement and building trust in automated security decisions. This transparency addresses a common objection to ML-based security tools.
Development Phases (Aligned with CSIC 1.0 Timeline):
Stage 1 - Ideation (December 2025):
•	Problem understanding and solution design
•	System architecture planning
•	Initial research on ML models and XAI techniques
Stage 2-3 - Prototype Development (January - March 2026):
•	Dataset preparation using established benchmarks
•	Feature engineering and ML model evaluation
•	Initial prototype with basic detection and explanation capability
•	BPF filter implementation
Stage 4 - Mentorship and Enhancement (March - April 2026):
•	Incorporate mentor feedback for prototype refinement
•	XAI framework optimization
•	Mitigation protocol generator development
•	Integration testing across all four phases
Stage 5 - MVP Completion (April - May 2026):
•	Complete end-to-end working system
•	Performance optimization
•	Documentation and demonstration preparation
•	Final MVP submission

7. Target End-Use Cases:
Enterprise Security Operations Centers: Organizations with dedicated security teams can use the system to reduce investigation time and receive actionable mitigation guidance. The explanation capability helps analysts understand detections without manual log analysis.
Internet Service Providers: ISPs protecting customer infrastructure can deploy the system at network boundaries. The BPF-based filtering ensures minimal latency impact while explanations provide transparency for customer reporting.
Critical Infrastructure Protection: Government and critical sector organizations can leverage transparent decision-making to meet audit requirements while maintaining automated defenses.
Academic and Research Institutions: Universities and research labs can use the system for studying attack patterns and training security professionals with explainable detection outputs.
8. Technical Specifications
Machine Learning Component:
•	Algorithms to evaluate: Ensemble methods (Random Forest, XGBoost, Gradient Boosting), Neural Networks, SVM, and other classifiers
•	Training Dataset: Standard network intrusion benchmarks such as UNSW-NB15, CICIDS, or similar.
•	Features: Network flow characteristics including packet rates, byte counts, protocol information, and timing metrics.
•	Target Accuracy: Above 90% on benchmark datasets.
Explainability Component:
•	Techniques to explore: SHAP, LIME, or other suitable XAI methods
•	Output: Feature attribution explaining which traffic characteristics triggered detection
•	Goal: Human-readable explanations that can be translated to mitigation rules
Filtering Component:
•	Technology: Berkeley Packet Filter (BPF/eBPF)
•	Deployment: Linux kernel integration
•	Goal: Microsecond-level packet filtering for known attack signatures.
9. Team Capability
Our team combines expertise in machine learning, network security, and systems programming. We have completed the foundational work including model training, XAI integration, and architectural design. Our commitment is to progress from the current prototype through MVP development within the challenge timeline.
We approach this challenge not as a theoretical exercise but as a practical problem that security teams face daily. Our goal is to build something that organizations can actually deploy to improve their security posture.
10. Conclusion
The gap between detection and defense represents one of the most significant practical challenges in DDoS mitigation. Our system addresses this gap by placing explainability at the center of the architecture, using human-interpretable analysis as the mechanism that connects machine learning classification to automated mitigation protocols.
We are not building another detection tool. We are building the bridge between knowing an attack is happening and knowing exactly what to do about it.


