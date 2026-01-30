This document summarizes the key findings and evolution of the project based on the "Review 0" and "Review 1" documents.

### Key Takeaways from Previous Reviews

1.  **Project Title Change**: The title evolved from focusing on "HTTP DoS Prevention" to the broader "DoS Detection," indicating a scope adjustment early in the project.

2.  **Initial ADASYN Plan**: The project initially planned to use ADASYN to balance the UNSW-NB15 dataset. This was a core part of the methodology in "Review 0".

3.  **Critical Rejection of ADASYN**: A major outcome of the work done for "Review 1" was the formal rejection of ADASYN. A custom 5-tier validation framework was designed, which proved that ADASYN was degrading the quality of the dataset. This is a critical research finding and a key decision point.

4.  **Shift to Natural Balancing**: Following the rejection of ADASYN, the methodology shifted to creating a balanced dataset by extracting and sampling from the original raw data, resulting in a quality dataset of 8,178 samples.

5.  **Feature Engineering Success**: A 6-phase feature engineering pipeline was successfully implemented, reducing the feature count from 45 to a more efficient set of 10.

6.  **Clarification of Phased Approach**: The project is divided into phases.
    *   **Phase 1 (Current Scope)**: Focuses strictly on **Objective 1 (Data Processing)** and **Objective 2 (Model Training)**. This includes the data extraction, feature engineering, and benchmarking of models like Random Forest, XGBoost, and Logistic Regression.
    *   **Phase 2 (Future Work)**: Will cover **Objective 3 (XAI Integration)** and **Objective 4 (Mitigation Engine)**. All work related to SHAP, LIME, and the rule-based engine is deferred to this next phase.

7.  **Literature Review Insights**: The initial literature review identified gaps in existing research, including a lack of real-time detection focus, failure to address class imbalance properly, and abstract XAI outputs not tied to actionable responses. This project, even in Phase 1, addresses the class imbalance and feature selection gaps noted in the literature.
