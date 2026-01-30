# A. APPENDIX

This appendix consolidates supporting elements that complement the main report. It includes technical design details, data source references, visual artifacts, and interpretive aids that enhance the understanding of the research and its implementation.

### Technical Details

The project is implemented as a modular, script-based machine learning pipeline primarily using Python. The architecture is designed to ensure a clear separation of concerns for each stage of the research process. Key technologies and libraries include:
-   **Data Manipulation and Analysis**: Pandas, NumPy
-   **Machine Learning and Modeling**: Scikit-learn, XGBoost
-   **Visualization**: Matplotlib, Seaborn

The project workflow is organized into distinct phases, from data preparation and feature engineering to model training and validation, ensuring a reproducible and scalable research process.

### Raw Data and Tables

The foundational data for this research is the **UNSW-NB15 dataset**, a comprehensive collection of real and simulated network traffic. The raw data was processed to create a balanced working dataset (`working_dataset.csv`) and subsequently a cleaned, feature-engineered dataset (`cleaned_dataset.csv`) used for model training. The performance metrics for each of the five models, as detailed in the "Results and Discussion" chapter, are derived from this final dataset.

### Design Artifacts

Visual artifacts were created to support the analysis and interpretation of the model results. These include:
-   **Model F1-Score Comparison Chart**: A bar chart (`model_f1_score_comparison.png`) that provides a clear visual comparison of the primary performance metric across all five trained models.
-   **Confusion Matrices**: Placeholders for confusion matrices for each model (Random Forest, XGBoost, MLP, SVM, Logistic Regression) are included in the results chapter to provide a detailed view of their classification performance, distinguishing between true positives, true negatives, false positives, and false negatives.

### Supporting Documents

The methodology and future scope of this research were informed by seminal works in the fields of network security, machine learning, and explainable AI. Key references from the bibliography include:
-   The original paper for the **UNSW-NB15 dataset** [1], which guided data understanding.
-   The foundational papers for **ADASYN** [2], **LIME** [3], and **SHAP** [4], which informed the initial data balancing strategy and the planned future work on XAI.

### Clarifications

To ensure clarity and accurate interpretation, the following definitions for key terms and abbreviations are provided:

-   **DoS (Denial of Service)**: A type of cyber-attack in which the perpetrator seeks to make a machine or network resource unavailable to its intended users by temporarily or indefinitely disrupting services.
-   **Ensemble Learning**: A machine learning technique where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. Random Forest and XGBoost are examples.
-   **False Positive**: An outcome where the model incorrectly predicts the positive class. In this context, it means a normal activity is flagged as a DoS attack.
-   **False Negative**: An outcome where the model incorrectly predicts the negative class. In this context, it means a real DoS attack is missed and classified as normal activity.
-   **Feature Engineering**: The process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data.
-   **XAI (Explainable AI)**: Artificial intelligence (AI) in which the results of the solution can be understood by humans. It contrasts with the "black box" concept in which even its designers cannot explain why the AI arrived at a specific decision.
-   **LIME (Local Interpretable Model-agnostic Explanations)**: An XAI technique that explains the prediction of any classifier by learning an interpretable model locally around the prediction.
-   **SHAP (SHapley Additive exPlanations)**: A game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.
