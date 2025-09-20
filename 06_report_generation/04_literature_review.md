Literature Review

This chapter reviews existing research in the fields of Denial-of-Service (DoS) attack detection, feature engineering, and the application of machine learning in cybersecurity. The findings from these studies inform the methodology of this project and highlight the research gaps that this work aims to address.

Machine Learning for DoS Attack Detection

A significant body of research has demonstrated the effectiveness of machine learning (ML) for detecting DoS attacks. A comparative study by A. B. G. et al. (2025) evaluated several models, including Random Forest (RF), Convolutional Neural Networks (CNN), and Logistic Regression (LR). Their findings showed that RF achieved exceptional performance with 99% accuracy, while LR was noted as suitable for low-resource environments. However, the study also highlighted critical gaps, including a lack of focus on real-time detection and the challenge of overfitting due to class imbalance in datasets. This underscores the need for robust data preprocessing and model selection, which is a primary focus of our project's first phase.

Feature Selection and Engineering

The performance of ML-based detection systems is heavily dependent on the quality of the input features. Kasongo and Sun (2020) conducted a performance analysis on the same UNSW-NB15 dataset used in our project. Their work emphasized the importance of feature selection, demonstrating that using XGBoost for feature ranking significantly improved the accuracy of their Intrusion Detection System (IDS). A key limitation noted in their work was that class imbalance was not addressed, which they identified as a necessary step for future work. Our project directly addresses this gap by implementing a systematic data balancing and feature reduction pipeline before model training.

Similarly, research by Kamaldeep et al. (2023) on a standardized IoT dataset reinforced the value of feature engineering. Their framework, which included feature normalization and selection, showed that RF and Multilayer Perceptron (MLP) models outperformed other classifiers. This highlights a consistent theme in the literature: effective feature engineering is a critical precursor to building high-performance detection models.

Gaps Identified for Future Work

While the focus of this report is on data preparation and model training, the literature also points to significant gaps that motivate the long-term vision of our project. Studies on Explainable AI (XAI) by Gaspar et al. (2024) and Wang et al. (2020) have successfully applied techniques like SHAP and LIME to intrusion detection systems, but often in a theoretical context. Feng et al. (2023) noted that while some systems provide explanations, they often lack actionable data that can be directly used for mitigation. This "actionability gap" is a key area that Phase 2 of our project will address by linking XAI insights to a practical mitigation engine.

Summary

The existing literature confirms that machine learning is a powerful tool for DoS detection but emphasizes that success is contingent on careful data preparation and feature engineering to handle issues like class imbalance and high dimensionality. Our Phase 1 work is positioned to address these foundational challenges directly, building a high-performance model that will serve as the basis for incorporating explainability and automated mitigation in future research.

Inferences drawn from literature review

Based on the review of existing literature, several key inferences have been drawn that directly shape the methodology of this project:

Data Quality over Model Complexity: The literature consistently shows that the performance of machine learning models in intrusion detection is more sensitive to the quality of data and features than to the complexity of the model itself. Studies repeatedly demonstrate that even simpler models can achieve high accuracy when trained on well-processed, balanced, and relevant features. This project, therefore, prioritizes a rigorous and systematic data preprocessing and feature engineering pipeline as its foundational first step.

The Critical Need to Address Class Imbalance: A recurring limitation in many studies is the failure to properly address class imbalance, which leads to models that are biased towards the majority class and perform poorly in detecting actual attacks. This highlights the necessity of implementing data balancing techniques. Our work directly confronts this by evaluating and selecting an appropriate method to ensure the model is trained on a representative distribution of normal and attack data.

Feature Engineering is Non-Negotiable: Research confirms that raw network traffic data is not suitable for direct use in machine learning models. Effective feature selection and engineering are critical for reducing dimensionality, eliminating noise, and improving model accuracy and efficiency. The success of methods like XGBoost-based feature ranking in previous studies validates our project's focus on a multi-stage feature reduction process.

Actionability is the Next Frontier: While this phase of the project focuses on detection, the literature review clearly indicates a significant gap between detecting an attack and providing actionable insights for mitigation. The research in XAI is promising but often remains theoretical. This "actionability gap" validates the long-term vision of our project, where the high-performance model developed in Phase 1 will serve as the core for a future XAI-driven mitigation engine in Phase 2.
