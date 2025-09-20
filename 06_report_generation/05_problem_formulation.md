PROBLEM FORMULATION AND PROPOSED WORK

Introduction

The proliferation of Denial-of-Service (DoS) attacks poses a significant and ongoing threat to the availability and reliability of network services. These attacks, in their simplest form, aim to overwhelm a target system with a flood of illegitimate traffic, rendering it incapable of serving legitimate users. While the concept is straightforward, the increasing sophistication of attack vectors and the sheer volume of network data make effective and timely detection a complex challenge. Traditional signature-based intrusion detection systems often fall short, as they struggle to identify novel or zero-day attack patterns.

This limitation has led to a paradigm shift towards machine learning-based detection systems, which have demonstrated a superior ability to learn from data and identify anomalous patterns indicative of an attack. However, as highlighted in the literature review, the success of these systems is not guaranteed. It is heavily contingent on addressing several foundational challenges, including poor data quality, class imbalance in training datasets, and the high dimensionality of network features. Failure to address these issues can lead to models that are either inaccurate or biased, resulting in a high rate of false positives or negatives.

This research formulates the problem as a supervised machine learning task aimed at binary classification: accurately distinguishing malicious DoS traffic from benign, normal network traffic. The proposed work directly confronts the aforementioned challenges by introducing a systematic and rigorous methodology. This methodology is not focused on developing a novel algorithm but rather on engineering a high-quality data pipeline and applying established machine learning models to achieve a high degree of accuracy and reliability. The central hypothesis is that a meticulously preprocessed and balanced dataset, combined with a well-tuned machine learning model, can form the basis of a highly effective DoS detection system. This chapter details the specific research questions, objectives, and the scope of the work undertaken to validate this hypothesis.

Problem Statement

The widespread adoption of networked systems has made them critical infrastructure, yet they remain highly vulnerable to Denial-of-Service (DoS) attacks. Existing detection methods, including both traditional signature-based systems and basic machine learning models, consistently struggle with two fundamental issues: the high rate of false positives, which can lead to blocking legitimate users, and the failure to detect novel or sophisticated attack variants, leaving systems exposed.

Therefore, the central problem is the lack of a reliable and robust framework for DoS attack detection that can effectively minimize false alarms while maintaining high detection accuracy for diverse attack patterns. This problem is rooted not in the absence of powerful machine learning algorithms, but in the inadequate and often-ad-hoc data preprocessing and feature engineering applied before model training. The presence of class imbalance, redundant and irrelevant features, and un-normalized data in raw network traffic datasets severely degrades the performance of even state-of-the-art models.

This research directly addresses this problem by focusing on the systematic engineering of the data itself to create an optimal foundation for machine learning-based detection.

Objectives

To address the problem statement, this research pursues a set of specific, measurable objectives. These objectives are structured to logically progress from data acquisition to model deployment, ensuring a comprehensive and rigorous approach. The primary objectives of this work are:

1. To implement a comprehensive data preprocessing and feature engineering pipeline for the UNSW-NB15 dataset. This involves systematically cleaning the data, encoding categorical features, balancing the class distribution, and applying feature selection techniques to produce a high-quality dataset optimized for machine learning.

2. To train, evaluate, and compare a suite of machine learning models for the task of DoS attack detection. This includes Random Forest, XGBoost, SVM, Logistic Regression, and MLP. The goal is to identify the best-performing model based on key metrics such as accuracy, precision, recall, and F1-score.

3. To develop a framework for Explainable AI (XAI) to interpret the decisions of the best-performing model. This involves using techniques like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to understand which features are most influential in detecting an attack. (Future Work)

4. To design and propose a proof-of-concept for an automated mitigation engine. This engine would leverage the insights from the XAI framework to generate actionable, human-readable mitigation rules, bridging the gap between detection and response. (Future Work)

This report focuses specifically on the completion of Objective 1 and Objective 2. Objectives 3 and 4 are part of the broader project vision and are designated as future work.

Proposed Work

![1758222600153](image/05_problem_formulation/1758222600153.png)

