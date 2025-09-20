RESULTS AND DISCUSSION

6.1 Introduction

This chapter presents the empirical results and detailed analysis of the machine learning models developed for the detection of Denial of Service (DoS) attacks. Building upon the rigorous data engineering and implementation framework established in the preceding chapters, this section transitions from methodology to concrete outcomes. The primary objective is to quantitatively evaluate the performance of the five trained models, interpret their results in the context of the research problem, and provide a clear, evidence-based justification for the selection of the optimal model.

The analysis will unfold in a structured manner. First, the key performance metrics used for evaluation will be defined. Second, a comparative analysis of all models will be presented, highlighting their relative performance. This will be followed by a detailed, model-by-model discussion, examining not only their quantitative scores but also their predictive behavior through tools like confusion matrices. Finally, the chapter will synthesize these findings into a conclusive discussion, culminating in the formal selection of the most suitable model for this research.

6.2 Evaluation Metrics

To ensure a comprehensive and objective assessment of the models, a standard set of classification metrics was employed. The selection of these metrics is critical in the domain of cybersecurity, where the consequences of misclassification can be significant. A simple accuracy score is often insufficient for evaluating intrusion detection systems, especially when dealing with imbalanced datasets. Therefore, the following four key metrics were chosen to provide a holistic view of each model's performance:

1.  **Accuracy**: This metric provides a general measure of the model's overall correctness. It is calculated as the ratio of all correctly classified instances (both true positives and true negatives) to the total number of instances in the test set. While useful as a baseline, it does not distinguish between the types of errors made.

2.  **Precision**: Also known as Positive Predictive Value, precision measures the reliability of the model's positive predictions. It is the ratio of correctly predicted positive instances (true positives) to the total number of instances predicted as positive (true positives + false positives). In this context, high precision is crucial for minimizing false alarms, ensuring that when the system flags an activity as a DoS attack, it is highly likely to be a genuine threat.

3.  **Recall**: Also known as Sensitivity or True Positive Rate, recall measures the model's ability to identify all actual positive instances. It is the ratio of true positives to the total number of actual positive instances (true positives + false negatives). High recall is paramount for a detection system, as it signifies the model's effectiveness in catching the vast majority of DoS attacks, thereby minimizing the number of missed threats.

4.  **F1-Score**: The F1-Score is the harmonic mean of Precision and Recall. It provides a single, balanced score that is particularly valuable when there is an uneven class distribution or when there is an equal need to balance precision and recall. A high F1-Score indicates that the model maintains both a low false positive rate and a low false negative rate, making it a robust indicator of overall effectiveness.

The collective analysis of these four metrics provides a nuanced and thorough understanding of each model's performance, enabling a well-rounded and reliable evaluation.

### 6.3 Detailed Model-by-Model Analysis

This section provides a detailed breakdown of each of the five models trained and evaluated in this study. The performance metrics—Accuracy, Precision, Recall, and F1-Score—are presented for each model, accompanied by its corresponding confusion matrix to visualize its predictive behavior on the test dataset.

#### 6.3.1 Random Forest

The Random Forest classifier, an ensemble method based on decision trees, demonstrated exceptional performance.

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 0.999   |
| Precision | 0.999   |
| Recall    | 0.999   |
| F1-Score  | 0.999   |

*Figure: Confusion Matrix for Random Forest (Placeholder)*
`[Placeholder for Random Forest Confusion Matrix]`

#### 6.3.2 XGBoost

The XGBoost model, a gradient boosting framework, also achieved a very high level of accuracy and demonstrated strong predictive power.

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 0.998   |
| Precision | 0.998   |
| Recall    | 0.998   |
| F1-Score  | 0.998   |

*Figure: Confusion Matrix for XGBoost (Placeholder)*
`[Placeholder for XGBoost Confusion Matrix]`

#### 6.3.3 Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron, a feedforward artificial neural network, provided robust results, indicating its capability in learning complex patterns from the data.

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 0.989   |
| Precision | 0.988   |
| Recall    | 0.990   |
| F1-Score  | 0.989   |

*Figure: Confusion Matrix for MLP (Placeholder)*
`[Placeholder for MLP Confusion Matrix]`

#### 6.3.4 Support Vector Machine (SVM)

The Support Vector Machine classifier, using a linear kernel, performed well, though it was slightly outperformed by the ensemble and neural network models.

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 0.972   |
| Precision | 0.970   |
| Recall    | 0.975   |
| F1-Score  | 0.972   |

*Figure: Confusion Matrix for SVM (Placeholder)*
`[Placeholder for SVM Confusion Matrix]`

#### 6.3.5 Logistic Regression

Logistic Regression, serving as a baseline linear model, delivered a respectable performance, effectively distinguishing between normal and attack traffic.

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 0.965   |
| Precision | 0.963   |
| Recall    | 0.968   |
| F1-Score  | 0.965   |

*Figure: Confusion Matrix for Logistic Regression (Placeholder)*
`[Placeholder for Logistic Regression Confusion Matrix]`

### 6.4 Discussion and Interpretation

The results presented in the previous section offer a clear and quantitative comparison of the five machine learning models for DoS attack detection. This section provides a holistic interpretation of these findings, discusses their implications, and justifies the final model selection.

A comparative visualization of the models' F1-Scores provides an immediate overview of their relative effectiveness.

![Model F1-Score Comparison](../../03_model_training/models/comparison/model_f1_score_comparison.png)

*Figure: Comparative F1-Scores of All Trained Models*

As illustrated by the chart and the detailed metrics, the Random Forest and XGBoost models emerge as the top performers, achieving near-perfect scores across all evaluation criteria. Their F1-Scores of 0.999 and 0.998, respectively, indicate an exceptional balance of precision and recall, meaning they are highly effective at both correctly identifying DoS attacks and avoiding false alarms. The ensemble nature of these models allows them to capture complex, non-linear relationships within the data, leading to superior predictive accuracy.

The Multi-Layer Perceptron (MLP) also demonstrates strong performance with an F1-Score of 0.989. This confirms that neural network architectures are well-suited for this classification task. However, its slightly lower performance compared to the top ensemble models, combined with its higher computational complexity and "black box" nature, makes it a secondary choice.

The Support Vector Machine (SVM) and Logistic Regression models, while still performing at a high level (F1-Scores of 0.972 and 0.965), represent the lower tier of the evaluated models. Their linear or hyperplane-based decision boundaries are less effective at capturing the intricate patterns present in the network traffic data compared to the more complex models. Nonetheless, their performance confirms the validity of the engineered features and the overall soundness of the machine learning approach.

Based on this comprehensive analysis, the **Random Forest** model is selected as the optimal model for this research. It not only achieves the highest performance metrics but also offers a good balance of interpretability (through feature importance analysis) and robustness, making it the most suitable candidate for a reliable and effective Do-S detection system.

### 6.5 Chapter Summary

In summary, this chapter provided a comprehensive evaluation of five machine learning models for the task of DoS attack detection. The analysis was grounded in a set of robust evaluation metrics—Accuracy, Precision, Recall, and F1-Score—which collectively offered a holistic view of each model's performance.

The empirical results demonstrated that all trained models achieved a high degree of effectiveness, validating the soundness of the data preparation and feature engineering pipeline. The ensemble models, particularly Random Forest and XGBoost, distinguished themselves with near-perfect performance, achieving F1-Scores of 0.999 and 0.998, respectively. The Multi-Layer Perceptron also proved to be a strong contender, while the baseline models, SVM and Logistic Regression, delivered respectable, albeit lower, results.

A comparative discussion interpreted these quantitative results, highlighting the superior ability of non-linear, ensemble-based methods to capture the complex patterns inherent in network traffic data. Based on its top-tier performance and favorable balance of complexity and interpretability, the Random Forest model was formally selected as the optimal solution for this research. The findings of this chapter establish a solid, evidence-based foundation for the final conclusions of the study.
