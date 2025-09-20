METHODOLOGY

Introduction

This chapter outlines the systematic methodology employed in this research to develop a high-performance DoS attack detection model. The approach is grounded in the principle that rigorous data engineering is a prerequisite for building effective machine learning systems. The methodology is structured into three primary stages, which directly correspond to the objectives outlined in the previous chapter:

1.  Dataset Selection and Description: Justifying the choice of the UNSW-NB15 dataset and describing its key characteristics.
2.  Data Preprocessing and Feature Engineering Pipeline: A detailed, 6-phase pipeline designed to transform the raw data into a clean, balanced, and optimized format for model training.
3.  Model Training and Evaluation: The process for training, tuning, and evaluating a suite of machine learning models to identify the most effective classifier for DoS detection.

Each stage is designed to be systematic and reproducible, ensuring that the results are both reliable and verifiable. This data-centric approach directly addresses the common pitfalls of model development, such as class imbalance and feature redundancy, which were identified as critical gaps in the literature review.

Implementation Strategy

The implementation strategy for this project follows a structured, multi-phase pipeline designed to transform raw network traffic data into a highly accurate DoS detection model. Each phase is modular, ensuring that the process is both systematic and reproducible. The strategy encompasses data preprocessing, feature engineering, model training, and evaluation.

Phase 1: Data Cleaning and Preparation
The process begins with loading the UNSW-NB15 dataset. Initial cleaning involves removing duplicate records and handling any missing values to ensure data integrity. Categorical features are identified and prepared for numerical conversion.

Phase 2: Feature Encoding
Non-numerical features such as 'proto', 'service', and 'state' are converted into a machine-readable format using one-hot encoding. An encoding map is generated and saved to ensure that the same transformations can be consistently applied to new data in the future.

Phase 3: Data Balancing and Feature Engineering
To address the significant class imbalance between "Normal" and "DoS" attack data, a suitable sampling technique is applied to the training set. This prevents the model from developing a bias towards the majority class. A two-step feature reduction process is then executed: first by removing highly correlated features to reduce multicollinearity, and second by eliminating low-variance features that offer little discriminatory value.

Phase 4: Data Normalization
The final, refined feature set is normalized using StandardScaler. This scales all features to have a mean of zero and a standard deviation of one, which is crucial for the performance of many machine learning algorithms.

Phase 5: Model Training and Hyperparameter Tuning
A suite of machine learning models (Random Forest, XGBoost, SVM, Logistic Regression, MLP) is trained on the fully preprocessed data. GridSearchCV is employed to systematically search for the optimal hyperparameters for each model, ensuring each is configured for its best possible performance.

Phase 6: Model Evaluation and Selection
The performance of the tuned models is rigorously evaluated and compared using standard metrics, including Accuracy, Precision, Recall, and F1-Score. The model that demonstrates the best overall performance is selected as the final, proposed model for this research phase.

4.3 Tools/Software to be Used

The development and implementation of this project rely on a combination of open-source programming languages, libraries, and development tools. The selection was based on industry standards for data science and machine learning, ensuring robustness, scalability, and reproducibility.

| Layer | Tool/Technology Used | Purpose |
| :--- | :--- | :--- |
| Programming Language | Python | Core language for data analysis and model development. |
| Data Manipulation | Pandas, NumPy | For efficient data loading, cleaning, and transformation. |
| Machine Learning | Scikit-learn, XGBoost | To implement and train models (Random Forest, SVM, etc.) and for preprocessing tasks. |
| Hyperparameter Tuning | GridSearchCV (from Scikit-learn) | To systematically find the best model parameters. |
| Development Environment | Jupyter Notebook, VS Code | For exploratory data analysis, script development, and code management. |
| Version Control | Git, GitHub | For source code management and collaboration. |

4.4 Expected Outcome

The expected outcome of this research is a highly accurate and robust machine learning model for the detection of Denial-of-Service (DoS) attacks, validated through a systematic and reproducible methodology. The work will produce a comprehensive analysis of the effectiveness of various data engineering techniques and machine learning algorithms on the UNSW-NB15 dataset.

The key deliverables of this phase include a fully preprocessed dataset, which is a high-quality, clean, balanced, and normalized dataset derived from the UNSW-NB15 dataset, ready for machine learning applications. Additionally, a suite of optimized models will be delivered, comprising a collection of trained and hyperparameter-tuned machine learning models, including Random Forest, XGBoost, SVM, Logistic Regression, and MLP. A comparative performance report will provide a detailed evaluation of all trained models, leading to the final deliverable: the single best-performing model, identified and saved as the primary outcome of this research phase.

The success of the project will be measured against several performance metrics. Model performance will be evaluated using Accuracy, which measures the overall percentage of correct classifications; Precision, which indicates the model's ability to avoid false positives; Recall (Sensitivity), which measures the ability to identify all actual DoS attacks; and the F1-Score, which provides a balanced measure of precision and recall.

Computational efficiency will be assessed by measuring the training and prediction time for each model. Finally, robustness will be determined by analyzing cross-validation scores to ensure the model generalizes well to unseen data.

