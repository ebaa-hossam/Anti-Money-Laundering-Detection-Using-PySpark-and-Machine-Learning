# Anti-Money-Laundering-Detection-Using-PySpark-and-Machine-Learning
This project focuses on detecting money laundering activities using machine learning techniques and PySpark for big data processing. The dataset includes financial transactions. The goal is to identify suspicious activities (money laundering) with high accuracy and efficiency, addressing class imbalance and leveraging scalable solutions.

## Key Components:

**1. Data Preparation:**

- Cleaned and preprocessed a large dataset **17.95 GB** containing financial transactions.
- Handled imbalanced data using SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority class (isLaundering).
- Extracted new features like FanOut, FanIn, and AvgAmountSent using PySpark's Window functions to improve feature richness.
  
**2. Feature Engineering:**

- Applied feature scaling to numeric features (e.g., Amount_Received, FanOut, FanIn) using PySpark's StandardScaler.
- Converted categorical features (e.g., PatternTypeIndex, CurrencyIndex) to indexed numeric formats for machine learning compatibility.
  
**3. Exploratory Data Analysis:**

- Visualized feature distributions and generated correlation heatmaps to understand relationships between features and the target variable (isLaundering).
  
**4. Model Training:**

- Built machine learning models using PySpark's MLlib, including:
- Random Forest Classifier: Achieved high F1 scores **0.963** and robust performance with a focus on feature importance.
- Performed hyperparameter tuning using cross-validation to optimize model performance.
  
**5. Model Evaluation:**

- Evaluated model on validation and test datasets using:
- F1 Score: Balanced measure of precision and recall.
- Precision & Recall: To assess the trade-off between false positives and false negatives.
- ROC AUC Score: Demonstrated strong classification performance with scores above 0.92.
- Analyzed feature importance for interpretability and identified key contributors like **DayOfWeek** and **PatternTypeIndex**.
  
**6. Insights and Recommendations:**

- The Random Forest model achieved:
  
Validation F1 Score: 0.963

Test F1 Score: 0.963

Suggested improvements like adding more features, addressing class imbalance through class weights, and leveraging non-linear models.

**Technologies Used:**

PySpark: For distributed data processing and machine learning.

Machine Learning Algorithms: Random Forest.

Visualization Tools: Matplotlib, Seaborn for EDA and feature analysis.

Big Data Tools: Databricks for scalable notebook execution.

**How to Use the Notebook:**
- Clone the repository.
- Unzip the ipynb
- You can find the dataset used on kaggle https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml or you can use the files attached.
- Ensure PySpark and necessary dependencies are installed.
- Follow the steps for:
  
Data preprocessing and feature engineering.

Model training and evaluation.

Customize hyperparameters and re-train models for new datasets.

**Conclusion:**

This project demonstrates the power of big data processing with PySpark for tackling imbalanced classification problems in anti-money laundering. With robust models and scalable pipelines, it provides a foundation for further exploration and deployment in real-world scenarios.
