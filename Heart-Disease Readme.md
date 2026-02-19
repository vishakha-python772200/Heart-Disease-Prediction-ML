# Heart-Disease-Prediction-ML
Machine Learning model to predict heart disease using Decision Tree &amp; SVM with complete EDA, ROC analysis and model evaluation.
# 🫀 Heart Disease Prediction using Machine Learning

This project focuses on predicting the presence of heart disease using supervised machine learning algorithms.

The project includes:
- Complete Exploratory Data Analysis (EDA)
- Feature relationship visualization
- Data preprocessing & scaling
- Model training (Decision Tree & SVM)
- Model evaluation using Accuracy, Confusion Matrix, Classification Report
- ROC Curve & AUC Score analysis

---

## 📊 Problem Statement

Heart disease is one of the leading causes of death worldwide. 
The goal of this project is to build a classification model that predicts whether a patient has heart disease based on medical attributes.

---

## 📁 Dataset Information

The dataset contains 3000+ patient records with features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression (Oldpeak)
- Target (0 = No Disease, 1 = Disease)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA includes:

- Histogram for age distribution
- Violin plot (Cholesterol vs Target)
- Bar plot (Gender vs Disease Risk)
- Scatter plot (Age vs Cholesterol)
- Correlation Heatmap
- Pairplot for feature interaction

EDA helped in understanding feature importance and class separability.

---

## 🤖 Machine Learning Models Used

### 🌳 Decision Tree Classifier
- max_depth = 4
- min_samples_split = 5
- min_samples_leaf = 2

### ⚙️ Support Vector Machine (SVM)
- Kernel: RBF
- Feature Scaling applied
- probability=True for ROC curve

---

## 📈 Model Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve
- AUC Score = 0.97 (Excellent Performance)

---

## 📊 Results

The SVM model achieved an AUC score of 0.971, indicating strong classification capability.

The ROC curve shows excellent separation between positive and negative classes.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature engineering
- Deployment using Flask/Streamlit
- Model explainability (SHAP / Feature Importance)

---

## 💡 Key Learnings

- Importance of EDA before modeling
- Feature scaling impact on SVM
- Understanding ROC-AUC interpretation
- Comparing multiple classification models

---

## 👩‍💻 Author

Vishakha Badgujar  
Aspiring Data Scientist | Machine Learning Enthusiast
