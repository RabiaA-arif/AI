# Credit Risk Analysis Using Machine Learning

 # Goal
  machine learning that predict the risk lavel of credit
  good -> low risk ->suppose 1
  bad -> high risk -> suppose 0


  # Project overview 
   in this project i used german credit risk dataset and developed a prediction     model that can predict whether borrower is repay or not based on their       
   financial and some features 

    the solution include data cleaning ,feature encoding,model  
    training,evalutation,deployment via streamlit 

# Key Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost,DecisionTreeClassifier ,RandomForecast,ExtraTreeClassifier Streamlit, joblib

# Dataset and Initial Analysis
source  : German Credit Risk Dataset from kaggle
Rows /Column : 1000/11
Categorical Columns : 6
Numerical Column : 5
Selected Features : Age, Sex, Job, Housing, Saving Accounts, Checking Accounts, Credit Amount, Duration
Target feature : Risk 

# Data Cleaning & Feature Engineering
1. Missing Values : Rows with missing values in Saving Accounts or Checking Accounts were dropped (df.dropna())

2. Final data shape : 522 rows and 8 columns

3. Duplicate : not found

4. Categorical Encoding : Label Encoding applied to all categorical columns using LabelEncoder

5. Scalling : Not applied — tree-based models are scale-invariant

6. Persistent Assets : All fitted encoders were saved as .pkl files (sex_encoder.pkl, housing_encoder.pkl


# Modeling, Evaluation & Selection
1. Data Split
Train-Test Split: 80% Train / 20% Test
Sampling Strategy: Stratified (to preserve class distribution)
Random State: 1 (ensures reproducibility)
2. Class Imbalance Handling

Used class_weight='balanced' for tree-based models.
Calculated scale_pos_weight for XGBoost.

2. Models Trained

Decision Tree Classifier
Random Forest Classifier
Extra Trees Classifier
XGBoost Classifier

3. Hyperparameter Tuning
Used Grid Search Cross Validation (CV=5) to identify optimal parameters for each model.

4. Evaluation Metric
Accuracy Score

# Results 
Model               :  Accuracy



