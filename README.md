# 🚗 Used Car Price Prediction & Market Analysis

## 📌 Project Overview
This project applies predictive analytics and machine learning techniques to analyze and predict used car prices using the Craigslist Cars & Trucks dataset. The pipeline covers data cleaning, exploratory analysis, classification, regression, and clustering.

## 📂 Dataset
- Source: Kaggle (Craigslist Cars & Trucks)
- Size: Millions of records
- Features include price, year, odometer, fuel type, transmission, drive, and location.

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🔍 Key Tasks Performed
- Large-scale data cleaning using chunk processing
- Exploratory Data Analysis (EDA)
- Price category classification (Budget / Midrange / Premium)
- Multiple ML model comparison
- Price prediction using regression models
- Market segmentation using clustering
- Bias-Variance analysis

## 🤖 Models Implemented
### Classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Neural Networks

### Regression
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Clustering
- K-Means
- Hierarchical Clustering

## 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, Log Loss
- MAE, RMSE, R² Score
- Silhouette Score

## 🚀 Results
- Random Forest models performed best for both classification and regression.
- Year and odometer are the most influential features.
- Cars can be meaningfully segmented into 3 market clusters.

## ▶️ How to Run
1. Clone the repository
2. Install dependencies
3. Place `vehicles.csv` in the project folder
4. Run the notebook or Python scripts

## 📌 Future Improvements
- Deploy using Streamlit
- Hyperparameter tuning
- Time-series price forecasting
