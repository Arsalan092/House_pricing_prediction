🏡 House Price Prediction using LightGBM
This repository contains a machine learning pipeline for predicting house prices. The project leverages LightGBM Regressor with advanced preprocessing steps such as missing value imputation, frequency encoding, outlier removal, and log transformation to improve prediction accuracy.

✨ Key Features:

- 🔧 Data Preprocessing
- Handles missing values using mean, median, and backfill strategies.
- Extracts year, month, day from prev_sold_date.
- Frequency encoding for categorical features: status, city, state.
  
- 📉 Outlier Handling
- Removes extreme values using the 1st and 99th percentiles.
- 🔄 Log Transformation
- Applies log1p transformation on target variable (price) to reduce skewness.
  
- 🤖 Model Training
- LightGBM Regressor with tuned hyperparameters.
- Evaluation metric: R² score.
- Regularization (reg_alpha, reg_lambda) to prevent overfitting.
  
- 📊 Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
