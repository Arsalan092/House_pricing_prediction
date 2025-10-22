from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score 
from lightgbm import LGBMRegressor as lgbmr 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import numpy as np 

#Acquire the dataset 
df = pd.read_csv('price_pred.csv', nrows = 1000000)
df_copy = df.copy() 


#engineering null counts 
df_copy['brokered_by'] = df_copy['brokered_by'].fillna(df_copy['brokered_by'].mean())
df_copy['price'] = df_copy['price'].fillna(df_copy['price'].mean())
df_copy['bed'] = df_copy['bed'].fillna(df_copy['bed'].median())
df_copy['bath'] = df_copy['bath'].fillna(df_copy['bath'].median())
df_copy['acre_lot'] = df_copy['acre_lot'].fillna(df_copy['acre_lot'].median())
df_copy['street'] = df_copy['street'].fillna(df_copy['street'].median())
df_copy['city'] = df_copy['city'].fillna(df_copy['city'].bfill())
df_copy['state'] = df_copy['state'].fillna(df_copy['state'].bfill())
df_copy['zip_code'] = df_copy['zip_code'].fillna(df_copy['zip_code'].median())
df_copy['house_size'] = df_copy['house_size'].fillna(df_copy['house_size'].median())
df_copy['prev_sold_date'] = df_copy['prev_sold_date'].fillna(df_copy['prev_sold_date'].bfill())

#engineering date
df_copy['prev_sold_date'] = pd.to_datetime(df_copy['prev_sold_date'], errors = 'coerce')
df_copy['years'] = df_copy['prev_sold_date'].dt.year 
df_copy['month'] = df_copy['prev_sold_date'].dt.month 
df_copy['days'] = df_copy['prev_sold_date'].dt.day
df_copy[['years', 'month', 'days']] = df_copy[['years', 'month', 'days']].fillna(df_copy[['years', 'month', 'days']].bfill())

#Freq encoding 
cat_cols = ['status', 'city', 'state']
for i in cat_cols:
    freq = df_copy[i].value_counts() / len(df_copy)
    df_copy[i] = df_copy[i].map(freq)
    
# training 
x = df_copy.drop(columns = ['price', 'prev_sold_date'], axis = 1)
y = df_copy['price']

#removing outlier
q_low, q_high = y.quantile([0.01, 0.99])
mask = (y > q_low) & (y < q_high)
x, y = x[mask],  y[mask]

#log transformation to reduce skewness
y = np.log1p(y)


x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)


#def
def long_data(x_train, y_train, x_test, y_test):
    lgb = lgbmr(objective = 'regression',
        boosting_type = 'gbdt',
        num_leaves = 128,
        n_estimators = 3000,
        learning_rate = 0.05,
        max_depth = 12, 
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_samples = 50,
        n_jobs = -1,
        random_state = 42,
        reg_alpha = 0.3,
        reg_lambda = 0.5
    )
    lgb.fit(x_train, y_train, eval_set = [(x_test, y_test)], eval_metric = 'r2')
    pred = lgb.predict(x_test)
    y_pred = np.clip(pred, np.percentile(y, 1), np.percentile(y, 99))
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('mae', mean_absolute_error(y_test, y_pred))
    print('mse', mse)
    print('rmse', rmse)
    print('r2', r2_score(y_test, y_pred))
    return lgb, y_pred, pred 

model = long_data(x_train, y_train, x_test, y_test)
