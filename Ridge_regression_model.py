import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error , mean_squared_error

#load csv data
data = pd.read_csv('housing.csv')

#feature engennering 
# to drop capped values
data = data[data['median_house_value'] < 500000]
# log transformation of target variable
data['log_median_house_value'] = np.log(data['median_house_value'])
# to drop ocean_proximity
data = data.drop(["ocean_proximity"], axis=1)
# fill null values of total_bedrooms with mean 
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())

# Seprate feature from target
X = data.drop(["median_house_value" , "log_median_house_value"], axis=1)
Y = data["log_median_house_value"]

# train-test split
X_train , X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# scale numerical features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train ridge regression
model = Ridge(alpha=0.01)
model.fit(X_train, Y_train)

# predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# calculate r2 score
r2 = r2_score(Y_test,  Y_test_pred)
mae = mean_absolute_error(Y_test, Y_test_pred)
mse = mean_squared_error(Y_test, Y_test_pred)
rmse = np.sqrt(mse)

# print values
print("R2 score : ", r2)
print("MAE : ", mae)
print("RMSE : ", rmse)


