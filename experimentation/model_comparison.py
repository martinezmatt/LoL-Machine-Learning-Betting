## Random Forest best MAE, MSE, R2

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Read in raw data with ALL info of games that were played in 2023
alldata = pd.read_csv(r"data/cleaned_data.csv")

# Create features and target
features = ["position_top", "position_jng", "position_mid", "position_bot", "position_sup", "prob_result", "damageshare", "killshare"]
target = alldata['kills']

# Create X and y (a lil redundant but just for my brains sake)
X = alldata[features]
y = target

# Split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

## Define the LR model
LR = LogisticRegression(penalty="l2", C=0.1)
LR.fit(X_train, y_train)
LR_test_predictions = LR.predict(X_test)
# Show MAE
LR_test_mae = mean_absolute_error(LR_test_predictions, y_test)
print("Test MAE for linear regression model: {:,.0f}".format(LR_test_mae))

LR_mse = mean_squared_error(y_test, LR_test_predictions)
print(f'Mean Squared Error (MSE): {LR_mse}')

LR_r_squared = r2_score(y_test, LR_test_predictions)
print(f'R-squared (R²): {LR_r_squared}')
##

## Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=6)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_preds)
print(f'MAE Random Forest: {mae_rf}')

rf_mse = mean_squared_error(y_test, rf_preds)
print(f'Mean Squared Error (MSE): {rf_mse}')

rf_r_squared = r2_score(y_test, rf_preds)
print(f'R-squared (R²): {rf_r_squared}')
##

## XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=6 )
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, xgb_preds)
print(f'MAE XGBoost: {mae_xgb}')

xgb_mse = mean_squared_error(y_test, xgb_preds)
print(f'Mean Squared Error (MSE): {xgb_mse}')

xgb_r_squared = r2_score(y_test, xgb_preds)
print(f'R-squared (R²): {xgb_r_squared}')
##

## MLP NN
# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the neural network with the best hyperparameters we found
nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, learning_rate_init=.001, random_state=6)
nn_model.fit(X_train_scaled, y_train)
joblib.dump(nn_model, 'nn_model.joblib')

# Make predictions
nn_preds = nn_model.predict(X_test_scaled)

# Evaluate the model
mae_nn = mean_absolute_error(y_test, nn_preds)
print(f'MAE Neural Network: {mae_nn}')

nn_mse = mean_squared_error(y_test, nn_preds)
print(f'Mean Squared Error (MSE): {nn_mse}')

nn_r_squared = r2_score(y_test, nn_preds)
print(f'R-squared (R²): {nn_r_squared}')
##

#Test MAE for linear regression model: 1
#Mean Squared Error (MSE): 3.222950997398092
#R-squared (R²): 0.5786896032771078

#MAE Random Forest: 0.47082049818913646
#Mean Squared Error (MSE): 0.9333960326350716
#R-squared (R²): 0.8704017471407579

#MAE XGBoost: 0.6045130118912547
#Mean Squared Error (MSE): 1.2118339260776978
#R-squared (R²): 0.8317417751049863

#MAE Neural Network: 0.794109623716176
#Mean Squared Error (MSE): 1.6481722651427628
#R-squared (R²): 0.7711579667094295