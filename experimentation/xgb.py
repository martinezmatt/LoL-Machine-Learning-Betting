import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_exp(data):  
  # Create features and target
  features = ["position_top", "position_jng", "position_mid", "position_bot", "position_sup", "prob_result", "damageshare", "killshare"]
  target = data['kills']

  # Create X and y
  X = data[features]
  y = target
  
  # Split data into training data and test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

  xgb_model = XGBRegressor(max_depth=20, objective ='reg:squarederror', n_estimators=200, random_state=6)
  xgb_model.fit(X_train, y_train)
  xgb_preds = xgb_model.predict(X_test)
  mae_xgb = mean_absolute_error(y_test, xgb_preds)
  print(f'MAE XGBoost: {mae_xgb}')

  # Save the best model
  joblib.dump(xgb_model, 'xgb_model.joblib')

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv(R'data/cleaned_data.csv')  # Replace with your actual data file

    # Call the train_and_grid_search function
    train_exp(data)