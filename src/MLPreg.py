import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def train_nn(data):  
  # Create features and target
  features = ["position_top", "position_jng", "position_mid", "position_bot", "position_sup", "prob_result", "damageshare", "killshare"]
  target = data['kills']

  # Create X and y
  X = data[features]
  y = target

  # Split data into training data and test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

  # Create and fit the scaler
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Train the model with scaled data
  model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, learning_rate_init=0.001, random_state=6)
  model.fit(X_train_scaled, y_train)

  # Save the best model
  joblib.dump(model, 'nn_model.joblib')

  # Save the scaler
  joblib.dump(scaler, 'scaler.joblib')

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv(R'data/cleaned_data.csv')  # Replace with your actual data file

    # Call the train_and_grid_search function
    train_nn(data)