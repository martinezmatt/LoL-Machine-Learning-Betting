import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Read in raw data with ALL info of games that were played in 2023
alldata = pd.read_csv(r"data\allmatchesthisyear.csv")

# Clean data to what we want: LCS, Spring
alldata = alldata[(alldata['league'].str.contains("LCS")) & (alldata['split'].str.contains("Spring")) & (alldata['datacompleteness'].str.contains("complete"))]

# Subset the data to the columns we want
alldata = alldata[["position", "playername", "teamname", "result", "kills", "deaths", "assists", "teamkills", "teamdeaths", "team kpm"]]

# Create features and target, need to add enemy team
features = ['playername', 'teamname']
target = alldata['kills']

# Create X and y (a lil redundant but just for my brains sake)
X = alldata[features]
y = target

# Split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Define the model
LR = LogisticRegression(penalty="l2", C=0.1)
LR.fit(X_train, y_train)
LR_test_predictions = LR.predict(X_test)
LR_test_mae = mean_absolute_error(LR_test_predictions, y_test)

print("Test MAE for linear regression model: {:,.0f}".format(LR_test_mae))