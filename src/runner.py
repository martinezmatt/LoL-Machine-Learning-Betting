import pandas as pd
import os
#from cleaner import clean_data
#from MLPReg import train_nn
import joblib

def runner():
  # get the data
  #data = clean_data()
  data = pd.read_csv(r"data/cleaned_data.csv")
  print("Clean data acquired")
  
  # get prizepicks data
  prizepicks = pd.read_csv(r"data/prizepicks.csv")
  prizepicks['My Projected Kills'] = 0.0
  print("Prizepicks data acquired and column added")
  
  # run the script for training the model
  # train_nn(data)

  ## find the models folder
  models_folder = os.path.join(os.path.dirname(__file__), 'models')

  model_path = os.path.join(models_folder, 'nn_model.joblib')
  scaler_path = os.path.join(models_folder, 'scaler.joblib')
  
  # load the model
  load_model = joblib.load(model_path)
  
  # load scaler
  scaler = joblib.load(scaler_path)
  ##
  
  # give me the player's name and potential result
  #player_name = input("Enter the player's name: ")
  #prob_result = input("Probability of winning from 0 to 1: ")
  
  # iterate thru prizepicks data
  for index, row in prizepicks.iterrows():\
    # get the name and win probability
    player_name = row['Player Name']
    prob_result = 0.5 # for now, need to actually get the win probability
    
    # get a df of the player's stats
    player_data = data[data['playername'] == player_name]
    
    if player_data.empty:
      print(f"{player_name} not found in the data.")
    else:
      new_data = pd.DataFrame({
        "position_top": player_data['position_top'].iloc[0],
        "position_jng": player_data['position_jng'].iloc[0],
        "position_mid": player_data['position_mid'].iloc[0],
        "position_bot": player_data['position_bot'].iloc[0],
        "position_sup": player_data['position_sup'].iloc[0],
        "prob_result": pd.to_numeric(prob_result),
        "damageshare": player_data['damageshare'].mean(),
        "killshare": player_data['killshare'].mean(),
      }, index=[0])
    
      # scale the data
      new_data = scaler.transform(new_data)
    
      # make the prediction
      prediction = load_model.predict(new_data).round(2)
    
      # assign result to prizepicks dataframe
      prizepicks.at[index, 'My Projected Kills'] = prediction[0]
  
  # return
  return(prizepicks)

if __name__ == '__main__':
  runner()