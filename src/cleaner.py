import pandas as pd
import os

def clean_data(data_path="data/allmatchessince2018.csv"):
  # Read in raw data with ALL info of games that were played in 2023
  alldata = pd.read_csv(data_path)
  
  # Clean data to what we want: LCS
  #alldata = alldata[(alldata['league'].str.contains("LCS")) &
                    #(alldata['datacompleteness'].str.contains("complete"))]
  
  # Clean data to what we want
  alldata = alldata[(alldata['datacompleteness'].str.contains("complete"))]
  
  # Subset the data to the columns we want
  alldata = alldata[["gameid", "league", "year", "split", "side", "position", "playername", "teamname", "result", "kills", "deaths", "assists", "teamkills", "teamdeaths", "team kpm", "damagetochampions", "damageshare"]]
  
  # Get rid of the entries with position == team
  alldata = alldata[alldata['position'] != 'team']
  
  # Add a killshare column
  alldata['killshare'] = alldata['kills'] / alldata['teamkills']

  # Make a prob_result to be used in the model instead of the binary result
  alldata['prob_result'] = alldata['result'].map({0: 0.1, 1: 0.9})
  
  ####ADDING ENEMY TEAM NAME USING gameid AND side####
  # Create a dictionary to map "side" to the corresponding opponent side
  opponent_mapping = {'Blue': 'Red', 'Red': 'Blue'}
  
  # Create a new column to represent the opponent side
  alldata['opponent_side'] = alldata['side'].map(opponent_mapping)
  
  # Create a DataFrame with unique gameid and side combinations
  unique_sides = alldata[['gameid', 'side', 'teamname']].drop_duplicates()
  
  # Rename the columns for merging
  unique_sides.columns = ['gameid', 'opponent_side', 'enemyteamname']
  
  # Merge the changes back to the original DataFrame
  alldata = pd.merge(alldata, unique_sides, on=['gameid', 'opponent_side'], how='left')
  
  # Drop the temporary columns
  alldata = alldata.drop(['opponent_side'], axis=1)
  ####################################################
  
  # One hot encode the position column
  alldata_encoded = pd.get_dummies(alldata, columns=['position'], prefix='position')
  
  # Fill rows where kill share is empty because there is no kills in the match
  alldata_encoded.fillna(0, inplace=True)
  
  # Export to CSV
  alldata_encoded.to_csv('data/cleaned_data.csv', index=False)

  print("Cleaned data saved to 'data/cleaned_data.csv'")
  # Return the cleaned data
  return alldata_encoded
  
if __name__ == "__main__":
  clean_data()