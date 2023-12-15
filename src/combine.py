import pandas as pd

data = pd.read_csv(r'league/2023LOL.csv')

files = [r"league/2018LOL.csv", r"league/2019LOL.csv", r"league/2020LOL.csv", r"league/2021LOL.csv", r"league/2022LOL.csv"]

for file in files:
  extra = pd.read_csv(file)
  data = pd.concat([data, extra], ignore_index=True)

data.to_csv(r'data/allmatchessince2018.csv', index=False)