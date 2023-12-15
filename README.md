# Personal Machine Learning Project
This is a project where I want to see if I can accurately project the amount of kills a player will get in their next professional match of League of Legends.
I have a webscraper script that will gather the PrizePicks lines for each player. I have a script that gathered all the data from previous professional matches dating back to 2018, also scripts that clean the data up.
I tried a lot of different models and hyperparameters but for now I am sticking with a Multi Layer Perceptron as it is the one I found to be most sensitive to the win probability; so there is a script that trains this Neural Network model as well.
There is a runner script that will grab and clean the historical data, grab the PrizePicks data, load the trained model, iterate through the data and spit out projections for each player, and then output that data.
Finally, I have a script that will generate a Flask App to view the data that I accumulated. The app will show you a table of the players, the PrizePicks Line for them, my model's prediction, and the difference between the line and prediction (ordered by absolute value).
![Screenshot 2023-12-15 112748](https://github.com/martinezmatt/LoL-Machine-Learning-Betting/assets/140680779/c675c9d2-b5c1-45d6-a059-19046e83bae4)

## Issues as of now
- There is no League of Legends being played until January, so the webscraper won't find anything on PrizePicks until that comes back.
- The model accounts for win probability, but as of right now I set it to .5 for all players until I work out some more details.
- The app works locally with the files I have, but I can't upload the files to github because of the size limit.
- I am planning on running the app online to share with my friends in the future, hopefully will be done soon

To run the app though if you have files:
```
python flask/app.py
```

# IF YOU NEED FILES TO RUN THIS THEN CONTACT ME I CAN SEND THEM TO YOU :smiley:
