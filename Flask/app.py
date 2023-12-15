import sys
sys.path.append(sys.path[0] + '/..')
from flask import Flask, render_template
import pandas as pd
from src.runner import runner

app = Flask(__name__)

# load dataframe
data = runner()

# we're only projecting Map 1 Kills right now so we can get rid of the rest
data = data[data['Prop'] == 'MAP 1 Kills'].drop('Prop', axis=1)

# add difference column
data['Difference'] = data['My Projected Kills'] - data['PrizePicks Line']

# sort the data by absolute diffence
data = data.reindex(data['Difference'].abs().sort_values(ascending=False).index)

# convert the DataFrame to HTML
table_html = data.to_html(classes='table table-bordered', index=False)

# add route
@app.route('/')
def index():
  # render dataframe template
  return render_template('index.html', table_html=table_html)

# Run the app
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)