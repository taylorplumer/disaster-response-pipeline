import json
import plotly
import plotly.graph_objs as go
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Table
from sklearn.externals import joblib
from sqlalchemy import create_engine

from package.sankey import transform_df, create_node_dict, create_link_df, create_nodes_df, create_sankey

app = Flask(__name__)

def tokenize(text):

    """
    Uses nltk to case normalize, lematize, and word tokenize text

    Args:
        text: the natural language message we are analyzing

    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)
report_df = pd.read_csv('../data/report.csv').iloc[:, 1:]

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays data visualizations and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # visualization 1- count by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # visualization 2- results from the classification report

    result_df = report_df

    # visualization 3- Sankey Diagram illustrating links between genres and categories
    transformed_df = transform_df(df)

    node_dict = create_node_dict(transformed_df)

    link_df = create_link_df(transformed_df, node_dict)

    nodes_df = create_nodes_df(node_dict)



    # create visuals
    graphs = [
        {
            'data': [
                Table(
                        header=dict(values=list(result_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
                        cells=dict(values=[result_df.categories[:-4],result_df.f1[:-4], result_df.precision[:-4], result_df.recall[:-4], result_df.support[:-4]],
                        fill_color='lavender',
                        align='left')
                     )
            ],

            'layout': {
                'title': 'Classification Report Table',

            }
            },
            {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                dict(
                    type = 'sankey',
                    orientation = "h",
                    valueformat = ".0f",
                    node = dict(
                        pad = 10,
                        thickness = 30,
                        line = dict(
                        color = "black",
                        width = 0.5
              ),
                label =  nodes_df['Label']
            ),
                link = dict(
                  source = link_df['Source'].dropna(axis=0, how='any'),
                  target = link_df['Target'].dropna(axis=0, how='any'),
                  value = link_df['Value'].dropna(axis=0, how='any'))
                  )
            ],

            'layout': {
                'title': 'Sankey Diagram',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }

        }
       ]
       
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
