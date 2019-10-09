import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings('ignore')

import joblib


def load_data(database_filepath):

    """
    Loads data from sqlite database and creates X,Y variables for model building

    Args:
        database_filename: the path where to connect to the database to load dataframe

    Returns:
        X: the message column of the dataframe
        Y: the 0 and 1 category column values stored in np array
        category_names: labels for model

    """
    # connect to the database
    conn = sqlite3.connect(database_filepath)

    # get a cursor
    cur = conn.cursor()

    # load data from database
    df = pd.read_sql("SELECT * FROM InsertTableName", con=conn)

    conn.commit()
    conn.close()

    # seperate data into X and Y for model building
    X = df['message']
    Y = df.iloc[:, 4:]

    # create labels aka category_names
    category_names = Y.columns


    Y = Y.values

    return X, Y, category_names



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


def build_model():

    """
    Builds pipeline and use grid search to perform multioutputclassification

    Returns:
        cv: GridSearchCV pipeline with best parameters for the model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__min_samples_split': [2,4]

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluates model by providing individual category and summary metrics of model performance

    Args:
        model: MultiOutputClassifier model
        X_test: subset of X values withheld from the model building process
        Y_test: subset of Y values witheld from the model building process and used to evaluate model predictions
        category_names: labels for model

    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)

    """
    y_pred = model.predict(X_test)

    report = classification_report(y_pred, Y_test, target_names= category_names, output_dict=True)

    print(report)


    return report



def save_report(report, report_filepath):

    """
    Loads classification report to csv file

    Args:
        report: classification report returned from evaluate_model function
        report_filepath: path for where to save report

    Returns:
        report_df: save dataframe as a csv at specified file path

    """

    report_df = pd.DataFrame(report).transpose()

    report_df.columns = ['f1', 'precision', 'recall', 'support']

    report_df['categories'] = report_df.index

    report_df = report_df[['categories','f1', 'precision', 'recall', 'support']]

    report_df.to_csv(report_filepath)


    return report_df



def save_model(model, model_filepath):

    """
    Stores model into a pickle file at a designated file path

    Args:
        model: classifier model
        model_filepath: path for where to save pickle file
        
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, report_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        report = evaluate_model(model, X_test, Y_test, category_names)


        print('Saving report...')
        save_report(report, report_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
