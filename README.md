# Disaster Response Pipeline Project

### Summary
This project contains multilingual disaster response messages curated by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). The data has been encoded with 36 different categories related to disaster response. 

The repository contains working code for running an ETL pipeline and ML pipeline and to run a Flask app locally. Instructions are below.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl data/report.csv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Important Files:

1.  data/process_data.py: ETL script to clean and load data into sqlite3 database
2.  models/train_classifier.py: builds, trains, evaluates, and saves machine learning classifier
3.  models/classifier.pkl: file that contains the saved machine learning model
4.  app/run.py: this file is used to run the Flask application


###  Installation
Use latest versions (as of October 8, 2019) for sklearn and plotly

sklearn v0.21.3- ensures that sklearn.metrics.classification report contains parameter output_dict in order to create classification report data table visualization 

plotly v4.1.1- ensures that Table graph object is available
