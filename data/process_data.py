import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Loads data from csv files and merge

    Args:
        messages_filepath: file path to the messages csv file
        categories_filepath: file path to the categories csv file

    Returns:
        df: merged dataframe  of the messages.csv and categories.csv files

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Cleans the merged dataset

    Args:
        df: merged dataframe that was returned from the load_data function

    Returns:
        clean_df: cleansed dataframe that has transformed merged dataset into a format usable for the model_classifier.py process

    """
    categories = df.categories.str.split(pat = ';', n=36, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # row variable utilized to create a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-1]).tolist()

    # renames columns of `categories`
    categories.columns = category_colnames

    # replaces column value with last charactor value, which should be 0 or 1
    for column in categories:

        categories[column] = categories[column].astype(str).str[-1:]


        categories[column] = pd.to_numeric(categories[column].astype(str).str[-1:])

    # drops original categeories column
    df = df.drop(columns=['categories'], axis=1)

    df_concat = pd.concat([df, categories], axis=1)

    df_dropped_duplicates = df_concat[df_concat.duplicated() != True]

    clean_df = df_dropped_duplicates

    # replace any 2 values with the mode of the column
    binary = [0, 1]
    for column in clean_df:
        mode = clean_df[column].mode()[0]
        non_binary = clean_df[~clean_df[column].isin(binary)][column].tolist()
        clean_df[column] = clean_df[column].replace(non_binary,list(mode for i in range(len(non_binary))) )

    return clean_df


def save_data(df, database_filename):
    """
    Load data to sqlite database

    Args:
        df: cleansed dataframe to load into the database
        database_filename: path to store the database
        
    """
    engine = create_engine('sqlite:///'+ database_filename, echo=False)
    df.to_sql('InsertTableName', engine, if_exists='replace',index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
