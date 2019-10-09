
# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import plotly.graph_objs as go
import plotly.offline as pyo

def transform_df(df):

    """
    Transforms dataframe from wide to deep format and renames columns to accord to Sankey graph object terminology
    Args:
        df: dataframe

    Returns:
        df: transformed dataframe

    """

    # drop unnecessary columns for visualization
    df = df.drop(columns = ['message', 'original'])

    # Unpivot the DataFrame from wide format to long format
    df = pd.melt(df, id_vars=df.columns[:2].tolist(), value_vars=df.columns[2:].tolist())

    df = df.groupby(['variable', 'value', 'genre']).count()


    # Return an index of values at the 'value' level in order to filter to only include 1 values
    df = df[df.index.get_level_values('value').isin([1])].iloc[:,:1]

    # flattens hierarchical index and drop unnecessary value column now that there is only one value, 1
    df = pd.DataFrame(df.to_records()).drop(columns=['value'])

    # Renames columns to follow Sankey Diagram node terminology

    df = df.rename(columns={'variable': 'Target', 'genre': 'Source', 'id': 'Value'})

    return df


def create_node_dict(df):
    """
    Create dictionary with key value pairs for node representation
    Source nodes should be placed in sequential order before Target nodes

    Args:
        df: transformed dataframe

    Returns:
        node_dict: dictionary with key value pairs for nodes

    """

    source_list = df.Source.unique().tolist()
    target_list= df.Target.unique().tolist()

    combined_list = source_list+target_list
    node_dict = {}

    for element in combined_list:
        value = combined_list.index(element)
        node_dict[element] = value


    return node_dict

def create_link_df(df, node_dict):

    """
    Creates links dataframe for Sankey Diagram

    Args:
        df: transformed dataframe
        node_dict: dictonary with key value pairs for nodes

    Returns:
        link_df: dataframe that replaces category and genre values with integer node id in node_dict

    """

    link_df = df.replace({"Source": node_dict, "Target": node_dict})

    return link_df

def create_nodes_df(node_dict):
    """
    Creates nodes dataframe for Sankey Diagram

    Args:
        node_dict: dictionary with key value pairs for nodes

    Returns:
        nodes_df: dataframe of nodes represented by their unique ID and Label

    """

    nodes_list = []
    for key, value in node_dict.items():
        items_list = [value, key]
        nodes_list.append(items_list)

    nodes_df = pd.DataFrame(nodes_list, columns = ['ID', 'Label'])

    return nodes_df

def create_sankey(nodes_df, link_df, filename):

    """
    Creates and saves Sankey Diagram to designated file path

    Args:
        nodes_df: dataframe of nodes represented by their unique ID to ensure correct Label is displayed
        link_df: dataframe of links between nodes
        filename: file name for where html output should be saved

    Returns:
        plot_fig: Sankey Diagram Plotly Figure

    """
    data_trace = dict(
        type='sankey',
        #domain = dict( x =  [0,1],    y =  [0,1]    ),
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
          value = link_df['Value'].dropna(axis=0, how='any'),
      )
    )

    layout =  dict(
        title = "Sankey Diagram",
        height = 772,
        width = 950,
        font = dict(
          size = 10
        ),
    )


    fig = dict(data=[data_trace], layout=layout)

    plot_fig = pyo.plot(fig, filename= filename, auto_open=False)

    return plot_fig
