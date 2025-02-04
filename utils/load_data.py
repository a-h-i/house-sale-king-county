import pandas as pd

def load_data(path):
    """

    :param path: path to csv file
    :return: pandas dataframe
    """
    return pd.read_csv(path, sep=',')