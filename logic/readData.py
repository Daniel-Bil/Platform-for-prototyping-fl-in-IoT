import os

import pandas as pd


def read_data():
    '''
    Reads all csv from driectory as opandas dataframes
    :return: list of all dataFrames
    '''
    file_names = os.listdir("..//dane")
    data = [pd.read_csv(f'..//dane//{file}') for file in file_names]
    return data


if __name__ == "__main__":
    data = read_data()
