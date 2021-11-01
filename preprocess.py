import os
import numpy as np
import pandas as pd


def get_data_to_df() -> pd.DataFrame:
    """
    will save the whole data in csv format with
    ["id","text","author"] columns
    :return: result df
    """
    df = pd.DataFrame()
    data_path = "./books1/epubtxt"
    for filename in next(os.walk(data_path))[2]:
        with open(os.path.join(data_path,filename)) as file:
            pass
    return df