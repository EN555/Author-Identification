import pandas as pd
from preprocess import get_data, replace_name
import csv

def main():
    # df = get_data()
    # print(df)
    data = replace_name()
    print(data.head(10))

if __name__ == '__main__':
    main()
