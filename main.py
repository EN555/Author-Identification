import pandas as pd
from preprocess import get_data, replace_name, divide_book_to_chucnks
import csv

def main():
    # df = get_data()
    # print(df)
    data = replace_name()
    divide_book_to_chucnks(data)
if __name__ == '__main__':
    main()
