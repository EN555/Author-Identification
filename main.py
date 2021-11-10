import pandas as pd
from preprocess import get_data
import csv

def main():
    df = get_data()
    print(df)

if __name__ == '__main__':
    main()
