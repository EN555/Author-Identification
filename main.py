from preprocess import get_data


def main():
    df = get_data()
    df.to_csv("data.csv")
    print(df.head())


if __name__ == '__main__':
    main()
