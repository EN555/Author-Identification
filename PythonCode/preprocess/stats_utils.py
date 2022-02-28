import nltk
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PythonCode.Constants import TEXT_COLUMN_LABEL
import seaborn as sns
import math
from tqdm import tqdm


def basic_stats(df):
    print("sentences number:\n")
    print(df["book_text"].swifter.apply(lambda s: len(nltk.tokenize.sent_tokenize(s))).describe())
    print("\nword(whitespace based) number:\n")
    print(df["book_text"].swifter.apply(lambda s: len(s.split(" "))).describe())
    print("\nword(word_tokenize based) number:\n")
    print(df["book_text"].swifter.apply(lambda s: len(nltk.tokenize.word_tokenize(s))).describe())

    long_string = ','.join(list(df[TEXT_COLUMN_LABEL].values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    wordcloud.to_image()


def plot_feature_dist(data: pd.DataFrame, num_in_row: int = 3):
    columns = data.columns
    fig, axs = plt.subplots(math.ceil(len(columns) / num_in_row), num_in_row, figsize=(20, 20))
    fig.suptitle('Features Distribution')
    for i, col_name in tqdm(enumerate(columns)):
        try:
            sns.histplot(x=data[col_name], ax=axs[i // num_in_row, i % num_in_row])
        except Exception as e:
            print(e)