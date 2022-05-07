import math

import nltk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from wordcloud import WordCloud

from src.config.Constants import TEXT_COLUMN_NAME


def basic_stats(df):
    print("sentences number:\n")
    print(
        df["book_text"]
        .swifter.apply(lambda s: len(nltk.tokenize.sent_tokenize(s)))
        .describe()
    )
    print("\nword(whitespace based) number:\n")
    print(
        df["book_text"].swifter.apply(lambda s: len(s.split(" "))).describe()
    )
    print("\nword(word_tokenize based) number:\n")
    print(
        df["book_text"]
        .swifter.apply(lambda s: len(nltk.tokenize.word_tokenize(s)))
        .describe()
    )

    long_string = ",".join(list(df[TEXT_COLUMN_NAME].values))
    wordcloud = WordCloud(
        background_color="white",
        max_words=5000,
        contour_width=3,
        contour_color="steelblue",
    )
    wordcloud.generate(long_string)
    wordcloud.to_image()


def plot_feature_dist(data: pd.DataFrame, num_in_row: int = 3):
    columns = data.columns
    fig, axs = plt.subplots(
        math.ceil(len(columns) / num_in_row), num_in_row, figsize=(20, 20)
    )
    fig.suptitle("Features Distribution")
    for i, col_name in tqdm(enumerate(columns)):
        try:
            sns.histplot(
                x=data[col_name], ax=axs[i // num_in_row, i % num_in_row]
            )
        except Exception as e:
            print(e)


def get_results(y_test, prediction):
    sns.set_theme()
    ax = plt.subplot()
    mat = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(10, 7))
    sns.heatmap(mat, annot=True, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.draw()
    # save as fig the classification report
    plt.figure(figsize=(10, 7))
    clf_report = classification_report(
        y_test,
        prediction,
        target_names=["AaronPressman", "AlanCrosby"],
        output_dict=True,
    )
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :-2].T, annot=True)
    plt.savefig("classification report.png")
    plt.draw()
