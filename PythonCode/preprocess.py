import os
import sys
from abc import ABC, abstractmethod

import pandas as pd
import swifter
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from typing import List
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def simple_style(x_train, x_test,**args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_train = pd.concat(
        [x_train,
         FeaturesExtraction.pos_count(x_train, text_column_label='book_text'),
         FeaturesExtraction.stop_words(x_train, text_column_label='book_text'),
         FeaturesExtraction.avg_word_len(x_train, text_column_label='book_text'),
         FeaturesExtraction.avg_sentence_len(x_train, text_column_label='book_text'),
         FeaturesExtraction.punctuation_marks(x_train, text_column_label='book_text')], axis=1)
    x_test = pd.concat(
        [x_test,
         FeaturesExtraction.pos_count(x_test, text_column_label='book_text'),
         FeaturesExtraction.stop_words(x_test, text_column_label='book_text'),
         FeaturesExtraction.avg_word_len(x_test, text_column_label='book_text'),
         FeaturesExtraction.avg_sentence_len(x_test, text_column_label='book_text'),
         FeaturesExtraction.punctuation_marks(x_test, text_column_label='book_text')], axis=1)
    return x_train, x_test


def bag_of_words(x_train,x_test,min_df:int,text_column_label):
    vectorizer = CountVectorizer(min_df=min_df)
    vectorizer.fit(x_train[text_column_label])
    x_train = pd.concat([x_train, FeaturesExtraction.bag_of_words(x_train, vectorizer, text_column_label)], axis=1)
    x_test = pd.concat([x_test, FeaturesExtraction.bag_of_words(x_test, vectorizer, text_column_label)], axis=1)
    return x_train,x_test


def preprocess_pipeline(data_path: str, repesention_handler,normalize:bool, test_size: float = 0.3,save_path="./Data/clean/",text_column_label='book_text',min_df=0.001,cache=True) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    if cache:
        pass#TODO
    df = load_data(data_path)
    X, Y = pd.DataFrame(df['book_text'], columns=['book_text']), df['author_name']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    x_train,x_test = repesention_handler(x_train, x_test,min_df,text_column_label)

    x_train, x_test = x_train.drop('book_text', axis=1), x_test.drop('book_text', axis=1)
    if normalize:
        scaler = Normalization.get_standard_scaler(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]

    # turn to correct format
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    x_train.to_csv(os.path.join(data_path,"x_train_clean.csv"))
    x_test.to_csv(os.path.join(data_path,"x_test_clean.csv"))
    y_train.to_csv(os.path.join(data_path,"y_train_clean.csv"))
    y_test.to_csv(os.path.join(data_path,"y_test_clean.csv"))

    return x_train, x_test, y_train, y_test


def load_data(path: str) -> pd.DataFrame:
    rows_list = []
    _, authors, _ = next(os.walk(path))
    for author_name in authors:
        curr_row = {"author_name": author_name}
        author_path = os.path.join(path, author_name)
        _, _, books_files = next(os.walk(author_path))
        for book_name in books_files:
            curr_row["book_name"] = book_name
            with open(os.path.join(author_path, book_name), "r") as book:
                curr_row["book_text"] = book.read()
            rows_list.append(curr_row.copy())
    return pd.DataFrame(rows_list)


class Normalization:
    @staticmethod
    def get_standard_scaler(x_train: pd.DataFrame) -> StandardScaler:
        return StandardScaler().fit(x_train)

    @staticmethod
    def get_min_max_scaler(x_train: pd.DataFrame) -> StandardScaler:
        return MinMaxScaler().fit(x_train)


class FeaturesExtraction:

    @staticmethod
    def __stem_text(text: str):
        ps = PorterStemmer()
        return (ps.stem(word) for word in word_tokenize(text))

    @staticmethod
    def get_bag_of_words_vectorizer(x_train: pd.DataFrame, text_column_label: str = 'Text',
                                    min_df: float = 0.01) -> CountVectorizer:
        vectorizer = CountVectorizer(min_df=min_df, analyzer=FeaturesExtraction.__stem_text)
        return vectorizer.fit(x_train[text_column_label])

    @staticmethod
    def bag_of_words(X: pd.DataFrame, vectorizer: CountVectorizer, text_column_label: str = 'Text') -> pd.DataFrame:
        data = vectorizer.transform(X[text_column_label]).toarray()
        return pd.DataFrame(data=data, columns=vectorizer.get_feature_names_out())

    @staticmethod
    def avg_word_len(df: pd.DataFrame, text_column_label: str = 'Text') -> pd.DataFrame:
        avg_word_len = df[text_column_label].astype(str).swifter.apply(
            lambda s: pd.Series(nltk.word_tokenize(s)).map(len).mean()).rename("avg_word_len")
        return pd.DataFrame(avg_word_len)

    @staticmethod
    def avg_sentence_len(df: pd.DataFrame, text_column_label: str = 'Text') -> pd.DataFrame:
        sentence_count = df[text_column_label].astype(str).swifter.apply(
            lambda text: pd.Series(nltk.sent_tokenize(text)).map(
                lambda sent: len(nltk.word_tokenize(sent))).mean()).rename("avg_sentence_len")

        return pd.DataFrame(sentence_count)

    @staticmethod
    def punctuation_marks(df: pd.DataFrame, text_column_label: str = 'Text') -> pd.DataFrame:
        to_return = pd.DataFrame()
        for mark in list(string.punctuation):
            to_return[mark] = df[text_column_label].astype(str).apply(lambda s: s.count(mark) / len(s))
        return to_return

    @staticmethod
    def stop_words(df: pd.DataFrame, text_column_label: str = 'Text') -> pd.DataFrame:
        to_return = pd.DataFrame()
        for word in list(stopwords.words('english')):
            to_return[word] = df[text_column_label].astype(str).apply(lambda s: s.count(word) / len(s))
        return to_return

    @staticmethod
    def pos_count(df: pd.DataFrame, text_column_label: str = 'Text') -> pd.DataFrame:
        def group_pos(tag):
            groups = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "verb": ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                      "adverb": ['RB', 'RBR', 'RBS'], "adjective": ['JJ', 'JJR', 'JJS']}
            for key, group in groups.items():
                if tag in group:
                    return key
            return None

        features = df[text_column_label].astype(str).swifter.apply(
            lambda s: pd.Series([x[1] for x in nltk.pos_tag(nltk.word_tokenize(s))]).
                apply(group_pos).value_counts(normalize=True).copy())
        features = features.fillna(0)
        return features


