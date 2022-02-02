from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import pandas as pd
from PythonCode.Constants import *
import nltk


class FeaturesExtraction:
    @staticmethod
    def __stem_text(text: str):
        ps = PorterStemmer()
        return (ps.stem(word) for word in word_tokenize(text))

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
    def pos_count(df: pd.DataFrame) -> pd.DataFrame:
        def group_pos(tag):
            groups = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "verb": ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                      "adverb": ['RB', 'RBR', 'RBS'], "adjective": ['JJ', 'JJR', 'JJS']}
            for key, group in groups.items():
                if tag in group:
                    return key
            return None

        features = df[TEXT_COLUMN_LABEL].astype(str).swifter.apply(
            lambda s: pd.Series([x[1] for x in nltk.pos_tag(nltk.word_tokenize(s))]).
                apply(group_pos).value_counts(normalize=True).copy())
        features = features.fillna(0)
        return features