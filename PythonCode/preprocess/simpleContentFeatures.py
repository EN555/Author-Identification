import pandas as pd
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import CountVectorizer
import swifter


def name_entity_recognition(df: pd.DataFrame) -> pd.DataFrame:
    NER = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE',
           'TIME', 'PERCENT', 'MONEY' 'QUANTITY', 'ORDINAL', 'CARDINAL']
    name_entity = df["book_text"].swifter.apply(lambda s: pd.Series([x.label_ for x in nlp(s).ents]).value_counts())
    return name_entity.fillna(value=0).reset_index(drop=True)


def noun_phrases_extraction(df: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    phrases = df["book_text"].swifter.apply(
        lambda s: pd.Series(nlp(s).noun_chunks).map(lambda c: c.text.lower()).value_counts())
    return phrases.fillna(value=0).reset_index(drop=True)


def bag_of_words(df: pd.DataFrame, cv: None) -> pd.DataFrame:
    # add count vectorizer object in the function call
    res = cv.transform(df["book_text"])
    return pd.DataFrame(data=res.toarray(), columns=cv.get_feature_names_out()).fillna(value=0)


def simple_content_feature_extraction(x_train: pd.DataFrame, x_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    cv = CountVectorizer(stop_words="english", lowercase=True)
    cv.fit(x_train["book_text"])

    def concat_all(df: pd.DataFrame) -> pd.DataFrame:
        ne = name_entity_recognition(df)
        np = noun_phrases_extraction(df)
        bow = bag_of_words(df, cv)
        res = pd.concat([ne, np, bow], axis=1).fillna(value=0)
        return np

    xs = concat_all(x_train)
    xd = concat_all(x_test)
    x_tmp = xs.loc[:xd.shape[0] - 1, :]
    x_tmp.loc[:, :] = 0
    col_intersec = list(xs.columns.intersection(xd.columns))
    x_tmp[col_intersec] = xd[col_intersec]
    return xs, x_tmp
