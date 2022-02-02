import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from PythonCode.Constants import *


def bag_of_words(x_train, x_test, *args, **kwargs):
    vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(x_train[TEXT_COLUMN_LABEL])
    x_train = pd.DataFrame(columns=vectorizer.get_feature_names(),
                           data=vectorizer.transform(x_train[TEXT_COLUMN_LABEL]).toarray())
    x_test = pd.DataFrame(columns=vectorizer.get_feature_names(),
                          data=vectorizer.transform(x_test[TEXT_COLUMN_LABEL]).toarray())
    return x_train, x_test