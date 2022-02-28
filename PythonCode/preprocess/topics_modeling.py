import pandas as pd

from PythonCode.preprocess.simpleStyleFeatures import *
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim.corpora as corpora
import pyLDAvis.gensim
import pyLDAvis
from pprint import pprint
from typing import List, Tuple

nltk.download('stopwords')
stop_words = stopwords.words('english')
additional_ignore_words = ["said"]


def default_preprocess(df) -> (corpora.Dictionary, List[List[Tuple[int, int]]]):
    ignore_words = stop_words + additional_ignore_words

    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)  # removes punctuations

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in ignore_words] for doc in texts]

    data_words = list(sent_to_words(df[TEXT_COLUMN_LABEL].values.tolist()))
    data_words = remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words)
    return id2word, [id2word.doc2bow(text) for text in data_words]


def post_process(topics):
    col_res = []
    for i, topic in enumerate(topics):
        col_res.append(max(topic, key=lambda item: item[1])[0])
    return pd.Series(col_res)


def topic_modeling(x_train: pd.DataFrame, x_test: pd.DataFrame, num_topics: int = 5) -> (pd.Series, pd.Series):
    id2word, corpus = default_preprocess(x_train)
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    pprint(lda_model.print_topics())
    pyLDAvis.enable_notebook()
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, f'./ldavis_prepared_{str(num_topics)}.html')
    print(LDAvis_prepared)
    return post_process(lda_model[corpus]), post_process(lda_model[default_preprocess(x_test)[1]])
