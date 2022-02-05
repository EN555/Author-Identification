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


def default_preprocess(df) -> (corpora.Dictionary, List[List[Tuple[int, int]]]):
    ignore_words = stop_words + ["said"]

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


def topic_modeling(df: pd.DataFrame, num_topics: int = 5):
    id2word, corpus = default_preprocess(df)
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    pprint(lda_model.print_topics())
    pyLDAvis.enable_notebook()
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, f'./ldavis_prepared_{str(num_topics)}.html')
    print(LDAvis_prepared)

