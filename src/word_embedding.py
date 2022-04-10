from Constants import *
from src.preprocess.common import merge_datasets
from keras.layers import Dense, GRU, AvgPool1D, Masking
from keras.models import Sequential
from sklearn.metrics import classification_report
from src.preprocess.word_embedding_features import article_level_preprocess,sentence_level_preprocess
from src.models import train_model

DATA_PATH = "../data/C50"


def article_level():
    df = merge_datasets(DATA_PATH)
    X_train, X_test, y_train, y_test = article_level_preprocess(df)
    model = article_level_model()
    train_model(model,X_train,y_train,"article_based_model")
    # eval_model(model,X_test,y_test)
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))


def article_level_model():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(MAX_NUMBER_OF_SENTENCE, MAX_SENTENCE_LENGTH)))
    model.add(GRU(100, recurrent_dropout=0.2, return_sequences=True))
    model.add(AvgPool1D(pool_size=(MAX_NUMBER_OF_SENTENCE,)))
    model.add(Dense(50, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    print(model.summary())
    return model


def sentence_level_model():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(MAX_LENGTH, EMBEDDING_SIZE)))
    model.add(GRU(100, recurrent_dropout=0.2, return_sequences=True))
    model.add(AvgPool1D(pool_size=(170,)))
    model.add(Dense(50, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    print(model.summary())
    return model


def sentence_level():
    df = merge_datasets(DATA_PATH)
    (X_train, y_train), (X_test, y_test) = sentence_level_preprocess(df)
    model = sentence_level_model()
    model = train_model(model, X_train, y_train, "sentence_level_preprocess")
    # eval_model(model, X_test, y_test)

