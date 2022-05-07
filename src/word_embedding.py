import pickle

import pandas as pd
import tensorflow as tf
from keras.layers import GRU, AvgPool1D, Dense, Masking
from sklearn.model_selection import train_test_split

from src.config.Constants import (
    AUTHOR_NAME_COLUMN_NAME,
    EMBEDDING_SIZE,
    MAX_LENGTH,
    MAX_NUMBER_OF_SENTENCE,
    MAX_SENTENCE_LENGTH,
    NUM_OF_SENTENCE_CHUNK,
    TEST_PART,
    TEXT_COLUMN_NAME,
    VALIDATION_PART,
)
from src.preprocess.common import num_sentences_based_chucking
from src.preprocess.word_embedding_features import sentence_level_preprocess

DATA_PATH = "../data/C50"


def article_level_pipeline(df: pd.DataFrame):
    def forward_pass(x):
        x = Masking(
            mask_value=0.0,
            input_shape=(MAX_NUMBER_OF_SENTENCE, MAX_SENTENCE_LENGTH),
        )(x)
        x = GRU(100, recurrent_dropout=0.2, return_sequences=True)(x)
        x = AvgPool1D(pool_size=(MAX_NUMBER_OF_SENTENCE,))(x)
        return Dense(50, activation="softmax")(x)

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN_NAME], df[AUTHOR_NAME_COLUMN_NAME], test_size=TEST_PART
    )
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    outputs = forward_pass(sentence_level_preprocess(inputs))
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    print(model.summary())
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_PART
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./article-level-checkpoints",
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=30000,
        shuffle=True,
        batch_size=200,
        validation_data=(X_val, y_val),
        callbacks=[callback, model_checkpoint_callback],
    )
    with open("article-level-history", "w") as file:
        pickle.dump(history, file)
    model.save("article-level")


def sentence_level_pipeline(df: pd.DataFrame):
    def forward_pass(x):
        x = Masking(mask_value=0.0, input_shape=(MAX_LENGTH, EMBEDDING_SIZE))(
            x
        )
        x = GRU(100, recurrent_dropout=0.2, return_sequences=True)(x)
        x = AvgPool1D(pool_size=(170,))(x)
        return Dense(50, activation="softmax")(x)

    data = num_sentences_based_chucking(df, NUM_OF_SENTENCE_CHUNK)
    X_train, X_test, y_train, y_test = train_test_split(
        data[TEXT_COLUMN_NAME],
        data[AUTHOR_NAME_COLUMN_NAME],
        test_size=TEST_PART,
    )
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    outputs = forward_pass(sentence_level_preprocess(inputs[0]))
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    print(model.summary())
    return model


# df = merge_datasets()
#
# sentence_level_pipeline(df)
