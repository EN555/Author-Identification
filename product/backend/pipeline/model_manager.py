import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import gensim.downloader
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import GRU, AvgPool1D, Dense, Masking
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow import keras

from product.backend.models.config import config
from product.backend.models.exceptions import ModelPathNotExists
from product.backend.models.models import TrainConfig, TrainResult
from product.backend.models.singelton import Singleton
from src.preprocess.word_embedding_features import sentence_level_preprocess


class ModelManager(metaclass=Singleton):
    def __init__(self):
        logging.info("loading embedding table")
        self.embedding_table = gensim.downloader.load(config.embedding_name)
        logging.info(f"loading model {config.init_model_path}")
        self.model = load_model(config.init_model_path)
        with open(config.authors_map_path, "r") as file:
            self.author_mapper = json.load(file)

    def infer(self, text: str) -> str:
        pre_text = sentence_level_preprocess(text, self.embedding_table)
        pred = self.model.predict(np.expand_dims(pre_text, axis=0))
        pred = np.argmax(pred)
        return self.author_mapper[str(pred)]

    def update_model(self, model_path: str, new_authors_map: Dict[str, str]):
        if not Path(model_path).is_dir():
            raise ModelPathNotExists(f"model {model_path} not found")
        self.model = load_model(model_path)
        self.author_mapper = new_authors_map

    @staticmethod
    def build_new_model(num_classes) -> Sequential:
        new_model = Sequential()
        new_model.add(Masking(mask_value=0.0, input_shape=(170, 50),))
        new_model.add(
            GRU(
                100,
                recurrent_dropout=0.2,
                input_shape=(170, 50),
                return_sequences=True,
            )
        )
        new_model.add(AvgPool1D(pool_size=(170,)))
        new_model.add(Dense(num_classes, activation="softmax"))
        new_model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return new_model

    def _preprocess(
        self, df: pd.DataFrame, y_codes: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        one_hot = keras.utils.to_categorical(
            y_codes, num_classes=num_classes, dtype="float32"
        )
        y = np.expand_dims(one_hot, axis=1)
        X = np.stack(
            [
                sentence_level_preprocess(text, self.embedding_table)
                for text in df["text"]
            ],
            axis=0,
        )
        return X, y

    def retrain(
        self, df: pd.DataFrame, model_name: str, train_config: TrainConfig
    ) -> Tuple[TrainResult, Dict[str, str]]:
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        labels = set(df["author_name"].unique()).union(
            set(self.author_mapper.values())
        )
        num_classes = len(labels)
        new_mapper = {
            label_name: str(label_code)
            for label_code, label_name in enumerate(labels)
        }
        y_codes = np.array(
            [int(new_mapper[author_name]) for author_name in df["author_name"]]
        )
        X, y = self._preprocess(df, y_codes, num_classes)
        new_model = self.build_new_model(num_classes)
        new_model.layers[1].set_weights(self.model.layers[0].trainable_weights)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=train_config.test_size
        )
        model_checkpoint_callback = ModelCheckpoint(
            filepath=f"{config.output_path}/{model_name}-checkpoints",
            monitor="loss",
            verbose=0,
            mode="auto",
            save_freq="epoch",
        )

        history = new_model.fit(
            x=X_train,
            y=y_train,
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            callbacks=[model_checkpoint_callback],
            verbose=1,
        )
        train_acc = np.mean(history.history["accuracy"]) * 100
        y_pred = np.argmax(new_model.predict(X_val), axis=-1)
        test_acc = np.sum(np.argmax(y_val, axis=-1) == y_pred)
        test_acc = 100 * (test_acc / y_pred.size)
        logging.info("retrain completed")
        new_model.save(f"{config.output_path}/{model_name}")
        train_result = TrainResult(
            train_acc=train_acc,
            test_acc=test_acc,
            duration=time.time() - start_time,
        )
        return train_result, new_mapper
