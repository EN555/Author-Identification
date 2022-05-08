import json
import logging
from typing import List

import gensim.downloader
import numpy as np
import pandas as pd
import shortuuid
from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU, AvgPool1D, Dense, Masking
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from starlette.middleware.cors import CORSMiddleware
from tensorflow import keras

from product.backend.models.exceptions import ResourceNotFound
from product.backend.models.models import (
    DatasetInstance,
    InferData,
    InferResponse,
    ModelConfig,
    RetrainResponse,
)
from src.config.Constants import EMBEDDING_SIZE, GLOVE_MODEL_NAME, MAX_LENGTH
from src.preprocess.word_embedding_features import (
    sentence_level_preprocess,
)

embedding_table = gensim.downloader.load(GLOVE_MODEL_NAME)

model = keras.models.load_model(
    "notebooks/outputs/sentence_level_preprocess-checkpoints"
)
with open("authors.json", "r") as file:
    author_mapper = json.load(file)

app = FastAPI(
    title="Infer Service API",
    description="Manage inference of our model",
    version="1.0.0",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ResourceNotFound)
def resource_not_found_exception_handler(
        request: Request, exc: ResourceNotFound
):
    return ORJSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": f"resource not found! got {exc}"},
    )


@app.put(
    "/api/model", status_code=status.HTTP_200_OK,
)
def update_model(body: ModelConfig):
    global model
    model = keras.models.load_model(body.model_name)
    return {"message": f"model update to {body.model_name} successfully"}


@app.post(
    "/api/infer", status_code=status.HTTP_200_OK,
)
def infer(body: InferData):
    pre_text = sentence_level_preprocess(body.text, embedding_table)
    pred = model.predict(np.expand_dims(pre_text, axis=0))
    pred = np.argmax(pred)
    return InferResponse(author_name=author_mapper[str(pred)])


@app.post("/api/retrain", status_code=status.HTTP_200_OK)
async def retrain(body: List[DatasetInstance]):
    logging.info("retrain started")
    model_name = f"retrain/sentence-level-{shortuuid.uuid()}"
    df = pd.DataFrame(jsonable_encoder(body))
    y_diff = set(df["author_name"].unique()).union(set(author_mapper.values()))
    y_codes = pd.Categorical(df["author_name"]).codes
    one_hot = keras.utils.to_categorical(
        y_codes, num_classes=len(y_diff), dtype="float32"
    )
    y = np.expand_dims(one_hot, axis=1)
    X = np.stack(
        [
            sentence_level_preprocess(text, embedding_table)
            for text in df["text"]
        ],
        axis=0,
    )
    new_model = Sequential()
    new_model.add(
        GRU(
            100,
            recurrent_dropout=0.2,
            input_shape=(MAX_LENGTH, 50),
            return_sequences=True,
        )
    )
    new_model.add(AvgPool1D(pool_size=(170,)))
    new_model.add(Dense(50 + 1, activation="softmax"))
    new_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    new_model.layers[0].set_weights(model.layers[0].trainable_weights)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2
    )
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"{model_name}-checkpoints",
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    history = new_model.fit(
        x=X_train,
        y=y_train,
        epochs=3,
        batch_size=100,
        callbacks=[model_checkpoint_callback],
    )
    train_acc = np.mean(history.history["accuracy"])
    y_pred = np.argmax(new_model.predict(X_val))
    test_acc = np.sum(np.argmax(y_val, axis=-1) == y_pred)
    logging.info("retrain completed")
    model.save(model_name)
    return RetrainResponse(model_name=model_name, train_accuracy=train_acc, test_accuracy=test_acc)
