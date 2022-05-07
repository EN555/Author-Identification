import numpy as np
from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.responses import ORJSONResponse
from starlette.middleware.cors import CORSMiddleware
from tensorflow import keras

from product.backend.models.exceptions import ResourceNotFound
from product.backend.models.models import InferData, InferResponse
from src.preprocess.word_embedding_features import sentence_level_preprocess

model = keras.models.load_model(
    "notebooks/outputs/sentence_level_preprocess-checkpoints"
)

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


@app.post(
    "/api/infer", status_code=status.HTTP_200_OK,
)
def infer(body: InferData, background_task: BackgroundTasks):
    pre_text = sentence_level_preprocess(body.text)
    pred = model.predict(np.expand_dims(pre_text, axis=0))
    pred = np.argmax(pred)
    return InferResponse(author_name=f"label {pred}")


@app.post("/api/retrain", status_code=status.HTTP_200_OK)
def retrain():
    pass
