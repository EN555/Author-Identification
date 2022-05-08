import datetime
import logging

import pandas as pd
import shortuuid
from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from starlette.middleware.cors import CORSMiddleware

from product.backend.models.exceptions import ResourceNotFound
from product.backend.models.models import (
    DatasetMeta,
    InferData,
    InferResponse,
    RetrainBody,
    RetrainResponse,
    TrainConfig,
)
from product.backend.pipeline.db_manager import MongoManager
from product.backend.pipeline.model_manager import ModelManager

model_manager = ModelManager()
mongodb_manager = MongoManager()

app = FastAPI(
    title="Infer Service API",
    description="Manage inference of our model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def update_model(model_id: str):
    model_path = mongodb_manager.get_model_by_id(model_id)
    model_manager.update_model(model_path)
    return {"message": f"model update to {model_id} successfully"}


@app.post(
    "/api/infer", status_code=status.HTTP_200_OK,
)
def infer(body: InferData, background_tasks: BackgroundTasks):
    start_time = datetime.datetime.now()
    author_name = model_manager.infer(body.text)
    took_time = datetime.datetime.now() - start_time
    took_time = int(took_time.total_seconds() * 1000)
    background_tasks.add_task(
        mongodb_manager.add_inference,
        dict(text=body.text, author_name=author_name, took_time=took_time),
    )
    return InferResponse(author_name=author_name)


@app.get("/api/inferences", status_code=status.HTTP_200_OK)
def inference_history():
    return mongodb_manager.get_inferences()


@app.get("/api/models", status_code=status.HTTP_200_OK)
def retrain_history():
    return mongodb_manager.get_models()


@app.post("/api/retrain", status_code=status.HTTP_200_OK)
async def retrain(body: RetrainBody):
    logging.info("retrain started")
    model_name = f"retrain/sentence-level-{shortuuid.uuid()}"
    dataset_size = len(body.dataset)
    df = pd.DataFrame(jsonable_encoder(body.dataset))
    batch_size = min(max(dataset_size // 5, 1), 200)
    epochs = int(body.max_time // (batch_size * 0.5))
    train_config = TrainConfig(epochs=epochs, batch_size=batch_size)
    train_result = model_manager.retrain(df, model_name, train_config)
    train_result.train_config = train_config
    new_labels_sizes = df["author_name"].value_counts(normalize=True).to_dict()
    train_result.dataset = DatasetMeta(
        size=dataset_size, new_labels_sizes=new_labels_sizes
    )
    train_result.model_name = model_name
    model_id = mongodb_manager.add_model(train_result)
    retrain_result = RetrainResponse(
        model_id=model_id, train_result=train_result
    )
    return retrain_result
