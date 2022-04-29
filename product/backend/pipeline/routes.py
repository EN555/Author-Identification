from fastapi import (
    BackgroundTasks,
    FastAPI,
    Request,
    status,
)
from fastapi.responses import ORJSONResponse
from product.backend.models.exceptions import ResourceNotFound
from src.preprocess.word_embedding_features import sentence_level_preprocess
from tensorflow import keras

model = keras.models.load_model('notebooks/outputs/sentence_level_preprocess-checkpoints')


app = FastAPI(
    title="Infer Service API",
    description="Manage inference of our model",
    version="1.0.0",
)


@app.exception_handler(ResourceNotFound)
async def resource_not_found_exception_handler(
        request: Request, exc: ResourceNotFound
):
    return ORJSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": f"resource not found! got {exc}"},
    )


@app.get(
    "/infer",
    response_class=ORJSONResponse,
    status_code=status.HTTP_200_OK,
)
async def infer(text: str,background_task:BackgroundTasks):
    pre_text = sentence_level_preprocess(text)
    pred = model.predict(pre_text)
    pred
    return f"label {text} "
