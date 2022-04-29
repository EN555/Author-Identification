from fastapi import (
    BackgroundTasks,
    FastAPI,
    Request,
    status,
)
from fastapi.responses import ORJSONResponse

from models.exceptions import ResourceNotFound

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
async def get_dataset_meta(text: str,background_task:BackgroundTasks):
    return f"label {text} "
