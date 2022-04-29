from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Request,
    status,
)
from fastapi.responses import FileResponse, ORJSONResponse, PlainTextResponse
from fastapi_contrib.tracing.middlewares import OpentracingMiddleware
from fastapi_contrib.tracing.utils import setup_opentracing


app = FastAPI(
    title="Dataset Service API",
    description="Manage DatasetSets and Feedbacks",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    setup_opentracing(app)
    app.add_middleware(OpentracingMiddleware)


@app.exception_handler(ResourceNotFound)
async def resource_not_found_exception_handler(
    request: Request, exc: ResourceNotFound
):
    return ORJSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": f"resource not found! got {exc}"},
    )



@app.get("/healthz")
async def healthz() -> str:
    return "dataset api healthy"
