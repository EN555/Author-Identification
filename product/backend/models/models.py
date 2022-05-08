from pydantic import BaseModel


class InferData(BaseModel):
    text: str


class InferResponse(BaseModel):
    author_name: str


class DatasetInstance(BaseModel):
    text: str
    author_name: str


class RetrainResponse(BaseModel):
    train_accuracy: float
    test_accuracy: float
    model_name: str


class ModelConfig(BaseModel):
    model_name: str
