from typing import Dict

from pydantic import BaseModel, Field, conlist, constr


class InferData(BaseModel):
    text: constr(min_length=10)


class InferResponse(BaseModel):
    author_name: str


class DatasetInstance(BaseModel):
    text: constr(min_length=10)
    author_name: constr(min_length=1)


class RetrainBody(BaseModel):
    dataset: conlist(DatasetInstance, min_items=10)
    max_time: int = Field(..., gt=0, le=100)


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    test_size: float = 0.2


class DatasetMeta(BaseModel):
    size: int
    new_labels_sizes: Dict[str, int]


class TrainResult(BaseModel):
    train_acc: float = None
    test_acc: float = None
    duration: float = None
    train_config: TrainConfig = None
    dataset: DatasetMeta = None
    model_name: str = None


class RetrainResponse(BaseModel):
    train_result: TrainResult
    model_id: str
