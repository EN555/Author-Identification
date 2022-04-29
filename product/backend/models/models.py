from pydantic import BaseModel


class InferData(BaseModel):
    text: str


class InferResponse(BaseModel):
    author_name: str
