import datetime
import logging
from typing import Any, Dict, Mapping

import pymongo
from bson import ObjectId

from product.backend.models.config import Config
from product.backend.models.exceptions import ResourceNotFound
from product.backend.models.models import TrainResult
from product.backend.models.singelton import Singleton


class MongoManager(metaclass=Singleton):
    INFERENCES = "inferences"
    MODELS = "models"

    def __init__(self):
        self.client = self._get_dataset()

    @staticmethod
    def _get_dataset():
        config = Config()
        client = pymongo.MongoClient(config.mongo_connection_string)
        return client[config.mongo_db_name]

    def add_inference(self, data: dict):
        data["created_at"] = datetime.datetime.now()
        self.client.get_collection("inferences").insert_one(data)

    def add_model(
            self, retrain_result: TrainResult, authors_map: Dict[str, str]
    ) -> str:
        data = retrain_result.dict(exclude_none=True)
        data["created_at"] = datetime.datetime.now()
        data["authors_map"] = authors_map
        result = self.client.get_collection("models").insert_one(data)
        return str(result.inserted_id)

    @staticmethod
    def parse_result(mongo_result: list) -> list:
        result = []
        for elem in mongo_result:
            curr_id = elem.pop("_id")
            result.append(dict(id=str(curr_id), **elem))
        return result

    def get_models(self):
        result = []
        for elem in list(self.client.get_collection("models").find()):
            curr_id = elem.pop("_id")
            elem.pop("authors_map")
            result.append(dict(id=str(curr_id), **elem))
        return result

    def get_inferences(self):
        return self.parse_result(
            list(self.client.get_collection("inferences").find().sort("created_at", -1).limit(100))
        )

    def get_model_by_id(self, model_id: str) -> Mapping[str, Any]:
        try:
            model_res = self.client.get_collection("models").find_one(
                {"_id": ObjectId(model_id)}
            )
            if not model_res:
                raise ResourceNotFound(f"model with id {model_id} not found")
        except Exception as e:
            logging.error(e)
            raise ResourceNotFound(f"invalid id {model_id}")
        return model_res
