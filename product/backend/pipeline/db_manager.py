import pymongo
from product.backend.models.config import Config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MongoManager(metaclass=Singleton):
    def __init__(self):
        self.client = self.get_dataset()

    @staticmethod
    def get_dataset():
        config = Config()
        client = pymongo.MongoClient(config.mongo_connection_string)
        return client[config.mongo_db_name]

    def add_inference(self, text: str, author_name: str):
        self.client.get_collection("inferences").insert_one(
            dict(text=text, author_name=author_name)
        )

    def add_model(self, model_name):
        self.client.get_collection("models").insert_one(
            dict(model_name=model_name)
        )

    def get_models(self):
        return list(self.client.get_collection("models").find())

    def get_inferences(self):
        return list(self.client.get_collection("inferences").find())
