from pydantic import BaseSettings, Field


class Config(BaseSettings):
    app_host: str = Field("localhost", env="APP_HOST")
    app_port: int = Field(3900, env="APP_PORT")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    mongo_username: str = Field(None, env="MONGO_USER")
    mongo_password: str = Field(None, env="MONGO_PASS")
    mongo_db_name: str = Field("yosef_project", env="MONGO_DB_NAME")
    output_path: str = Field("product/backend/outputs/retrain", env="OUTPUT_PATH")
    init_model_path: str = Field(
        "notebooks/outputs/sentence_level_preprocess-checkpoints",
        env="INIT_MODEL_PATH",
    )
    embedding_name: str = Field("glove-wiki-gigaword-50", env="EMBEDDING_NAME")

    @property
    def authors_map_path(self):
        return f"{self.output_path}/authors.json"

    @property
    def examples_path(self):
        return f"{self.output_path}/examples.json"

    @property
    def mongo_connection_string(self):
        return f"mongodb://127.0.0.1:27017/{config.mongo_db_name}"

    class Config:
        validate_assignment = True


config = Config()
