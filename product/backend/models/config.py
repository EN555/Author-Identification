from pydantic import BaseSettings, Field


class Config(BaseSettings):
    app_host: str = Field("localhost", env="APP_HOST")
    app_port: int = Field(3900, env="APP_PORT")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    mongo_username: str = Field(None, env="MONGO_USER")
    mongo_password: str = Field(None, env="MONGO_PASS")
    mongo_db_name: str = Field("yosef_project", env="MONGO_DB_NAME")

    @property
    def mongo_connection_string(self):
        return f"mongodb://127.0.0.1:27017/{config.mongo_db_name}"

    class Config:
        validate_assignment = True


config = Config()
