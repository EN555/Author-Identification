from pydantic import BaseSettings, Field


class Config(BaseSettings):
    app_host: str = Field("localhost", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug_mode: bool = Field(False, env="DEBUG_MODE")

    class Config:
        validate_assignment = True


config = Config()