import logging
import uvicorn

from pipeline.routes import app
from models.config import config as settings

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting backend Service")
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
